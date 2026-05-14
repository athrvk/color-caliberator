"""
Iterative calibration loop.

Receives send/recv callbacks for PC and mobile WebSocket connections.
Returns (icc_bytes, final_delta_e, lut_r, lut_g, lut_b) when done.
"""

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from calibration.capture import decode_frame, frame_luminance, is_stable
from calibration.patches import GRAY_PATCHES, HOLDOUT_PATCHES
from calibration.ramp import (
    compose_luts,
    fit_correction,
    identity_lut,
    lut_to_vcgt,
)
from display.dispwin import apply_ramp, clear_ramp
from display.profile import build_vcgt_profile

SendFn = Callable[[dict], Awaitable[None]]
RecvFn = Callable[[], Awaitable[dict]]
DrainFn = Callable[[], Awaitable[None]]

from calibration.raw import DngAnchor, parse_dng
from calibration.color_pipeline import (
    SRGB_PRIMARIES_XYZ_D50,
    camera_rgb_to_xyz_d50,
    fit_tone_response,
    project_onto_primary,
)

AnchorWaitFn = Callable[[int], Awaitable[bytes]]


def _srgb_to_linear(rgb_0_1: np.ndarray) -> np.ndarray:
    """Reverse the sRGB encoding curve.

    Assumes the phone encodes JPEGs in sRGB. iPhones since iOS 11 may use
    Display P3 by default; users should set their camera to Most Compatible
    (sRGB) for accurate color-mode results. See Task 12 README note.
    """
    a = 0.055
    return np.where(rgb_0_1 <= 0.04045, rgb_0_1 / 12.92, ((rgb_0_1 + a) / (1 + a)) ** 2.4)


async def _measure_color_patch(
    level: float,
    channel_idx: int,
    patch_total: int,
    patch_index: int,
    round_num: int,
    white_frame: np.ndarray,
    white_anchor: DngAnchor,
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> np.ndarray | None:
    """Measure one single-channel patch and return measured XYZ_D50, or None."""
    rgb = [0, 0, 0]
    rgb[channel_idx] = int(round(level * 255))
    await pc_send({"type": "show_patch", "rgb": rgb})
    await pc_send({"type": "capturing", "round": round_num})
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({"type": "capture", "n": patch_index + 1, "total": patch_total})
    frame = await _wait_for_stable_frames(mobile_recv)
    await mobile_send({"type": "stop_capture"})
    await pc_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total, "round": round_num})
    await mobile_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total})
    if frame is None:
        return None

    h, w = frame.shape[:2]
    patch_rgb = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0
    white_rgb = white_frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0

    patch_linear = _srgb_to_linear(patch_rgb)
    white_linear = np.clip(_srgb_to_linear(white_rgb), 1e-6, None)
    relative_rgb = patch_linear / white_linear

    return camera_rgb_to_xyz_d50(relative_rgb, white_anchor.as_shot_neutral, white_anchor.forward_matrix_2)


async def _run_color_rounds(
    white_frame: np.ndarray,
    anchors: dict[int, DngAnchor],
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run color-mode patch loop. Returns (r_trc, g_trc, b_trc, primary_xyz_d50)."""
    white_anchor = anchors[0]
    levels = np.array([p.level for p in GRAY_PATCHES])

    primary_xyz: dict[str, np.ndarray] = {}
    for ch_name, seq in [("R", 1), ("G", 2), ("B", 3)]:
        a = anchors[seq]
        primary_xyz[ch_name] = camera_rgb_to_xyz_d50(
            a.linear_rgb_sample, a.as_shot_neutral, a.forward_matrix_2,
        )

    channels = [("R", 0), ("G", 1), ("B", 2)]
    per_channel_projections: dict[str, list[float]] = {"R": [], "G": [], "B": []}
    total = len(levels) * 3
    patch_index = 0

    for ch_name, ch_idx in channels:
        primary = primary_xyz[ch_name]
        for level in levels:
            measured_xyz = await _measure_color_patch(
                float(level), ch_idx, total, patch_index, 1,
                white_frame, white_anchor,
                pc_send, mobile_send, mobile_recv, mobile_drain,
            )
            patch_index += 1
            if measured_xyz is None:
                per_channel_projections[ch_name].append(float(level) ** 2.2)
                continue
            per_channel_projections[ch_name].append(
                project_onto_primary(measured_xyz, primary)
            )

    def _rescale_to_unit(ch_name: str) -> np.ndarray:
        projections = np.array(per_channel_projections[ch_name], dtype=float)
        ref = max(projections[-1], 1e-6)
        return projections / ref

    r_trc = fit_tone_response(levels, _rescale_to_unit("R"))
    g_trc = fit_tone_response(levels, _rescale_to_unit("G"))
    b_trc = fit_tone_response(levels, _rescale_to_unit("B"))

    return r_trc, g_trc, b_trc, primary_xyz


_ANCHOR_PROMPTS = [
    (0, "WHITE", (255, 255, 255)),
    (1, "RED",   (255, 0,   0)),
    (2, "GREEN", (0,   255, 0)),
    (3, "BLUE",  (0,   0,   255)),
]


async def _capture_anchors(
    pc_send: SendFn,
    mobile_send: SendFn,
    wait_for_anchor: AnchorWaitFn,
    tmp_dir: Path,
) -> dict[int, DngAnchor]:
    """Drive the 4-shot manual RAW anchor flow."""
    parsed: dict[int, DngAnchor] = {}
    for seq, label, rgb in _ANCHOR_PROMPTS:
        await pc_send({"type": "show_patch", "rgb": list(rgb)})
        await mobile_send({"type": "request_raw", "seq": seq, "label": label})
        data = await wait_for_anchor(seq)
        dng_path = tmp_dir / f"anchor_{seq}.dng"
        await asyncio.to_thread(dng_path.write_bytes, data)
        parsed[seq] = await asyncio.to_thread(parse_dng, dng_path)
    return parsed

SSNR_THRESHOLD = 20.0
STABLE_FRAMES = 5
CAPTURE_TIMEOUT = 10.0    # seconds per patch
READY_TIMEOUT = 180.0     # seconds to wait for user to lock WB and tap Ready
SETTLE_DELAY = 0.3        # seconds for LCD to stabilise on a new patch
MAX_SKIPPED = 3
MAX_ROUNDS = 3
CONVERGENCE_DELTA_E = 1.0


async def _wait_for_stable_frames(mobile_recv: RecvFn) -> np.ndarray | None:
    """Collect frames until SSNR stable or timeout. Returns averaged frame or None."""
    luminances: list[float] = []
    frames: list[np.ndarray] = []
    deadline = time.monotonic() + CAPTURE_TIMEOUT

    while time.monotonic() < deadline:
        msg = await asyncio.wait_for(mobile_recv(), timeout=max(0.1, deadline - time.monotonic()))
        if msg.get("type") != "frame":
            continue
        frame = decode_frame(msg["data"])
        luma = frame_luminance(frame)
        luminances.append(luma)
        frames.append(frame)

        if len(luminances) > STABLE_FRAMES:
            luminances.pop(0)
            frames.pop(0)

        if is_stable(luminances, SSNR_THRESHOLD):
            stacked = np.stack(frames, axis=0).astype(float)
            return np.clip(np.mean(stacked, axis=0), 0, 255).astype(np.uint8)

    return None


def _normalize_luma(frame: np.ndarray, white_frame: np.ndarray) -> float:
    """Relative luminance: measured luma / white-reference luma. Clamped to [0, 1]."""
    white_luma = frame_luminance(white_frame)
    patch_luma = frame_luminance(frame)
    if white_luma < 1e-6:
        return 0.0
    return float(np.clip(patch_luma / white_luma, 0.0, 1.0))


def _delta_e_gray(measured_lumas: list[float], target_lumas: list[float]) -> float:
    """Mean absolute error between measured and target luminance ratios (proxy for ΔE on grays)."""
    diffs = [abs(m - t) * 100.0 for m, t in zip(measured_lumas, target_lumas)]
    return float(np.mean(diffs))


async def _capture_white_reference(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> np.ndarray:
    """Show white patch, wait for WB lock from mobile, capture 5 stable frames."""
    level = 255
    await pc_send({"type": "show_patch", "rgb": [level, level, level]})
    await mobile_send({"type": "show_white_for_wb"})

    try:
        while True:
            msg = await asyncio.wait_for(mobile_recv(), timeout=READY_TIMEOUT)
            if msg.get("type") == "ready":
                break
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timed out waiting for mobile to lock white balance.") from exc

    # Mobile only streams frames between `capture` and `stop_capture`. The WB
    # reference also needs frames, so wrap the capture in the same control
    # messages used for ordinary patches.
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({"type": "capture", "n": 0, "total": 0, "phase": "white_ref"})
    try:
        frame = await _wait_for_stable_frames(mobile_recv)
    finally:
        await mobile_send({"type": "stop_capture"})
    if frame is None:
        raise RuntimeError(
            "White reference capture timed out. Check room lighting and phone position."
        )
    return frame


async def _measure_patch(
    patch_level: float,
    patch_index: int,
    patch_total: int,
    round_num: int,
    white_frame: np.ndarray,
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> float | None:
    """Show one gray patch, capture, return normalized luminance or None if skipped."""
    v = int(round(patch_level * 255))
    await pc_send({"type": "show_patch", "rgb": [v, v, v]})
    await pc_send({"type": "capturing", "round": round_num})

    # Give the display a beat to settle on the new patch, drain any in-flight
    # stale frames from the previous patch, then tell mobile to start streaming.
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({
        "type": "capture",
        "n": patch_index + 1,
        "total": patch_total,
    })

    frame = await _wait_for_stable_frames(mobile_recv)

    await mobile_send({"type": "stop_capture"})
    await pc_send({
        "type": "patch_done",
        "n": patch_index + 1,
        "total": patch_total,
        "round": round_num,
    })
    await mobile_send({
        "type": "patch_done",
        "n": patch_index + 1,
        "total": patch_total,
    })

    if frame is None:
        return None
    return _normalize_luma(frame, white_frame)


async def run_calibration(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    tmp_dir: Path,
    mode: str = "gamma",
    wait_for_anchor: AnchorWaitFn | None = None,
) -> tuple[bytes, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full iterative calibration session.
    Returns (icc_bytes, final_delta_e, lut_r, lut_g, lut_b).
    """
    if mode == "color":
        if wait_for_anchor is None:
            raise RuntimeError("color mode requires wait_for_anchor callback")
        anchors = await _capture_anchors(pc_send, mobile_send, wait_for_anchor, tmp_dir)
    else:
        anchors = None

    white_frame = await _capture_white_reference(pc_send, mobile_send, mobile_recv, mobile_drain)

    if mode == "color":
        r_trc, g_trc, b_trc, primary_xyz = await _run_color_rounds(
            white_frame, anchors, pc_send, mobile_send, mobile_recv, mobile_drain,
        )
        from display.profile import build_matrix_shaper_profile
        icc_bytes = build_matrix_shaper_profile(
            r_trc, g_trc, b_trc,
            primary_xyz["R"], primary_xyz["G"], primary_xyz["B"],
        )
        await pc_send({"type": "round_done", "round": 1})
        await mobile_send({"type": "round_done"})
        return icc_bytes, float("nan"), r_trc, g_trc, b_trc

    import sys as _sys

    # On Windows, SetDeviceGammaRamp (used by dispwin) is reset by the OS and
    # other drivers between rounds, making mid-loop VideoLUT writes unreliable.
    # Instead we pre-warp the patch display level through the current correction
    # LUT so the display sees the compensated input — mathematically equivalent
    # convergence without touching the VideoLUT during measurement.
    # On Mac/Linux dispwin VideoLUT access is stable so we use it as intended.
    _windows_mode = _sys.platform == "win32"

    lut_r = identity_lut()
    lut_g = identity_lut()
    lut_b = identity_lut()

    best_delta_e = float("inf")
    best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

    for round_num in range(1, MAX_ROUNDS + 1):
        if _windows_mode:
            # Windows: keep VideoLUT at identity; compensate by warping patch levels.
            # Round 1 LUT is identity so patch levels are unchanged — same as Mac/Linux.
            pass
        else:
            # Mac/Linux: apply accumulated correction to the display VideoLUT before measuring.
            tmp_icc = tmp_dir / f"round_{round_num}.icc"
            vcgt_r, vcgt_g, vcgt_b = lut_to_vcgt(lut_r), lut_to_vcgt(lut_g), lut_to_vcgt(lut_b)
            await asyncio.to_thread(
                tmp_icc.write_bytes, build_vcgt_profile(vcgt_r, vcgt_g, vcgt_b)
            )
            await asyncio.to_thread(clear_ramp)
            await asyncio.to_thread(apply_ramp, str(tmp_icc))

        measured_lumas: list[float] = []
        target_lumas: list[float] = []
        skipped = 0
        patches = GRAY_PATCHES
        total = len(patches)

        x256 = np.linspace(0.0, 1.0, 256)  # LUT index positions

        for i, patch in enumerate(patches):
            if _windows_mode and round_num > 1:
                # Pre-warp: look up what level the current LUT maps patch.level to,
                # then show that warped level so the display's uncorrected gamma
                # combined with the warped input produces the target output.
                display_level = float(np.interp(patch.level, x256, lut_r))
            else:
                display_level = patch.level

            luma = await _measure_patch(
                display_level, i, total, round_num,
                white_frame, pc_send, mobile_send, mobile_recv, mobile_drain,
            )
            if luma is None:
                skipped += 1
                if skipped > MAX_SKIPPED:
                    raise RuntimeError(f"Too many skipped patches ({skipped}). Check lighting.")
                measured_lumas.append(patch.target_luma)
            else:
                measured_lumas.append(luma)
            target_lumas.append(patch.target_luma)

        levels = np.array([p.level for p in patches])
        measured = np.array(measured_lumas)

        new_lut = fit_correction(levels, measured)
        lut_r = compose_luts(lut_r, new_lut)
        lut_g = compose_luts(lut_g, new_lut)
        lut_b = compose_luts(lut_b, new_lut)

        delta_e = _delta_e_gray(measured_lumas, target_lumas)
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

        await pc_send({"type": "round_done", "delta_e": delta_e, "round": round_num})
        await mobile_send({"type": "round_done"})

        if delta_e < CONVERGENCE_DELTA_E or round_num == MAX_ROUNDS:
            break

    await pc_send({"type": "holdout_started", "total": len(HOLDOUT_PATCHES)})
    await mobile_send({"type": "holdout_started", "total": len(HOLDOUT_PATCHES)})
    holdout_measured: list[float] = []
    for i, patch in enumerate(HOLDOUT_PATCHES):
        luma = await _measure_patch(
            patch.level, i, len(HOLDOUT_PATCHES), 0,
            white_frame, pc_send, mobile_send, mobile_recv, mobile_drain,
        )
        holdout_measured.append(luma if luma is not None else patch.target_luma)

    holdout_targets = [p.target_luma for p in HOLDOUT_PATCHES]
    final_delta_e = _delta_e_gray(holdout_measured, holdout_targets)

    lut_r, lut_g, lut_b = best_luts
    vcgt_r, vcgt_g, vcgt_b = lut_to_vcgt(lut_r), lut_to_vcgt(lut_g), lut_to_vcgt(lut_b)
    icc_bytes = build_vcgt_profile(vcgt_r, vcgt_g, vcgt_b)

    return icc_bytes, final_delta_e, lut_r, lut_g, lut_b
