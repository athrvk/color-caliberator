"""
Iterative calibration loop.

Receives send/recv callbacks for PC and mobile WebSocket connections.
Returns (icc_bytes, final_delta_e, lut_r, lut_g, lut_b, color_data) when done.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from calibration.capture import (
    decode_frame,
    frame_luminance,
    frame_luminance_linear,
    is_stable,
    srgb_to_linear,
)
from calibration.patches import GRAY_PATCHES, HOLDOUT_PATCHES
from calibration.ramp import (
    compose_luts,
    fit_correction,
    identity_lut,
    lut_to_vcgt,
)
from display.profile import build_vcgt_profile
from display.videolut import apply_ramp_arrays, clear_ramp

SendFn = Callable[[dict], Awaitable[None]]
RecvFn = Callable[[], Awaitable[dict]]
DrainFn = Callable[[], Awaitable[None]]

from calibration.raw import DngAnchor, parse_dng
from calibration.color_pipeline import (
    SRGB_PRIMARIES_XYZ_D50,
    SRGB_TO_XYZ_D50,
    camera_rgb_to_xyz_d50,
    fit_channel_gamma,
    forward_trc,
    project_onto_primary,
)

AnchorWaitFn = Callable[[int], Awaitable[bytes]]


_srgb_to_linear = srgb_to_linear


async def _measure_color_patch(
    level: float,
    channel_idx: int,
    patch_total: int,
    patch_index: int,
    round_num: int,
    white_frame: np.ndarray,
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    recorder: Any | None = None,
) -> np.ndarray | None:
    """Measure one single-channel patch and return measured XYZ_D50, or None.

    The DNG anchors are used by the caller to derive primary XYZ directions
    (camera-linear via ForwardMatrix2); per-patch measurement here works
    entirely from the JPEG preview stream, which is sRGB-encoded.
    """
    rgb = [0, 0, 0]
    rgb[channel_idx] = int(round(level * 255))
    await pc_send({"type": "show_patch", "rgb": rgb})
    await pc_send({"type": "capturing", "round": round_num})
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({"type": "capture", "n": patch_index + 1, "total": patch_total})
    on_raw = None
    if recorder is not None:
        ch_name = "RGB"[channel_idx]
        phase = f"color_{ch_name}"
        def on_raw(jpeg: bytes, _phase=phase, _i=patch_index, _lv=level) -> None:
            recorder.record_patch_raw_frame(_phase, round_num, _i, _lv, jpeg)
    frame = await _wait_for_stable_frames(mobile_recv, on_raw=on_raw)
    await mobile_send({"type": "stop_capture"})
    await pc_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total, "round": round_num})
    await mobile_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total})
    if frame is None:
        if recorder is not None:
            ch_name = "RGB"[channel_idx]
            recorder.record_patch_averaged(
                f"color_{ch_name}", round_num, patch_index, level, None,
                {"level": level, "channel": ch_name, "skipped": True},
            )
        return None

    h, w = frame.shape[:2]
    patch_rgb = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0
    white_rgb = white_frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0

    # JPEG is sRGB-encoded (post-ISP). ForwardMatrix2 from the DNG expects
    # *camera-linear* input (raw sensor space), so applying it to sRGB-linear
    # data double-counts the camera's color matrix. Instead, use the standard
    # sRGB→XYZ_D50 matrix here. The DNG-derived display primaries used in
    # `_run_color_rounds` stay correct (camera-linear DNG sample × FM2 →
    # XYZ_D50), so projection lands in a consistent XYZ_D50 space.
    patch_linear = _srgb_to_linear(patch_rgb)
    white_linear = np.clip(_srgb_to_linear(white_rgb), 1e-6, None)
    relative_srgb_linear = patch_linear / white_linear

    xyz = SRGB_TO_XYZ_D50 @ relative_srgb_linear
    if recorder is not None:
        ch_name = "RGB"[channel_idx]
        recorder.record_patch_averaged(
            f"color_{ch_name}", round_num, patch_index, level, frame,
            {"level": level, "channel": ch_name, "xyz_d50": xyz},
        )
    return xyz


async def _run_color_rounds(
    white_frame: np.ndarray,
    anchors: dict[int, DngAnchor],
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    recorder: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[float, float, float]]:
    """Run color-mode patch loop. Returns (r_trc, g_trc, b_trc, primary_xyz_d50)."""
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
                white_frame,
                pc_send, mobile_send, mobile_recv, mobile_drain,
                recorder=recorder,
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

    gamma_r = fit_channel_gamma(levels, _rescale_to_unit("R"))
    gamma_g = fit_channel_gamma(levels, _rescale_to_unit("G"))
    gamma_b = fit_channel_gamma(levels, _rescale_to_unit("B"))

    # The matrix-shaper ICC stores the *forward* (measured) display TRC so
    # the CMM can invert it during rendering. Returning the pre-warp here
    # would silently produce a profile that double-applies the curve.
    r_trc = forward_trc(gamma_r)
    g_trc = forward_trc(gamma_g)
    b_trc = forward_trc(gamma_b)

    return r_trc, g_trc, b_trc, primary_xyz, (gamma_r, gamma_g, gamma_b)


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
    recorder: Any | None = None,
) -> dict[int, DngAnchor]:
    """Drive the 4-shot manual RAW anchor flow."""
    parsed: dict[int, DngAnchor] = {}
    for seq, label, rgb in _ANCHOR_PROMPTS:
        await pc_send({"type": "show_patch", "rgb": list(rgb)})
        await mobile_send({"type": "request_raw", "seq": seq, "label": label})
        data = await wait_for_anchor(seq)
        dng_path = tmp_dir / f"anchor_{seq}.dng"
        await asyncio.to_thread(dng_path.write_bytes, data)
        if recorder is not None:
            await asyncio.to_thread(recorder.record_anchor, seq, label, data)
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


async def _wait_for_stable_frames(
    mobile_recv: RecvFn,
    on_raw: Callable[[bytes], None] | None = None,
) -> np.ndarray | None:
    """Collect frames until SSNR stable or timeout. Returns averaged frame or None.

    `on_raw`: optional callback invoked with each captured JPEG payload bytes.
    Used by the session recorder to archive every raw frame without coupling
    the capture loop to persistence.
    """
    import base64 as _b64

    luminances: list[float] = []
    frames: list[np.ndarray] = []
    deadline = time.monotonic() + CAPTURE_TIMEOUT

    while time.monotonic() < deadline:
        msg = await asyncio.wait_for(mobile_recv(), timeout=max(0.1, deadline - time.monotonic()))
        if msg.get("type") != "frame":
            continue
        raw_b64 = msg["data"]
        frame = decode_frame(raw_b64)
        if on_raw is not None:
            try:
                on_raw(_b64.b64decode(raw_b64))
            except Exception:
                pass
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
    """Relative luminance: measured / white-reference, in linear-light space.

    Linearizes both frames via sRGB→linear before BT.709 luminance so the
    ratio matches the linear `target_luma = level ** 2.2` model. Skipping
    the linearization makes the fit see measured ≈ level (because camera
    re-encodes the display's x^2.2 output back to sRGB), yielding γ_d ≈ 1
    and a bogus x^2.2 correction that crushes blacks.
    """
    white_luma = frame_luminance_linear(white_frame)
    patch_luma = frame_luminance_linear(frame)
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
    recorder: Any | None = None,
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
    on_raw = recorder.record_white_raw_frame if recorder is not None else None
    try:
        frame = await _wait_for_stable_frames(mobile_recv, on_raw=on_raw)
    finally:
        await mobile_send({"type": "stop_capture"})
    if frame is None:
        raise RuntimeError(
            "White reference capture timed out. Check room lighting and phone position."
        )
    if recorder is not None:
        recorder.record_white_averaged(frame)
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
    recorder: Any | None = None,
    phase: str = "round",
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

    on_raw = None
    if recorder is not None:
        def on_raw(jpeg: bytes, _phase=phase, _round=round_num, _i=patch_index, _lv=patch_level) -> None:
            recorder.record_patch_raw_frame(_phase, _round, _i, _lv, jpeg)
    frame = await _wait_for_stable_frames(mobile_recv, on_raw=on_raw)

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
        if recorder is not None:
            recorder.record_patch_averaged(phase, round_num, patch_index, patch_level, None,
                                           {"level": patch_level, "skipped": True})
        return None
    luma = _normalize_luma(frame, white_frame)
    if recorder is not None:
        recorder.record_patch_averaged(phase, round_num, patch_index, patch_level, frame,
                                       {"level": patch_level, "luma_normalized": luma})
    return luma


async def run_calibration(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    tmp_dir: Path,
    mode: str = "gamma",
    wait_for_anchor: AnchorWaitFn | None = None,
    recorder: Any | None = None,
) -> tuple[bytes, float, np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """
    Run the full iterative calibration session.

    Returns (icc_bytes, final_delta_e, lut_r, lut_g, lut_b, color_data).
    `color_data` is None in gamma mode; in color mode it carries the data the
    server needs to render an honest "after" preview through a software CMM:
    `{r_xyz, g_xyz, b_xyz, gamma_r, gamma_g, gamma_b}`.
    """
    if mode == "color":
        if wait_for_anchor is None:
            raise RuntimeError("color mode requires wait_for_anchor callback")
        anchors = await _capture_anchors(pc_send, mobile_send, wait_for_anchor, tmp_dir, recorder=recorder)
    else:
        anchors = None

    white_frame = await _capture_white_reference(pc_send, mobile_send, mobile_recv, mobile_drain, recorder=recorder)

    if mode == "color":
        r_trc, g_trc, b_trc, primary_xyz, gammas = await _run_color_rounds(
            white_frame, anchors, pc_send, mobile_send, mobile_recv, mobile_drain,
            recorder=recorder,
        )
        from display.profile import build_matrix_shaper_profile
        icc_bytes = build_matrix_shaper_profile(
            r_trc, g_trc, b_trc,
            primary_xyz["R"], primary_xyz["G"], primary_xyz["B"],
        )
        await pc_send({
            "type": "round_done", "round": 1,
            "gamma_r": gammas[0], "gamma_g": gammas[1], "gamma_b": gammas[2],
        })
        await mobile_send({"type": "round_done"})
        # Identity LUTs returned in color mode: the matrix-shaper ICC is the
        # deliverable; per-channel VideoLUT correction can't express primary
        # chromaticity adjustments anyway. Caller skips the live VideoLUT
        # preview and instead renders the chart through display.cmm.
        identity = identity_lut()
        color_data = {
            "r_xyz": primary_xyz["R"],
            "g_xyz": primary_xyz["G"],
            "b_xyz": primary_xyz["B"],
            "gamma_r": float(gammas[0]),
            "gamma_g": float(gammas[1]),
            "gamma_b": float(gammas[2]),
        }
        return icc_bytes, float("nan"), identity, identity.copy(), identity.copy(), color_data

    # Native VideoLUT manipulation via display.videolut backend:
    #   macOS  → CGSetDisplayTransferByTable (CoreGraphics, transient, no admin)
    #   Windows → SetDeviceGammaRamp (GDI, with retry against competing software)
    #   Linux/X11 → dispwin subprocess (XF86VidMode under the hood)
    # All three apply LUTs at session scope; the user's installed ColorSync /
    # system profile is restored on `clear_ramp`.
    lut_r = identity_lut()
    lut_g = identity_lut()
    lut_b = identity_lut()

    best_delta_e = float("inf")
    best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

    for round_num in range(1, MAX_ROUNDS + 1):
        # Apply the accumulated correction to the display before measuring.
        # Round 1: identity LUT (no-op load). Later rounds: the composed
        # per-channel LUTs from previous rounds.
        await asyncio.to_thread(apply_ramp_arrays, lut_r, lut_g, lut_b)

        measured_lumas: list[float] = []
        target_lumas: list[float] = []
        skipped = 0
        patches = GRAY_PATCHES
        total = len(patches)

        for i, patch in enumerate(patches):
            luma = await _measure_patch(
                patch.level, i, total, round_num,
                white_frame, pc_send, mobile_send, mobile_recv, mobile_drain,
                recorder=recorder, phase="round",
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
            recorder=recorder, phase="holdout",
        )
        holdout_measured.append(luma if luma is not None else patch.target_luma)

    holdout_targets = [p.target_luma for p in HOLDOUT_PATCHES]
    final_delta_e = _delta_e_gray(holdout_measured, holdout_targets)

    lut_r, lut_g, lut_b = best_luts
    vcgt_r, vcgt_g, vcgt_b = lut_to_vcgt(lut_r), lut_to_vcgt(lut_g), lut_to_vcgt(lut_b)
    icc_bytes = build_vcgt_profile(vcgt_r, vcgt_g, vcgt_b)

    # Restore the user's installed profile / identity LUT so the display
    # returns to a known state. The final ICC (with VCGT) is what the user
    # installs persistently; applying it is their explicit action.
    await asyncio.to_thread(clear_ramp)

    return icc_bytes, final_delta_e, lut_r, lut_g, lut_b, None
