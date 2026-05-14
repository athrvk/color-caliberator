"""
Iterative calibration loop.

Receives send/recv callbacks for PC and mobile WebSocket connections.
Returns (icc_bytes, final_delta_e) when done.
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
) -> tuple[bytes, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full iterative calibration session.
    Returns (icc_bytes, final_delta_e).
    """
    white_frame = await _capture_white_reference(pc_send, mobile_send, mobile_recv, mobile_drain)

    lut_r = identity_lut()
    lut_g = identity_lut()
    lut_b = identity_lut()

    best_delta_e = float("inf")
    best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

    for round_num in range(1, MAX_ROUNDS + 1):
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

        for i, patch in enumerate(patches):
            luma = await _measure_patch(
                patch.level, i, total, round_num,
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
