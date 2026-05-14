"""Rebuild a saved session from its cooked measurements.

Phase 2 of session persistence. Reads `measured.json` files from a saved
session, re-runs the fit functions, builds a fresh ICC, and renders a
fresh before/after preview. Raw frames stay archival — see advisor note
in `sessions.py`. Decoupling rebuild from raw frames keeps it robust to
capture-side changes (averaging algo, frame format, etc.).

Rebuild semantics:
    - Gamma mode: single-pass `fit_correction` over **round 1** measurements
      (round 1 captures the display through identity LUT, so it is the
      cleanest single-shot characterisation). The on-disk result was the
      product of up to 3 iterative rounds, so a rebuild may differ slightly
      — but is reproducible from disk and exposes the post-measurement
      math to changes.
    - Color mode: per-channel `fit_channel_gamma` over saved XYZ_D50
      projections, primaries re-derived from saved DNG anchors.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from calibration.color_pipeline import (
    camera_rgb_to_xyz_d50,
    fit_channel_gamma,
    forward_trc,
    project_onto_primary,
)
from calibration.ramp import fit_correction, lut_to_vcgt, apply_lut_to_image
from calibration.raw import parse_dng
from display.cmm import render_through_profile
from display.profile import build_matrix_shaper_profile, build_vcgt_profile
from persistence.sessions import sessions_dir


class RebuildError(RuntimeError):
    pass


def rebuild_session(
    session_id: str,
    *,
    base_dir: Path | None = None,
    chart_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild profile + preview from a saved session.

    Returns a dict shaped like the GET /api/sessions/<id> payload but with
    freshly computed `icc_b64`, `before_b64`, `after_b64`, and a `summary`
    that reflects the re-derived numbers.
    """
    base = base_dir or sessions_dir()
    root = base / session_id
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise RebuildError(f"session {session_id} not found or incomplete")
    meta = json.loads(meta_path.read_text())
    mode = meta.get("mode", "gamma")

    if chart_path is None:
        chart_path = Path(__file__).resolve().parents[1] / "web" / "static" / "test_chart.png"
    chart = np.array(Image.open(chart_path).convert("RGB"))

    if mode == "color":
        icc_bytes, before_png, after_png, summary = _rebuild_color(root, chart)
    else:
        icc_bytes, before_png, after_png, summary = _rebuild_gamma(root, chart)

    return {
        "meta": meta,
        "summary": summary,
        "icc_b64": base64.b64encode(icc_bytes).decode(),
        "before_b64": base64.b64encode(before_png).decode(),
        "after_b64": base64.b64encode(after_png).decode(),
    }


# ─────────────────────────────────────────────────────────────────────────────


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _read_round_measurements(round_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (levels, normalized_lumas) for non-skipped patches, sorted."""
    if not round_dir.exists():
        raise RebuildError(f"missing round directory: {round_dir}")
    rows: list[tuple[float, float]] = []
    for patch in sorted(round_dir.iterdir()):
        mpath = patch / "measured.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        if m.get("skipped"):
            continue
        if "luma_normalized" not in m:
            continue
        rows.append((float(m["level"]), float(m["luma_normalized"])))
    if not rows:
        raise RebuildError(f"no usable measurements under {round_dir}")
    rows.sort(key=lambda r: r[0])
    levels = np.array([r[0] for r in rows], dtype=float)
    lumas = np.array([r[1] for r in rows], dtype=float)
    return levels, lumas


def _rebuild_gamma(root: Path, chart: np.ndarray) -> tuple[bytes, bytes, bytes, dict]:
    round1 = root / "patches" / "round_1"
    levels, lumas = _read_round_measurements(round1)
    lut = fit_correction(levels, lumas)

    targets = np.where(levels > 0, levels ** 2.2, 0.0)
    delta_e = float(np.mean(np.abs(lumas - targets)) * 100.0)

    vcgt_r = lut_to_vcgt(lut)
    icc_bytes = build_vcgt_profile(vcgt_r, vcgt_r.copy(), vcgt_r.copy())

    before = chart
    after = apply_lut_to_image(chart, lut, lut, lut)
    summary = {
        "mode": "gamma",
        "delta_e": delta_e,
        "n_patches_used": int(len(levels)),
        "rebuilt": True,
    }
    return icc_bytes, _png_bytes(before), _png_bytes(after), summary


# ─────────────────────────────────────────────────────────────────────────────


_ANCHOR_LABEL_TO_SEQ = {"WHITE": 0, "RED": 1, "GREEN": 2, "BLUE": 3}


def _load_anchors(root: Path) -> dict[int, Any]:
    anchors_dir = root / "anchors"
    if not anchors_dir.exists():
        raise RebuildError("color session missing anchors/ directory")
    parsed: dict[int, Any] = {}
    for f in anchors_dir.glob("seq_*.dng"):
        try:
            seq = int(f.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        parsed[seq] = parse_dng(f)
    for needed in (0, 1, 2, 3):
        if needed not in parsed:
            raise RebuildError(f"color session missing anchor seq {needed}")
    return parsed


def _read_color_channel(channel_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (levels, projections-or-xyz). Caller projects onto primary."""
    if not channel_dir.exists():
        raise RebuildError(f"missing color channel dir: {channel_dir}")
    rows: list[tuple[float, np.ndarray]] = []
    for patch in sorted(channel_dir.iterdir()):
        mpath = patch / "measured.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        if m.get("skipped") or "xyz_d50" not in m:
            continue
        rows.append((float(m["level"]), np.array(m["xyz_d50"], dtype=float)))
    if not rows:
        raise RebuildError(f"no usable measurements under {channel_dir}")
    rows.sort(key=lambda r: r[0])
    levels = np.array([r[0] for r in rows], dtype=float)
    xyzs = np.stack([r[1] for r in rows], axis=0)
    return levels, xyzs


def _rebuild_color(root: Path, chart: np.ndarray) -> tuple[bytes, bytes, bytes, dict]:
    anchors = _load_anchors(root)
    primary_xyz = {}
    for ch_name, seq in (("R", 1), ("G", 2), ("B", 3)):
        a = anchors[seq]
        primary_xyz[ch_name] = camera_rgb_to_xyz_d50(
            a.linear_rgb_sample, a.as_shot_neutral, a.forward_matrix_2,
        )

    gammas: dict[str, float] = {}
    for ch_name in ("R", "G", "B"):
        ch_dir = root / "patches" / "color" / ch_name
        levels, xyzs = _read_color_channel(ch_dir)
        projections = np.array([
            project_onto_primary(xyzs[i], primary_xyz[ch_name]) for i in range(len(levels))
        ])
        ref = max(projections[-1], 1e-6)
        projections = projections / ref
        gammas[ch_name] = fit_channel_gamma(levels, projections)

    icc_bytes = build_matrix_shaper_profile(
        forward_trc(gammas["R"]), forward_trc(gammas["G"]), forward_trc(gammas["B"]),
        primary_xyz["R"], primary_xyz["G"], primary_xyz["B"],
    )

    # Software CMM render — same code path the live result uses.
    after = render_through_profile(
        chart,
        primary_xyz["R"], primary_xyz["G"], primary_xyz["B"],
        gammas["R"], gammas["G"], gammas["B"],
    )
    summary = {
        "mode": "color",
        "delta_e": None,
        "color_data": {
            "gamma_r": float(gammas["R"]),
            "gamma_g": float(gammas["G"]),
            "gamma_b": float(gammas["B"]),
            "r_xyz": primary_xyz["R"].tolist(),
            "g_xyz": primary_xyz["G"].tolist(),
            "b_xyz": primary_xyz["B"].tolist(),
        },
        "rebuilt": True,
    }
    return icc_bytes, _png_bytes(chart), _png_bytes(after), summary
