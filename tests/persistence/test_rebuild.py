import io

import numpy as np
import pytest
from PIL import Image

from persistence.rebuild import RebuildError, rebuild_session
from persistence.sessions import SessionRecorder


def _jpeg_bytes(value: int = 128) -> bytes:
    img = Image.new("RGB", (16, 16), (value, value, value))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _png_bytes(value: int = 128) -> bytes:
    img = Image.new("RGB", (16, 16), (value, value, value))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stage_calibrated_gamma_session(base_dir):
    """Fake a saved gamma session where round-1 measurements match a
    perfectly calibrated display (γ_d ≈ 2.2 → fit_correction returns identity).
    """
    rec = SessionRecorder.create(mode="gamma", base_dir=base_dir)
    rec.record_white_raw_frame(_jpeg_bytes(255))
    rec.record_white_averaged(np.full((16, 16, 3), 255, dtype=np.uint8))

    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, lv in enumerate(levels):
        luma = float(lv ** 2.2) if lv > 0 else 0.0
        rec.record_patch_averaged(
            "round", 1, i, lv,
            np.full((16, 16, 3), int(round(lv * 255)), dtype=np.uint8),
            {"level": lv, "luma_normalized": luma},
        )
    rec.record_result(b"FAKEICC", _png_bytes(50), _png_bytes(200),
                      {"mode": "gamma", "delta_e": 0.0})
    rec.finalize(ok=True, summary={"delta_e": 0.0, "mode": "gamma"})
    return rec.id


def test_rebuild_gamma_calibrated_display_yields_identity(tmp_path):
    sid = _stage_calibrated_gamma_session(tmp_path)
    chart_path = tmp_path / "chart.png"
    Image.fromarray(np.full((20, 80, 3), 128, dtype=np.uint8)).save(chart_path)

    result = rebuild_session(sid, base_dir=tmp_path, chart_path=chart_path)

    summary = result["summary"]
    assert summary["mode"] == "gamma"
    assert summary["rebuilt"] is True
    assert summary["delta_e"] < 0.5  # near-zero — fed perfect measurements
    assert summary["n_patches_used"] == 11

    # ICC is valid (acsp signature at offset 36)
    import base64
    icc = base64.b64decode(result["icc_b64"])
    assert icc[36:40] == b"acsp"


def test_rebuild_missing_session_raises(tmp_path):
    with pytest.raises(RebuildError):
        rebuild_session("nonexistent-id", base_dir=tmp_path)


def test_rebuild_gamma_no_measurements_raises(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.finalize(ok=True, summary={"mode": "gamma"})
    with pytest.raises(RebuildError):
        rebuild_session(rec.id, base_dir=tmp_path)


def _fake_anchor(linear_rgb_sample):
    from calibration.raw import DngAnchor
    return DngAnchor(
        color_matrix_2=np.eye(3),
        forward_matrix_2=np.array([
            [0.4361, 0.3851, 0.1431],
            [0.2225, 0.7169, 0.0606],
            [0.0139, 0.0971, 0.7141],
        ]),
        as_shot_neutral=np.array([1.0, 1.0, 1.0]),
        linear_rgb_sample=linear_rgb_sample,
        calibration_illuminant_2=21,
    )


def test_rebuild_color_session(tmp_path, monkeypatch):
    """End-to-end color rebuild over a synthetic saved session."""
    rec = SessionRecorder.create(mode="color", base_dir=tmp_path)
    rec.record_anchor(0, "WHITE", b"DNG_W")
    rec.record_anchor(1, "RED",   b"DNG_R")
    rec.record_anchor(2, "GREEN", b"DNG_G")
    rec.record_anchor(3, "BLUE",  b"DNG_B")

    anchors_by_path = {
        "seq_0_WHITE.dng": _fake_anchor(np.array([1.0, 1.0, 1.0])),
        "seq_1_RED.dng":   _fake_anchor(np.array([1.0, 0.0, 0.0])),
        "seq_2_GREEN.dng": _fake_anchor(np.array([0.0, 1.0, 0.0])),
        "seq_3_BLUE.dng":  _fake_anchor(np.array([0.0, 0.0, 1.0])),
    }

    def fake_parse_dng(path):
        return anchors_by_path[path.name]
    monkeypatch.setattr("persistence.rebuild.parse_dng", fake_parse_dng)

    # Synthetic XYZ projections for each level: simulate γ_d=2.2 along each
    # primary axis. project_onto_primary(measured, primary) → measured ∝ x^2.2.
    primaries = {
        "R": np.array([0.4361, 0.2225, 0.0139]),
        "G": np.array([0.3851, 0.7169, 0.0606]),
        "B": np.array([0.1431, 0.0606, 0.7141]),
    }
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ch in ("R", "G", "B"):
        for i, lv in enumerate(levels):
            scalar = lv ** 2.2
            xyz = scalar * primaries[ch]
            rec.record_patch_averaged(
                f"color_{ch}", 1, i, lv,
                np.full((8, 8, 3), int(round(lv * 255)), dtype=np.uint8),
                {"level": lv, "channel": ch, "xyz_d50": xyz.tolist()},
            )

    rec.record_result(b"OLD", _png_bytes(), _png_bytes(), {"mode": "color"})
    rec.finalize(ok=True, summary={"mode": "color"})

    chart_path = tmp_path / "chart.png"
    Image.fromarray(np.full((20, 80, 3), 128, dtype=np.uint8)).save(chart_path)

    result = rebuild_session(rec.id, base_dir=tmp_path, chart_path=chart_path)

    s = result["summary"]
    assert s["mode"] == "color"
    assert s["rebuilt"] is True
    # γ=2.2 simulated → fit should recover ~2.2 per channel.
    assert abs(s["color_data"]["gamma_r"] - 2.2) < 0.05
    assert abs(s["color_data"]["gamma_g"] - 2.2) < 0.05
    assert abs(s["color_data"]["gamma_b"] - 2.2) < 0.05

    import base64
    icc = base64.b64decode(result["icc_b64"])
    assert icc[36:40] == b"acsp"
    assert b"rTRC" in icc and b"rXYZ" in icc


def test_rebuild_skipped_patches_are_filtered(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.record_patch_averaged("round", 1, 0, 0.0, None, {"level": 0.0, "skipped": True})
    rec.record_patch_averaged("round", 1, 1, 0.5, None,
                              {"level": 0.5, "luma_normalized": 0.5 ** 2.2})
    rec.record_patch_averaged("round", 1, 2, 1.0, None,
                              {"level": 1.0, "luma_normalized": 1.0})
    rec.record_result(b"X", _png_bytes(), _png_bytes(),
                      {"mode": "gamma", "delta_e": 0})
    rec.finalize(ok=True, summary={"mode": "gamma"})

    chart_path = tmp_path / "chart.png"
    Image.fromarray(np.full((10, 10, 3), 100, dtype=np.uint8)).save(chart_path)
    result = rebuild_session(rec.id, base_dir=tmp_path, chart_path=chart_path)
    assert result["summary"]["n_patches_used"] == 2
