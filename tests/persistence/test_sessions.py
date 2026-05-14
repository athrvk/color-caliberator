import io
import json

import numpy as np
import pytest
from PIL import Image

from persistence.sessions import (
    SessionRecorder,
    delete_session,
    list_sessions,
    load_session,
    new_session_id,
)


def _jpeg_bytes(value: int = 128) -> bytes:
    img = Image.new("RGB", (32, 32), (value, value, value))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _png_bytes(value: int = 200) -> bytes:
    img = Image.new("RGB", (32, 32), (value, value, value))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_new_session_id_is_sortable_and_unique():
    a = new_session_id()
    b = new_session_id()
    assert a != b
    # YYYYMMDD-HHMMSS-hex6
    assert len(a) == len("20260514-185600-aabbcc")
    assert a[8] == "-" and a[15] == "-"


def test_recorder_writes_only_finalized_sessions(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.record_white_raw_frame(_jpeg_bytes())
    rec.record_white_averaged(np.full((16, 16, 3), 200, dtype=np.uint8))

    # Pre-finalize: no meta.json yet, so list_sessions skips it.
    assert list_sessions(tmp_path) == []

    rec.finalize(ok=True, summary={"delta_e": 1.23, "mode": "gamma"})

    sessions = list_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["id"] == rec.id
    assert sessions[0]["ok"] is True
    assert sessions[0]["delta_e"] == 1.23
    assert sessions[0]["size_bytes"] > 0


def test_recorder_records_full_gamma_flow(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.record_white_raw_frame(_jpeg_bytes())
    rec.record_white_averaged(np.full((16, 16, 3), 200, dtype=np.uint8))

    avg = np.full((16, 16, 3), 100, dtype=np.uint8)
    rec.record_patch_raw_frame("round", 1, 0, 0.5, _jpeg_bytes())
    rec.record_patch_raw_frame("round", 1, 0, 0.5, _jpeg_bytes(120))
    rec.record_patch_averaged("round", 1, 0, 0.5, avg, {"level": 0.5, "luma_normalized": 0.21})
    rec.record_patch_averaged("holdout", None, 0, 0.25, avg, {"level": 0.25, "luma_normalized": 0.05})

    rec.record_result(b"FAKEICC", _png_bytes(50), _png_bytes(200),
                      {"mode": "gamma", "delta_e": 0.9})
    rec.finalize(ok=True, summary={"delta_e": 0.9, "mode": "gamma"})

    sessions = list_sessions(tmp_path)
    assert len(sessions) == 1
    sid = sessions[0]["id"]

    # On-disk structure spot checks.
    root = tmp_path / sid
    raw_files = list((root / "white_ref" / "raw").glob("*.jpg"))
    assert len(raw_files) == 1
    assert (root / "white_ref" / "averaged.jpg").exists()

    patch_dir = root / "patches" / "round_1" / "patch_00_level_0.50"
    assert (patch_dir / "averaged.jpg").exists()
    assert len(list((patch_dir / "raw").glob("*.jpg"))) == 2
    measured = json.loads((patch_dir / "measured.json").read_text())
    assert measured["luma_normalized"] == 0.21

    assert (root / "patches" / "holdout" / "patch_00_level_0.25").exists()
    assert (root / "result" / "profile.icc").read_bytes() == b"FAKEICC"


def test_load_session_returns_payload(tmp_path):
    rec = SessionRecorder.create(mode="color", base_dir=tmp_path)
    rec.record_result(b"ICC", _png_bytes(40), _png_bytes(210),
                      {"mode": "color", "delta_e": None,
                       "color_data": {"gamma_r": 2.18, "gamma_g": 2.21, "gamma_b": 2.20}})
    rec.finalize(ok=True, summary={"mode": "color",
                                    "color_data": {"gamma_r": 2.18, "gamma_g": 2.21, "gamma_b": 2.20}})

    data = load_session(rec.id, base_dir=tmp_path)
    assert data is not None
    assert data["meta"]["mode"] == "color"
    assert data["icc_bytes"] == b"ICC"
    assert len(data["before_png"]) > 0
    assert len(data["after_png"]) > 0
    assert data["summary"]["color_data"]["gamma_r"] == 2.18


def test_load_session_missing_returns_none(tmp_path):
    assert load_session("nonexistent-id", base_dir=tmp_path) is None


def test_delete_session_removes_directory(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.finalize(ok=True, summary={})
    assert (tmp_path / rec.id).exists()
    assert delete_session(rec.id, base_dir=tmp_path) is True
    assert not (tmp_path / rec.id).exists()
    assert delete_session(rec.id, base_dir=tmp_path) is False


def test_recorder_anchor_records_only_color(tmp_path):
    rec = SessionRecorder.create(mode="color", base_dir=tmp_path)
    rec.record_anchor(0, "WHITE", b"DNGBYTES")
    rec.finalize(ok=True, summary={"mode": "color"})
    sid = rec.id
    anchor_files = list((tmp_path / sid / "anchors").glob("*.dng"))
    assert len(anchor_files) == 1
    assert anchor_files[0].name == "seq_0_WHITE.dng"


def test_listings_skip_corrupt_meta(tmp_path):
    rec = SessionRecorder.create(mode="gamma", base_dir=tmp_path)
    rec.finalize(ok=True, summary={})
    # Corrupt meta.json.
    (tmp_path / rec.id / "meta.json").write_text("not json")
    assert list_sessions(tmp_path) == []
