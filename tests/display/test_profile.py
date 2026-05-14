import struct

import numpy as np

from display.profile import build_vcgt_profile


def _identity_luts():
    lut = (np.linspace(0, 1, 256) * 65535).astype(np.uint16)
    return lut, lut.copy(), lut.copy()


def test_returns_bytes():
    r, g, b = _identity_luts()
    assert isinstance(build_vcgt_profile(r, g, b), bytes)


def test_declared_size_matches_actual_length():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    declared = struct.unpack(">I", data[:4])[0]
    assert declared == len(data)


def test_acsp_signature_at_offset_36():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[36:40] == b"acsp"


def test_display_class_at_offset_12():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[12:16] == b"mntr"


def test_rgb_colorspace_at_offset_16():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[16:20] == b"RGB "


def test_vcgt_tag_present_in_data():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert b"vcgt" in data


def _vcgt_offset(data: bytes) -> int:
    """Walk the ICC tag table and return the byte offset of the vcgt type block."""
    count = struct.unpack(">I", data[128:132])[0]
    for i in range(count):
        entry = 132 + i * 12
        sig = data[entry : entry + 4]
        if sig == b"vcgt":
            return struct.unpack(">I", data[entry + 4 : entry + 8])[0]
    raise AssertionError("vcgt tag not found in tag table")


def test_vcgt_lut_roundtrip():
    r = np.zeros(256, dtype=np.uint16)
    g = np.full(256, 32767, dtype=np.uint16)
    b = np.full(256, 65535, dtype=np.uint16)
    data = build_vcgt_profile(r, g, b)
    pos = _vcgt_offset(data)
    # vcgt type header: sig(4) + reserved(4) + gamma_type(4) + channels(2) + count(2) + entry_size(2) = 18 bytes
    # Then R entries (256 * 2 = 512 bytes), then G entries start.
    g_start = pos + 18 + 512
    first_g = struct.unpack(">H", data[g_start : g_start + 2])[0]
    assert first_g == 32767
