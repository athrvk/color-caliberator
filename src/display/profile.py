"""
Builds a minimal ICC v2 RGB display profile containing an Apple VCGT tag.
dispwin reads the VCGT tag to load the VideoLUT onto the display.
"""

import struct
from datetime import datetime, timezone

import numpy as np


def _s15f16(v: float) -> int:
    """Convert float to ICC s15Fixed16 (signed, big-endian 32-bit)."""
    return int(round(v * 65536)) & 0xFFFFFFFF


def _xyz_type(X: float, Y: float, Z: float) -> bytes:
    return struct.pack(">4sI3i", b"XYZ ", 0, _s15f16(X), _s15f16(Y), _s15f16(Z))


def _curv_gamma(gamma: float) -> bytes:
    """ICC 'curv' tag with a single gamma value (u8Fixed8Number)."""
    g = int(round(gamma * 256)) & 0xFFFF
    return struct.pack(">4sIIH", b"curv", 0, 1, g)


def _desc_type(text: str) -> bytes:
    """ICC v2 'desc' tag (ASCII subset only)."""
    ascii_bytes = text.encode("ascii") + b"\x00"
    count = len(ascii_bytes)
    body = struct.pack(">4sII", b"desc", 0, count)
    body += ascii_bytes
    body += struct.pack(">II", 0, 0)
    body += struct.pack(">HB", 0, 0)
    body += b"\x00" * 67
    return body


def _curv_table(curve: np.ndarray) -> bytes:
    """ICC v2 'curv' type with 256-entry uint16 lookup table."""
    table = np.clip(np.round(np.asarray(curve, dtype=float) * 65535.0), 0, 65535).astype(np.uint16)
    return struct.pack(">4sII", b"curv", 0, table.shape[0]) + table.astype(">u2").tobytes()


_D50_XYZ = np.array([0.9642, 1.0000, 0.8249])


def _vcgt_type(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> bytes:
    """Apple VideoCardGamma VCGT tag, table variant (type=0)."""
    header = struct.pack(">4sIIHHH", b"vcgt", 0, 0, 3, 256, 2)
    r_be = r.astype(">u2").tobytes()
    g_be = g.astype(">u2").tobytes()
    b_be = b.astype(">u2").tobytes()
    return header + r_be + g_be + b_be


def _cprt_type(text: str) -> bytes:
    """ICC v2 'text' type used for copyright tag."""
    return struct.pack(">4sI", b"text", 0) + text.encode("ascii") + b"\x00"


def _build_header(profile_size: int) -> bytes:
    now = datetime.now(tz=timezone.utc)
    h = b""
    h += struct.pack(">I", profile_size)
    h += b"    "
    h += struct.pack(">I", 0x02100000)
    h += b"mntr"
    h += b"RGB "
    h += b"XYZ "
    h += struct.pack(">6H",
                     now.year, now.month, now.day,
                     now.hour, now.minute, now.second)
    h += b"acsp"
    h += b"MSFT"   # primary platform: Microsoft
    h += struct.pack(">I", 0)
    h += struct.pack(">I", 0)
    h += struct.pack(">I", 0)
    h += struct.pack(">Q", 0)
    h += struct.pack(">I", 0)
    h += struct.pack(">iii", _s15f16(0.96420), _s15f16(1.00000), _s15f16(0.82491))
    h += struct.pack(">I", 0)
    h += b"\x00" * 16
    h += b"\x00" * 28
    assert len(h) == 128, f"Header must be 128 bytes, got {len(h)}"
    return h


def build_vcgt_profile(
    r_lut: np.ndarray,
    g_lut: np.ndarray,
    b_lut: np.ndarray,
) -> bytes:
    """
    Build a minimal ICC v2 sRGB display profile with an Apple VCGT tag.

    r_lut, g_lut, b_lut: uint16 arrays of length 256 (from ramp.lut_to_vcgt).
    Returns raw ICC profile bytes ready to write to a .icc file.
    """
    tags_data: dict[bytes, bytes] = {
        b"desc": _desc_type("Color Calibrator"),
        b"cprt": _cprt_type("Public Domain"),
        b"wtpt": _xyz_type(0.95045, 1.00000, 1.08905),
        b"rXYZ": _xyz_type(0.43607, 0.22249, 0.01392),
        b"gXYZ": _xyz_type(0.38515, 0.71687, 0.09708),
        b"bXYZ": _xyz_type(0.14307, 0.06061, 0.71410),
        b"rTRC": _curv_gamma(2.2),
        b"gTRC": _curv_gamma(2.2),
        b"bTRC": _curv_gamma(2.2),
        b"vcgt": _vcgt_type(r_lut, g_lut, b_lut),
    }

    n = len(tags_data)
    tag_table_offset = 128
    tag_data_start = 128 + 4 + n * 12

    tag_layout: list[tuple[bytes, int, int, bytes]] = []
    offset = tag_data_start
    for sig, data in tags_data.items():
        size = len(data)
        pad = (-size) % 4
        padded = data + b"\x00" * pad
        tag_layout.append((sig, offset, size, padded))
        offset += len(padded)

    profile_size = offset

    tag_table = struct.pack(">I", n)
    for sig, off, size, _ in tag_layout:
        tag_table += sig + struct.pack(">II", off, size)

    tag_data = b"".join(padded for _, _, _, padded in tag_layout)

    header = _build_header(profile_size)
    return header + tag_table + tag_data


def build_matrix_shaper_profile(
    r_trc: np.ndarray,
    g_trc: np.ndarray,
    b_trc: np.ndarray,
    r_xyz_d50: np.ndarray,
    g_xyz_d50: np.ndarray,
    b_xyz_d50: np.ndarray,
    r_vcgt_lut: np.ndarray | None = None,
    g_vcgt_lut: np.ndarray | None = None,
    b_vcgt_lut: np.ndarray | None = None,
) -> bytes:
    """Build an ICC v2 matrix-shaper profile.

    r/g/b_trc: float [0, 1] tone response curves, length 256 (forward).
    r/g/b_xyz_d50: display primaries in XYZ under D50 (PCS).
    r/g/b_vcgt_lut: optional VCGT correction LUTs (uint16). If None, VCGT is identity.
    """
    if r_vcgt_lut is None:
        identity = (np.linspace(0, 1, 256) * 65535).astype(np.uint16)
        r_vcgt_lut = g_vcgt_lut = b_vcgt_lut = identity

    tags_data: dict[bytes, bytes] = {
        b"desc": _desc_type("Color Calibrator (matrix)"),
        b"cprt": _cprt_type("Public Domain"),
        b"wtpt": _xyz_type(*_D50_XYZ),
        b"rXYZ": _xyz_type(*r_xyz_d50),
        b"gXYZ": _xyz_type(*g_xyz_d50),
        b"bXYZ": _xyz_type(*b_xyz_d50),
        b"rTRC": _curv_table(r_trc),
        b"gTRC": _curv_table(g_trc),
        b"bTRC": _curv_table(b_trc),
        b"vcgt": _vcgt_type(r_vcgt_lut, g_vcgt_lut, b_vcgt_lut),
    }

    n = len(tags_data)
    tag_data_start = 128 + 4 + n * 12
    tag_layout: list[tuple[bytes, int, int, bytes]] = []
    offset = tag_data_start
    for sig, data in tags_data.items():
        size = len(data)
        pad = (-size) % 4
        padded = data + b"\x00" * pad
        tag_layout.append((sig, offset, size, padded))
        offset += len(padded)

    profile_size = offset
    tag_table = struct.pack(">I", n)
    for sig, off, size, _ in tag_layout:
        tag_table += sig + struct.pack(">II", off, size)

    tag_data = b"".join(p for _, _, _, p in tag_layout)
    header = _build_header(profile_size)
    return header + tag_table + tag_data
