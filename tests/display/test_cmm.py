import numpy as np
import pytest

from display.cmm import render_through_profile


# sRGB primaries in XYZ_D50 (Bradford-adapted from D65). Identity case:
# when the destination profile claims sRGB primaries and γ=2.2, the CMM
# output should match the source's sRGB encoding closely.
_R = np.array([0.4361, 0.2225, 0.0139])
_G = np.array([0.3851, 0.7169, 0.0606])
_B = np.array([0.1431, 0.0606, 0.7141])


def test_identity_round_trip_returns_source_for_srgb_display():
    """If display primaries == sRGB and γ ≈ 2.2 → render returns ≈ source."""
    src = np.array([[[64, 128, 200]]], dtype=np.uint8)
    out = render_through_profile(src, _R, _G, _B, 2.2, 2.2, 2.2)
    assert out.shape == src.shape
    # sRGB encoding curve uses 2.4 in the high segment; with γ=2.2 round
    # trip we accept a small offset.
    np.testing.assert_allclose(out, src, atol=4)


def test_lower_display_gamma_darkens_framebuffer():
    """Display with γ=1.6 puts out more light per level than γ=2.2, so the
    CMM must send darker framebuffer values to hit the same target linear
    light. Output midtone for γ=1.6 < midtone for γ=2.2.
    """
    src = np.array([[[128, 128, 128]]], dtype=np.uint8)
    out_calibrated = render_through_profile(src, _R, _G, _B, 2.2, 2.2, 2.2)
    out_bright = render_through_profile(src, _R, _G, _B, 1.6, 1.6, 1.6)
    assert int(out_bright[0, 0, 0]) < int(out_calibrated[0, 0, 0])


def test_black_stays_black_and_white_stays_white():
    src_black = np.zeros((1, 1, 3), dtype=np.uint8)
    src_white = np.full((1, 1, 3), 255, dtype=np.uint8)
    out_b = render_through_profile(src_black, _R, _G, _B, 2.2, 2.2, 2.2)
    out_w = render_through_profile(src_white, _R, _G, _B, 2.2, 2.2, 2.2)
    assert int(out_b.max()) == 0
    assert int(out_w.min()) >= 250


def test_dtype_check_rejects_non_uint8():
    src = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
    with pytest.raises(TypeError):
        render_through_profile(src, _R, _G, _B, 2.2, 2.2, 2.2)
