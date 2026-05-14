import numpy as np
import pytest

from calibration.ramp import (
    apply_lut_to_image,
    compose_luts,
    fit_correction,
    identity_lut,
    lut_to_vcgt,
)


def test_identity_lut_shape_and_endpoints():
    lut = identity_lut()
    assert lut.shape == (256,)
    assert lut.dtype == float
    assert lut[0] == pytest.approx(0.0)
    assert lut[-1] == pytest.approx(1.0)


def test_fit_correction_perfect_display_returns_identity():
    # Display already at γ=2.2 → correction is identity.
    levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    measured = np.where(levels > 0, levels ** 2.2, 0.0)
    lut = fit_correction(levels, measured)
    expected = identity_lut()
    np.testing.assert_allclose(lut, expected, atol=0.02)


def test_fit_correction_too_bright_darkens():
    # Display gamma 1.5 (too bright at midtones) → LUT exponent = 2.2/1.5 > 1 → darken.
    levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    measured = np.where(levels > 0, levels ** 1.5, 0.0)
    lut = fit_correction(levels, measured)
    identity = identity_lut()
    assert lut[128] < identity[128]


def test_compose_luts_with_identity_is_noop():
    identity = identity_lut()
    dark = np.linspace(0, 1, 256) ** 2.0
    result = compose_luts(identity, dark)
    np.testing.assert_allclose(result, dark, atol=1e-6)


def test_compose_luts_double_darken():
    dark = np.linspace(0, 1, 256) ** 2.0
    once = compose_luts(identity_lut(), dark)
    twice = compose_luts(once, dark)
    assert twice[128] < once[128]


def test_lut_to_vcgt_dtype_and_endpoints():
    lut = identity_lut()
    vcgt = lut_to_vcgt(lut)
    assert vcgt.dtype == np.uint16
    assert vcgt.shape == (256,)
    assert vcgt[0] == 0
    assert vcgt[-1] == 65535


def test_lut_to_vcgt_midpoint():
    lut = identity_lut()
    vcgt = lut_to_vcgt(lut)
    assert 32000 < int(vcgt[128]) < 34000


def test_apply_lut_to_image_identity():
    lut = identity_lut()
    img = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
    result = apply_lut_to_image(img, lut, lut, lut)
    np.testing.assert_array_equal(result[:, :, 0], img[:, :, 0])


def test_apply_lut_to_image_black_lut():
    lut = np.zeros(256)
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    result = apply_lut_to_image(img, lut, lut, lut)
    assert result.max() == 0
