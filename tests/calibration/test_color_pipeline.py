import numpy as np
import pytest

from calibration.color_pipeline import (
    SRGB_PRIMARIES_XYZ_D50,
    camera_rgb_to_xyz_d50,
    fit_tone_response,
    project_onto_primary,
    white_balance,
)


def test_white_balance_unity_on_neutral():
    rgb = np.array([0.5, 1.0, 0.7])
    neutral = np.array([0.5, 1.0, 0.7])
    wb = white_balance(rgb, neutral)
    np.testing.assert_allclose(wb, [1.0, 1.0, 1.0], atol=1e-6)


def test_white_balance_scales_proportionally():
    rgb = np.array([0.25, 0.5, 0.35])
    neutral = np.array([0.5, 1.0, 0.7])
    wb = white_balance(rgb, neutral)
    np.testing.assert_allclose(wb, [0.5, 0.5, 0.5], atol=1e-6)


def test_camera_rgb_to_xyz_d50_returns_three_finite_values():
    cam_rgb = np.array([0.5, 0.5, 0.5])
    neutral = np.array([1.0, 1.0, 1.0])
    forward = np.eye(3) * 0.9642
    xyz = camera_rgb_to_xyz_d50(cam_rgb, neutral, forward)
    assert xyz.shape == (3,)
    assert np.all(np.isfinite(xyz))


def test_srgb_primaries_d50_constants():
    expected = {
        "R": np.array([0.4361, 0.2225, 0.0139]),
        "G": np.array([0.3851, 0.7169, 0.0971]),
        "B": np.array([0.1431, 0.0606, 0.7141]),
    }
    for k, v in expected.items():
        np.testing.assert_allclose(SRGB_PRIMARIES_XYZ_D50[k], v, atol=1e-3)


def test_project_onto_primary_full_alignment():
    primary = np.array([0.4361, 0.2225, 0.0139])
    assert abs(project_onto_primary(primary, primary) - 1.0) < 1e-6


def test_project_onto_primary_half_amount():
    primary = np.array([0.4361, 0.2225, 0.0139])
    measured = primary * 0.5
    assert abs(project_onto_primary(measured, primary) - 0.5) < 1e-6


def test_project_onto_primary_orthogonal_is_zero():
    primary = np.array([1.0, 0.0, 0.0])
    measured = np.array([0.0, 1.0, 0.0])
    assert abs(project_onto_primary(measured, primary)) < 1e-6


def test_fit_tone_response_perfect_returns_identity():
    levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    measured = np.where(levels > 0, levels ** 2.2, 0.0)
    lut = fit_tone_response(levels, measured)
    assert lut.shape == (256,)
    expected = np.linspace(0, 1, 256)
    np.testing.assert_allclose(lut, expected, atol=0.02)
