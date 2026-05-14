import numpy as np
import pytest

from calibration.color_pipeline import (
    SRGB_PRIMARIES_XYZ_D50,
    SRGB_TO_XYZ_D50,
    camera_rgb_to_xyz_d50,
    fit_channel_gamma,
    fit_tone_response,
    forward_trc,
    prewarp_lut,
    project_onto_primary,
    white_balance,
)


def test_srgb_to_xyz_d50_matrix_maps_white_to_d50():
    """sRGB-linear (1,1,1) → D50 white point (0.9642, 1.0, 0.8249)."""
    xyz = SRGB_TO_XYZ_D50 @ np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(xyz, [0.9643, 1.0000, 0.8249], atol=0.005)


def test_srgb_to_xyz_d50_columns_are_primaries():
    np.testing.assert_allclose(SRGB_TO_XYZ_D50[:, 0], SRGB_PRIMARIES_XYZ_D50["R"])
    np.testing.assert_allclose(SRGB_TO_XYZ_D50[:, 1], SRGB_PRIMARIES_XYZ_D50["G"])
    np.testing.assert_allclose(SRGB_TO_XYZ_D50[:, 2], SRGB_PRIMARIES_XYZ_D50["B"])


def test_srgb_red_patch_projects_onto_srgb_red_primary():
    """For a pure-red sRGB-linear input, sRGB→XYZ_D50 then projection onto
    the sRGB R primary should return ≈ 1.0 (pure red contribution).
    """
    red_srgb_linear = np.array([1.0, 0.0, 0.0])
    xyz = SRGB_TO_XYZ_D50 @ red_srgb_linear
    proj = project_onto_primary(xyz, SRGB_PRIMARIES_XYZ_D50["R"])
    assert abs(proj - 1.0) < 1e-6


def test_fit_channel_gamma_recovers_22():
    levels = np.linspace(0, 1, 11)
    measured = np.where(levels > 0, levels ** 2.2, 0.0)
    g = fit_channel_gamma(levels, measured)
    assert abs(g - 2.2) < 0.02


def test_forward_trc_is_power_curve():
    lut = forward_trc(2.2)
    assert lut.shape == (256,)
    assert lut[0] == pytest.approx(0.0)
    assert lut[-1] == pytest.approx(1.0)
    # forward_trc(2.2) should equal level**2.2 at the midpoint.
    assert lut[128] == pytest.approx((128 / 255) ** 2.2, abs=0.005)


def test_forward_and_prewarp_compose_to_identity_at_target_gamma():
    """For target=γ_d, prewarp = identity. For arbitrary γ_d, composing
    measured (forward) with the *inverse* of forward should give identity.

    Sanity check: forward_trc and prewarp_lut are inverses *only* when
    target_gamma == γ_d. Otherwise the composition produces level^target.
    """
    levels = np.linspace(0, 1, 256)
    gamma_d = 2.2
    fwd = forward_trc(gamma_d)
    np.testing.assert_allclose(prewarp_lut(gamma_d, target_gamma=gamma_d), levels, atol=1e-6)
    # forward(prewarp(level)) → composed display behaviour = level^target_gamma
    composed = prewarp_lut(1.8, target_gamma=2.2) ** 1.8
    target = np.linspace(0, 1, 256) ** 2.2
    np.testing.assert_allclose(composed, target, atol=0.01)


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
