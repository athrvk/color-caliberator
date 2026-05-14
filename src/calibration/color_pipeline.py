"""Color pipeline: camera RGB → XYZ_D50, primary projection, TRC fit.

Everything lives in PCS-D50 (the ICC working space). No Bradford CAT during
measurement — DNG ForwardMatrix2 already gives us D50 output, and the ICC
matrix profile consumes D50 primaries directly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# sRGB primaries in PCS-D50 (Bradford-adapted from D65 once, at compile time).
SRGB_PRIMARIES_XYZ_D50 = {
    "R": np.array([0.4361, 0.2225, 0.0139]),
    "G": np.array([0.3851, 0.7169, 0.0971]),
    "B": np.array([0.1431, 0.0606, 0.7141]),
}

# Convenience: column-stacked matrix form. Multiply by sRGB-linear RGB to
# get XYZ_D50 — used for per-patch JPEG measurements where ForwardMatrix
# cannot be used (JPEG data is sRGB-encoded, not camera-linear).
SRGB_TO_XYZ_D50 = np.column_stack([
    SRGB_PRIMARIES_XYZ_D50["R"],
    SRGB_PRIMARIES_XYZ_D50["G"],
    SRGB_PRIMARIES_XYZ_D50["B"],
])


def white_balance(camera_rgb: np.ndarray, as_shot_neutral: np.ndarray) -> np.ndarray:
    """Divide camera RGB by AsShotNeutral so neutral = (1, 1, 1)."""
    asn = np.clip(np.asarray(as_shot_neutral, dtype=float), 1e-6, None)
    return np.asarray(camera_rgb, dtype=float) / asn


def camera_rgb_to_xyz_d50(
    camera_rgb: np.ndarray,
    as_shot_neutral: np.ndarray,
    forward_matrix_2: np.ndarray,
) -> np.ndarray:
    """Camera RGB → XYZ_D50 via DNG ForwardMatrix2.

    camera_rgb: linear camera-RGB values.
    as_shot_neutral: from DNG; the neutral reference under capture illuminant.
    forward_matrix_2: from DNG; maps WB-applied camera RGB → XYZ_D50.
    """
    wb = white_balance(camera_rgb, as_shot_neutral)
    return forward_matrix_2 @ wb


def project_onto_primary(measured_xyz: np.ndarray, primary_xyz: np.ndarray) -> float:
    """Scalar amount of `primary` present in `measured`, both in XYZ.

    Computed as the projection of measured onto primary's direction, divided by
    primary's squared norm so that measured == primary → 1.0.
    """
    primary = np.asarray(primary_xyz, dtype=float)
    measured = np.asarray(measured_xyz, dtype=float)
    denom = float(np.dot(primary, primary))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(measured, primary) / denom)


def fit_tone_response(
    input_levels: np.ndarray,
    measured: np.ndarray,
    target_gamma: float = 2.2,
) -> np.ndarray:
    """Per-channel pre-warp LUT (backwards-compatible legacy API).

    Use `fit_channel_gamma` + `forward_trc` / `prewarp_lut` for new code.
    """
    gamma_d = fit_channel_gamma(input_levels, measured)
    return prewarp_lut(gamma_d, target_gamma)


def fit_channel_gamma(input_levels: np.ndarray, measured: np.ndarray) -> float:
    """Fit `measured = level^γ_d` and return γ_d.

    Black (level=0) excluded — phone-camera black level is unreliable.
    """
    input_levels = np.asarray(input_levels, dtype=float)
    measured = np.asarray(measured, dtype=float)
    mask = input_levels > 0.0
    m = np.clip(measured[mask], 1e-6, None)
    (gamma_d,), _ = curve_fit(
        lambda x, g: x ** g, input_levels[mask], m, p0=[2.2], bounds=(0.5, 5.0),
    )
    return float(gamma_d)


def forward_trc(gamma_d: float) -> np.ndarray:
    """Measured display TRC as a 256-entry [0,1] LUT: `level^γ_d`.

    This is what an ICC matrix-shaper profile's rTRC/gTRC/bTRC tag stores —
    the display's forward characterization, so the CMM can invert it.
    """
    return np.clip(np.linspace(0.0, 1.0, 256) ** gamma_d, 0.0, 1.0)


def prewarp_lut(gamma_d: float, target_gamma: float = 2.2) -> np.ndarray:
    """Pre-warp LUT to compose with display so the result hits target_gamma."""
    exponent = target_gamma / gamma_d
    return np.clip(np.linspace(0.0, 1.0, 256) ** exponent, 0.0, 1.0)
