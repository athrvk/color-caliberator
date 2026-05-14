"""Minimal in-process Color Management Module render.

Simulates what the OS CMM would push to the framebuffer when an sRGB-tagged
source is shown on a display characterised by a matrix-shaper ICC profile.
Used to produce honest "after" preview chart in color mode — where a 1D
VideoLUT preview cannot represent the primary-chromaticity step.

Pipeline:
    sRGB pixel → linear sRGB → XYZ_D50 → linear display RGB → device RGB
"""

from __future__ import annotations

import numpy as np

from calibration.capture import srgb_to_linear

# sRGB primaries expressed in XYZ_D50 (Bradford-adapted from D65). Each
# column is a primary's XYZ. Same numbers as SRGB_PRIMARIES_XYZ_D50 in
# calibration.color_pipeline; replicated here to avoid a cross-package
# import for what is effectively a display-side concern.
_SRGB_TO_XYZ_D50 = np.array([
    [0.4361, 0.3851, 0.1431],
    [0.2225, 0.7169, 0.0606],
    [0.0139, 0.0971, 0.7141],
])


def render_through_profile(
    chart_srgb_uint8: np.ndarray,
    r_xyz_d50: np.ndarray,
    g_xyz_d50: np.ndarray,
    b_xyz_d50: np.ndarray,
    gamma_r: float,
    gamma_g: float,
    gamma_b: float,
) -> np.ndarray:
    """Render an sRGB chart through a matrix-shaper destination profile.

    Returns the framebuffer (device-encoded) RGB values that the OS CMM
    would produce after installing the profile. On the actual calibrated
    display these device values yield the source's intended XYZ.

    chart_srgb_uint8: H x W x 3, uint8, sRGB-encoded.
    r/g/b_xyz_d50:     display primaries (XYZ under D50). Length-3 vectors.
    gamma_r/g/b:       measured per-channel display TRC exponents.
    """
    if chart_srgb_uint8.dtype != np.uint8:
        raise TypeError("chart must be uint8 sRGB")
    h, w, _ = chart_srgb_uint8.shape

    linear_src = srgb_to_linear(chart_srgb_uint8.astype(np.float64) / 255.0)
    xyz = linear_src.reshape(-1, 3) @ _SRGB_TO_XYZ_D50.T

    m_dst = np.column_stack([
        np.asarray(r_xyz_d50, dtype=float),
        np.asarray(g_xyz_d50, dtype=float),
        np.asarray(b_xyz_d50, dtype=float),
    ])
    m_dst_inv = np.linalg.inv(m_dst)
    linear_dst = (xyz @ m_dst_inv.T).reshape(h, w, 3)
    linear_dst = np.clip(linear_dst, 0.0, 1.0)

    # Inverse TRC: device = linear^(1/γ_d) so display(device) = linear.
    device = np.empty_like(linear_dst)
    device[:, :, 0] = linear_dst[:, :, 0] ** (1.0 / max(gamma_r, 1e-3))
    device[:, :, 1] = linear_dst[:, :, 1] ** (1.0 / max(gamma_g, 1e-3))
    device[:, :, 2] = linear_dst[:, :, 2] ** (1.0 / max(gamma_b, 1e-3))

    return np.clip(np.round(device * 255.0), 0, 255).astype(np.uint8)
