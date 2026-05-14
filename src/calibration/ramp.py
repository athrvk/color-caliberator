import numpy as np
from scipy.optimize import curve_fit


def identity_lut() -> np.ndarray:
    """256-entry float [0, 1] identity LUT."""
    return np.linspace(0.0, 1.0, 256)


def fit_correction(
    input_levels: np.ndarray,
    measured_luma: np.ndarray,
    target_gamma: float = 2.2,
) -> np.ndarray:
    """
    Fit the display's effective gamma γ_d from (input, measured_luma) pairs,
    then build a pre-warp LUT so the composed pipeline matches target_gamma.

    Physical model:
        measured_luma ≈ input_levels ** γ_d
    To make `display(LUT(x)) = x^target_gamma`, we need
        LUT(x) = x ** (target_gamma / γ_d)

    Black (input=0) is excluded — phone camera black level is unreliable.
    """
    mask = input_levels > 0.0
    measured = np.clip(measured_luma[mask], 1e-6, None)
    (gamma_d,), _ = curve_fit(
        lambda x, g: x ** g,
        input_levels[mask],
        measured,
        p0=[2.2],
        bounds=(0.5, 5.0),
    )
    exponent = target_gamma / gamma_d
    return np.clip(np.linspace(0.0, 1.0, 256) ** exponent, 0.0, 1.0)


def compose_luts(prev_lut: np.ndarray, new_lut: np.ndarray) -> np.ndarray:
    """
    Apply new_lut on top of prev_lut.
    Both must be float [0, 1], length 256.
    Uses np.interp to stay in float space throughout.
    """
    x = np.linspace(0.0, 1.0, 256)
    return np.interp(new_lut, x, prev_lut)


def lut_to_vcgt(lut: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] LUT to uint16 for ICC VCGT tag emission."""
    return np.clip(np.round(lut * 65535.0), 0, 65535).astype(np.uint16)


def apply_lut_to_image(
    img: np.ndarray,
    r_lut: np.ndarray,
    g_lut: np.ndarray,
    b_lut: np.ndarray,
) -> np.ndarray:
    """
    Apply float [0,1] per-channel LUTs to an H×W×3 uint8 image.
    Returns a new uint8 array (does not modify img).

    LUTs MUST be float [0, 1] (not the uint16 VCGT form). Passing uint16 LUTs
    here would scale 65535 → 255 and clip everything to white.
    """
    for name, lut in (("r", r_lut), ("g", g_lut), ("b", b_lut)):
        if lut.dtype == np.uint16:
            raise TypeError(f"{name}_lut is uint16 (VCGT form); pass float [0,1] LUTs")
    r256 = np.clip(np.round(r_lut * 255.0), 0, 255).astype(np.uint8)
    g256 = np.clip(np.round(g_lut * 255.0), 0, 255).astype(np.uint8)
    b256 = np.clip(np.round(b_lut * 255.0), 0, 255).astype(np.uint8)
    out = np.empty_like(img)
    out[:, :, 0] = r256[img[:, :, 0]]
    out[:, :, 1] = g256[img[:, :, 1]]
    out[:, :, 2] = b256[img[:, :, 2]]
    return out
