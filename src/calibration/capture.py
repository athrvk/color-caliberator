import base64
from io import BytesIO

import numpy as np
from PIL import Image


def frame_luminance(frame: np.ndarray) -> float:
    """BT.601 luma of the center 50% crop of an H×W×3 uint8 array.

    Operates in sRGB-encoded space — fine for stability/SSNR checks where only
    variance matters. For calibration math (gamma fit, luminance ratios), use
    `frame_luminance_linear` instead so the math stays in physical light.
    """
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    r, g, b = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    return float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean())


def srgb_to_linear(rgb_0_1: np.ndarray) -> np.ndarray:
    """Reverse the sRGB encoding curve.

    Assumes the phone encodes JPEGs in sRGB. iPhones since iOS 11 may use
    Display P3 by default; users should set their camera to Most Compatible
    (sRGB) for accurate results.
    """
    a = 0.055
    return np.where(
        rgb_0_1 <= 0.04045,
        rgb_0_1 / 12.92,
        ((rgb_0_1 + a) / (1 + a)) ** 2.4,
    )


def frame_luminance_linear(frame: np.ndarray) -> float:
    """BT.709 relative luminance of the center 50% crop after sRGB→linear.

    Use this for any math that compares against physical luminance targets
    (e.g. patch_luma / white_luma vs. level**2.2).
    """
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].astype(np.float64) / 255.0
    lin = srgb_to_linear(crop)
    r = lin[:, :, 0].mean()
    g = lin[:, :, 1].mean()
    b = lin[:, :, 2].mean()
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def ssnr_db(luminances: list[float]) -> float:
    """Signal-to-Noise Ratio in dB. Returns inf when std == 0."""
    arr = np.array(luminances, dtype=float)
    std = arr.std()
    if std == 0.0:
        return float("inf")
    return float(20.0 * np.log10(arr.mean() / std))


def is_stable(luminances: list[float], threshold_db: float = 20.0) -> bool:
    """True when ≥5 frames with SSNR ≥ threshold_db."""
    return len(luminances) >= 5 and ssnr_db(luminances) >= threshold_db


def decode_frame(b64_jpeg: str) -> np.ndarray:
    """Decode a base64-encoded JPEG string to an H×W×3 uint8 numpy array."""
    data = base64.b64decode(b64_jpeg)
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)
