import base64
from io import BytesIO

import numpy as np
from PIL import Image


def frame_luminance(frame: np.ndarray) -> float:
    """BT.601 luma of the center 50% crop of an H×W×3 uint8 array."""
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    r, g, b = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    return float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean())


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
