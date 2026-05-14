import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from calibration.capture import decode_frame, frame_luminance, is_stable, ssnr_db


def test_ssnr_db_perfectly_stable():
    lumas = [100.0] * 5
    result = ssnr_db(lumas)
    assert result == float("inf")


def test_ssnr_db_noisy():
    lumas = [50.0, 150.0, 50.0, 150.0, 100.0]
    assert ssnr_db(lumas) < 20.0


def test_is_stable_requires_5_frames():
    assert not is_stable([100.0] * 4)


def test_is_stable_with_constant_lumas():
    assert is_stable([128.0] * 5)


def test_is_stable_noisy_returns_false():
    assert not is_stable([50.0, 150.0, 50.0, 150.0, 100.0])


def test_frame_luminance_white_frame():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    luma = frame_luminance(frame)
    assert abs(luma - 255.0) < 1.0


def test_frame_luminance_black_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert frame_luminance(frame) == 0.0


def test_frame_luminance_uses_center_crop():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[25:75, 25:75] = 255
    luma = frame_luminance(frame)
    assert luma > 200.0


def _make_b64_jpeg(color: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (64, 64), color)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_frame_returns_numpy_array():
    b64 = _make_b64_jpeg((128, 64, 32))
    arr = decode_frame(b64)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8
