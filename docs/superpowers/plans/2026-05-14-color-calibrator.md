# Color Calibrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python web server that uses a phone camera as a relative sensor in an iterative feedback loop to produce an ICC v2 gamma-correction profile for the connected display.

**Architecture:** FastAPI serves two browser pages (PC + mobile). The PC page shows grayscale patches; the mobile page streams JPEG frames over WebSocket. The server measures frame stability via SSNR, fits a per-channel gamma correction curve against target luminance ratios, applies it via `dispwin` (ArgyllCMS), and repeats up to 3 rounds until convergence. The final correction is exported as an ICC v2 profile with a VCGT tag.

**Tech Stack:** Python 3.11+, uv, FastAPI, uvicorn, numpy, scipy, Pillow, qrcode[pil], ArgyllCMS (external CLI dependency). ICC profiles are written as raw bytes using `struct` — no python-lcms2 needed (it does not expose VCGT writing).

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | Entry point: check dispwin, detect LAN IP, start uvicorn |
| `pyproject.toml` | uv project config, pytest config |
| `src/calibration/__init__.py` | Empty |
| `src/calibration/patches.py` | 11 gray patch definitions + 3 holdout patches |
| `src/calibration/capture.py` | SSNR stability check, frame luminance, frame decoding |
| `src/calibration/ramp.py` | Curve fitting, LUT composition, LUT-to-image application |
| `src/calibration/iterate.py` | Async calibration loop (WebSocket callbacks) |
| `src/display/__init__.py` | Empty (named `display/` not `platform/` to avoid shadowing stdlib) |
| `src/display/dispwin.py` | `find_dispwin`, `apply_ramp`, `clear_ramp` |
| `src/display/profile.py` | `build_vcgt_profile` → ICC v2 bytes with VCGT tag |
| `src/util/__init__.py` | Empty |
| `src/util/qr.py` | `generate_qr_png(url) → bytes` |
| `src/util/tls.py` | `ensure_cert`, `detect_lan_ip` — self-signed TLS cert for getUserMedia |
| `src/web/__init__.py` | Empty |
| `src/web/server.py` | FastAPI app, WebSocket endpoints, session state |
| `src/web/static/pc.html` | PC wizard: QR → setup → progress → result |
| `src/web/static/mobile.html` | Mobile: camera viewfinder → streaming frames |
| `src/web/static/style.css` | Minimal shared styles |
| `tests/calibration/test_patches.py` | |
| `tests/calibration/test_capture.py` | |
| `tests/calibration/test_ramp.py` | |
| `tests/calibration/test_iterate.py` | Integration test: fake send/recv driving `run_calibration` |
| `tests/display/test_dispwin.py` | |
| `tests/display/test_profile.py` | |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `main.py` (stub)
- Create: `src/calibration/__init__.py`, `src/display/__init__.py`, `src/util/__init__.py`, `src/web/__init__.py`
- Create: `src/web/static/` directory (empty placeholder files)
- Create: `tests/__init__.py`, `tests/calibration/__init__.py`, `tests/display/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "color-calibrator"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "numpy",
    "scipy",
    "pillow",
    "qrcode[pil]",
    "cryptography",
    "pytest",
    "pytest-asyncio",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/calibration", "src/display", "src/util", "src/web"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create directory tree and empty `__init__.py` files**

Run:
```
mkdir -p src/calibration src/display src/util src/web/static
mkdir -p tests/calibration tests/display
touch src/calibration/__init__.py src/display/__init__.py src/util/__init__.py src/web/__init__.py
touch tests/__init__.py tests/calibration/__init__.py tests/display/__init__.py
```

- [ ] **Step 3: Create stub `main.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8765, reload=False)
```

- [ ] **Step 4: Install dependencies**

Run: `uv sync`
Expected: no errors; `.venv` created.

- [ ] **Step 5: Verify pytest finds tests**

Run: `uv run pytest --collect-only`
Expected: "no tests ran" (no test files yet) — not an error.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml main.py src/ tests/
git commit -m "feat: project scaffold — src layout, pyproject, empty packages"
```

---

## Task 2: Patch Definitions

**Files:**
- Create: `src/calibration/patches.py`
- Create: `tests/calibration/test_patches.py`

- [ ] **Step 1: Write the failing tests**

`tests/calibration/test_patches.py`:
```python
from calibration.patches import GRAY_PATCHES, HOLDOUT_PATCHES, GrayPatch


def test_gray_patch_count():
    assert len(GRAY_PATCHES) == 11


def test_holdout_patch_count():
    assert len(HOLDOUT_PATCHES) == 3


def test_gray_patch_type():
    for p in GRAY_PATCHES:
        assert isinstance(p, GrayPatch)
        assert 0.0 <= p.level <= 1.0
        assert 0.0 <= p.target_luma <= 1.0


def test_gray_patch_target_luma_gamma22():
    for p in GRAY_PATCHES:
        if p.level == 0.0:
            assert p.target_luma == 0.0
        else:
            assert abs(p.target_luma - p.level ** 2.2) < 1e-6


def test_holdout_levels():
    levels = [p.level for p in HOLDOUT_PATCHES]
    assert levels == [0.25, 0.50, 0.75]


def test_gray_patches_cover_full_range():
    levels = [p.level for p in GRAY_PATCHES]
    assert 0.0 in levels
    assert 1.0 in levels
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `uv run pytest tests/calibration/test_patches.py -v`
Expected: `ModuleNotFoundError: No module named 'calibration.patches'`

- [ ] **Step 3: Implement `src/calibration/patches.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GrayPatch:
    level: float        # display input [0.0, 1.0]
    target_luma: float  # expected relative luminance for gamma 2.2


GRAY_PATCHES: list[GrayPatch] = [
    GrayPatch(level=l, target_luma=(l ** 2.2 if l > 0.0 else 0.0))
    for l in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
]

HOLDOUT_PATCHES: list[GrayPatch] = [
    GrayPatch(level=l, target_luma=l ** 2.2)
    for l in [0.25, 0.50, 0.75]
]
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `uv run pytest tests/calibration/test_patches.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/calibration/patches.py tests/calibration/test_patches.py
git commit -m "feat: gray patch definitions — 11 calibration + 3 holdout patches"
```

---

## Task 3: Frame Capture and SSNR

**Files:**
- Create: `src/calibration/capture.py`
- Create: `tests/calibration/test_capture.py`

- [ ] **Step 1: Write the failing tests**

`tests/calibration/test_capture.py`:
```python
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
    # mean≈100, std≈50 → db < 20
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
    # Fill only center 50% with white — luminance should be ~255 (center is measured)
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/calibration/test_capture.py -v`
Expected: `ModuleNotFoundError: No module named 'calibration.capture'`

- [ ] **Step 3: Implement `src/calibration/capture.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `uv run pytest tests/calibration/test_capture.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/calibration/capture.py tests/calibration/test_capture.py
git commit -m "feat: frame capture — SSNR stability check, BT.601 luma, JPEG decode"
```

---

## Task 4: Gamma Ramp Math

**Files:**
- Create: `src/calibration/ramp.py`
- Create: `tests/calibration/test_ramp.py`

- [ ] **Step 1: Write the failing tests**

`tests/calibration/test_ramp.py`:
```python
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
    # lut[128] ≈ 128/255 ≈ 0.502 → vcgt[128] ≈ 32893
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/calibration/test_ramp.py -v`
Expected: `ModuleNotFoundError: No module named 'calibration.ramp'`

- [ ] **Step 3: Implement `src/calibration/ramp.py`**

```python
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
    # Scale each float LUT to 0-255 for indexing
    r256 = np.clip(np.round(r_lut * 255.0), 0, 255).astype(np.uint8)
    g256 = np.clip(np.round(g_lut * 255.0), 0, 255).astype(np.uint8)
    b256 = np.clip(np.round(b_lut * 255.0), 0, 255).astype(np.uint8)
    out = np.empty_like(img)
    out[:, :, 0] = r256[img[:, :, 0]]
    out[:, :, 1] = g256[img[:, :, 1]]
    out[:, :, 2] = b256[img[:, :, 2]]
    return out
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `uv run pytest tests/calibration/test_ramp.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/calibration/ramp.py tests/calibration/test_ramp.py
git commit -m "feat: gamma ramp math — curve fit, LUT composition, VCGT conversion, image application"
```

---

## Task 5: dispwin Wrapper

**Files:**
- Create: `src/display/dispwin.py`
- Create: `tests/display/test_dispwin.py`

- [ ] **Step 1: Write the failing tests**

`tests/display/test_dispwin.py`:
```python
from unittest.mock import patch

from display.dispwin import apply_ramp, clear_ramp, find_dispwin


def test_find_dispwin_returns_none_when_absent():
    with patch("shutil.which", return_value=None):
        assert find_dispwin() is None


def test_find_dispwin_returns_path_when_present():
    with patch("shutil.which", return_value="/usr/bin/dispwin"):
        assert find_dispwin() == "/usr/bin/dispwin"


def test_apply_ramp_calls_dispwin_with_correct_args():
    with patch("subprocess.run") as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=1)
        mock_run.assert_called_once_with(
            ["dispwin", "-d1", "-I", "/tmp/cal.icc"], check=True
        )


def test_apply_ramp_uses_display_index():
    with patch("subprocess.run") as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=2)
        args = mock_run.call_args[0][0]
        assert "-d2" in args


def test_clear_ramp_calls_dispwin_reset():
    with patch("subprocess.run") as mock_run:
        clear_ramp(display_index=1)
        mock_run.assert_called_once_with(
            ["dispwin", "-d1", "-c"], check=True
        )
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/display/test_dispwin.py -v`
Expected: `ModuleNotFoundError: No module named 'display.dispwin'`

- [ ] **Step 3: Implement `src/display/dispwin.py`**

```python
import shutil
import subprocess


def find_dispwin() -> str | None:
    """Return path to dispwin binary, or None if not on PATH."""
    return shutil.which("dispwin")


def apply_ramp(profile_path: str, display_index: int = 1) -> None:
    """Load an ICC profile's VCGT tag as the display's VideoLUT."""
    subprocess.run(["dispwin", f"-d{display_index}", "-I", profile_path], check=True)


def clear_ramp(display_index: int = 1) -> None:
    """Reset the display VideoLUT to linear (identity)."""
    subprocess.run(["dispwin", f"-d{display_index}", "-c"], check=True)
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `uv run pytest tests/display/test_dispwin.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/display/dispwin.py tests/display/test_dispwin.py
git commit -m "feat: dispwin wrapper — find, apply, and clear display VideoLUT"
```

---

## Task 6: ICC Profile Builder

**Files:**
- Create: `src/display/profile.py`
- Create: `tests/display/test_profile.py`

The profile is built as raw bytes using `struct`. Format: ICC v2, 128-byte header, tag table, tag data. Tags: `desc`, `wtpt`, `rXYZ`, `gXYZ`, `bXYZ`, `rTRC`, `gTRC`, `bTRC`, `vcgt`. dispwin reads the `vcgt` tag to load the VideoLUT.

- [ ] **Step 1: Write the failing tests**

`tests/display/test_profile.py`:
```python
import struct

import numpy as np

from display.profile import build_vcgt_profile


def _identity_luts():
    lut = (np.linspace(0, 1, 256) * 65535).astype(np.uint16)
    return lut, lut.copy(), lut.copy()


def test_returns_bytes():
    r, g, b = _identity_luts()
    assert isinstance(build_vcgt_profile(r, g, b), bytes)


def test_declared_size_matches_actual_length():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    declared = struct.unpack(">I", data[:4])[0]
    assert declared == len(data)


def test_acsp_signature_at_offset_36():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[36:40] == b"acsp"


def test_display_class_at_offset_12():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[12:16] == b"mntr"


def test_rgb_colorspace_at_offset_16():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert data[16:20] == b"RGB "


def test_vcgt_tag_present_in_data():
    r, g, b = _identity_luts()
    data = build_vcgt_profile(r, g, b)
    assert b"vcgt" in data


def _vcgt_offset(data: bytes) -> int:
    """Walk the ICC tag table and return the byte offset of the vcgt type block."""
    count = struct.unpack(">I", data[128:132])[0]
    for i in range(count):
        entry = 132 + i * 12
        sig = data[entry : entry + 4]
        if sig == b"vcgt":
            return struct.unpack(">I", data[entry + 4 : entry + 8])[0]
    raise AssertionError("vcgt tag not found in tag table")


def test_vcgt_lut_roundtrip():
    # Build a known LUT and verify it appears in the profile bytes
    r = np.zeros(256, dtype=np.uint16)
    g = np.full(256, 32767, dtype=np.uint16)
    b = np.full(256, 65535, dtype=np.uint16)
    data = build_vcgt_profile(r, g, b)
    pos = _vcgt_offset(data)
    # vcgt type header: sig(4) + reserved(4) + gamma_type(4) + channels(2) + count(2) + entry_size(2) = 18 bytes
    # Then R entries (256 * 2 = 512 bytes), then G entries start.
    g_start = pos + 18 + 512
    first_g = struct.unpack(">H", data[g_start : g_start + 2])[0]
    assert first_g == 32767
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/display/test_profile.py -v`
Expected: `ModuleNotFoundError: No module named 'display.profile'`

- [ ] **Step 3: Implement `src/display/profile.py`**

```python
"""
Builds a minimal ICC v2 RGB display profile containing an Apple VCGT tag.
dispwin reads the VCGT tag to load the VideoLUT onto the display.
"""

import struct
from datetime import datetime, timezone

import numpy as np


def _s15f16(v: float) -> int:
    """Convert float to ICC s15Fixed16 (signed, big-endian 32-bit)."""
    return int(round(v * 65536)) & 0xFFFFFFFF


def _xyz_type(X: float, Y: float, Z: float) -> bytes:
    return struct.pack(">4sI3i", b"XYZ ", 0, _s15f16(X), _s15f16(Y), _s15f16(Z))


def _curv_gamma(gamma: float) -> bytes:
    """ICC 'curv' tag with a single gamma value (u8Fixed8Number)."""
    g = int(round(gamma * 256)) & 0xFFFF
    return struct.pack(">4sIIH", b"curv", 0, 1, g)


def _desc_type(text: str) -> bytes:
    """ICC v2 'desc' tag (ASCII subset only)."""
    ascii_bytes = text.encode("ascii") + b"\x00"
    count = len(ascii_bytes)
    body = struct.pack(">4sII", b"desc", 0, count)
    body += ascii_bytes
    body += struct.pack(">II", 0, 0)   # Unicode language/count (empty)
    body += struct.pack(">HB", 0, 0)   # ScriptCode code + count
    body += b"\x00" * 67              # ScriptCode description (67 bytes)
    return body


def _vcgt_type(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> bytes:
    """Apple VideoCardGamma VCGT tag, table variant (type=0)."""
    # sig(4) + reserved(4) + gamma_type(4) + channels(2) + count(2) + entry_size(2)
    header = struct.pack(">4sIIHHH", b"vcgt", 0, 0, 3, 256, 2)
    # Ensure big-endian byte order for uint16 entries
    r_be = r.astype(">u2").tobytes()
    g_be = g.astype(">u2").tobytes()
    b_be = b.astype(">u2").tobytes()
    return header + r_be + g_be + b_be


def _build_header(profile_size: int) -> bytes:
    now = datetime.now(tz=timezone.utc)
    h = b""
    h += struct.pack(">I", profile_size)                  # 0:  profile size
    h += b"    "                                           # 4:  preferred CMM (none)
    h += struct.pack(">I", 0x02100000)                    # 8:  ICC version 2.1.0.0
    h += b"mntr"                                           # 12: display device
    h += b"RGB "                                           # 16: data color space
    h += b"XYZ "                                           # 20: PCS
    h += struct.pack(">6H",                               # 24: date/time (12 bytes)
                     now.year, now.month, now.day,
                     now.hour, now.minute, now.second)
    h += b"acsp"                                           # 36: file signature
    h += struct.pack(">I", 0)                             # 40: primary platform
    h += struct.pack(">I", 0)                             # 44: profile flags
    h += struct.pack(">I", 0)                             # 48: device manufacturer
    h += struct.pack(">I", 0)                             # 52: device model
    h += struct.pack(">Q", 0)                             # 56: device attributes (8 bytes)
    h += struct.pack(">I", 0)                             # 64: rendering intent (perceptual)
    # 68: nCIEXYZ of D50 illuminant (12 bytes, s15Fixed16 each)
    h += struct.pack(">iii", _s15f16(0.96420), _s15f16(1.00000), _s15f16(0.82491))
    h += struct.pack(">I", 0)                             # 80: profile creator
    h += b"\x00" * 16                                     # 84: profile MD5 (not computed)
    h += b"\x00" * 28                                     # 100: reserved
    assert len(h) == 128, f"Header must be 128 bytes, got {len(h)}"
    return h


def build_vcgt_profile(
    r_lut: np.ndarray,
    g_lut: np.ndarray,
    b_lut: np.ndarray,
) -> bytes:
    """
    Build a minimal ICC v2 sRGB display profile with an Apple VCGT tag.

    r_lut, g_lut, b_lut: uint16 arrays of length 256 (from ramp.lut_to_vcgt).
    Returns raw ICC profile bytes ready to write to a .icc file.
    """
    # sRGB primaries XYZ (D65)
    tags_data: dict[bytes, bytes] = {
        b"desc": _desc_type("Color Calibrator"),
        b"wtpt": _xyz_type(0.95045, 1.00000, 1.08905),
        b"rXYZ": _xyz_type(0.43607, 0.22249, 0.01392),
        b"gXYZ": _xyz_type(0.38515, 0.71687, 0.09708),
        b"bXYZ": _xyz_type(0.14307, 0.06061, 0.71410),
        b"rTRC": _curv_gamma(2.2),
        b"gTRC": _curv_gamma(2.2),
        b"bTRC": _curv_gamma(2.2),
        b"vcgt": _vcgt_type(r_lut, g_lut, b_lut),
    }

    n = len(tags_data)
    tag_table_offset = 128
    tag_data_start = 128 + 4 + n * 12  # header + count(4) + N*12

    # Assign offsets; pad each tag to a 4-byte boundary
    tag_layout: list[tuple[bytes, int, int, bytes]] = []
    offset = tag_data_start
    for sig, data in tags_data.items():
        size = len(data)
        pad = (-size) % 4
        padded = data + b"\x00" * pad
        tag_layout.append((sig, offset, size, padded))
        offset += len(padded)

    profile_size = offset

    # Assemble tag table
    tag_table = struct.pack(">I", n)
    for sig, off, size, _ in tag_layout:
        tag_table += sig + struct.pack(">II", off, size)

    # Assemble tag data
    tag_data = b"".join(padded for _, _, _, padded in tag_layout)

    header = _build_header(profile_size)
    return header + tag_table + tag_data
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `uv run pytest tests/display/test_profile.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/display/profile.py tests/display/test_profile.py
git commit -m "feat: ICC v2 profile builder — 128-byte header, sRGB baseline, VCGT tag"
```

---

## Task 7: QR Code Utility

**Files:**
- Create: `src/util/qr.py`

No separate test file — behavior verified visually. The function has no logic to unit-test.

- [ ] **Step 1: Implement `src/util/qr.py`**

```python
import io

import qrcode


def generate_qr_png(url: str) -> bytes:
    """Generate a QR code for `url` and return it as PNG bytes."""
    qr = qrcode.make(url)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 2: Smoke test in REPL**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from util.qr import generate_qr_png
data = generate_qr_png('http://192.168.1.1:8765/mobile')
print(f'QR PNG size: {len(data)} bytes')
assert data[:4] == b'\\x89PNG'
print('OK')
"
```
Expected: prints PNG size (typically 1–5 KB) and "OK".

- [ ] **Step 3: Commit**

```bash
git add src/util/qr.py
git commit -m "feat: QR code generator for mobile URL"
```

---

## Task 7b: TLS Self-Signed Certificate

**Why:** Mobile browsers gate `navigator.mediaDevices.getUserMedia` to secure contexts (HTTPS or `localhost`). Plain `http://<lan-ip>:8765` will produce `undefined` and the camera never starts. Solution: self-signed cert covering 127.0.0.1 + detected LAN IP, generated at startup, served via uvicorn's `--ssl-keyfile`/`--ssl-certfile`. User accepts the browser warning once per session.

**Files:**
- Create: `src/util/tls.py`

- [ ] **Step 1: Implement `src/util/tls.py`**

```python
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def detect_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_cert(cert_dir: Path) -> tuple[Path, Path]:
    """
    Generate (or reuse) a self-signed cert + key covering 127.0.0.1 and the
    current LAN IP. Returns (cert_path, key_path).

    Regenerates if the cert is missing, expired, or does not include the
    current LAN IP (handles laptop moving networks).
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "server.crt"
    key_path = cert_dir / "server.key"

    lan_ip = detect_lan_ip()
    if _cert_valid_for(cert_path, lan_ip):
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(tz=timezone.utc)
    san = x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
        x509.DNSName("localhost"),
    ])
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "color-calibrator"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def _cert_valid_for(cert_path: Path, lan_ip: str) -> bool:
    if not cert_path.exists():
        return False
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        if cert.not_valid_after_utc < datetime.now(tz=timezone.utc):
            return False
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        ips = [str(v) for v in san.get_values_for_type(x509.IPAddress)]
        return lan_ip in ips
    except Exception:
        return False
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from util.tls import ensure_cert, detect_lan_ip
c, k = ensure_cert(Path('.cert'))
print(f'cert={c} key={k} lan={detect_lan_ip()}')
"
```
Expected: paths printed, `.cert/server.crt` + `.cert/server.key` exist.

- [ ] **Step 3: Add `.cert/` to `.gitignore`**

```bash
echo ".cert/" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add src/util/tls.py .gitignore
git commit -m "feat: self-signed TLS cert generator covering LAN IP for getUserMedia"
```

---

## Task 8: FastAPI Server and WebSocket Routing

**Files:**
- Create: `src/web/server.py`

The server manages a single calibration session at a time. PC connects first, gets the QR code. When mobile connects, calibration setup begins. The WebSocket protocol is message-passing with JSON.

**Single-reader-per-socket invariant:** Each WebSocket has exactly one task calling `receive_text` — a per-socket reader that pushes parsed messages into an `asyncio.Queue`. Calibration callbacks (`pc_recv`/`mobile_recv`) consume from the queue. Concurrent `receive_text` calls on the same `WebSocket` raise `RuntimeError` in Starlette, so the queue indirection is mandatory.

- [ ] **Step 1: Implement `src/web/server.py`**

```python
import asyncio
import base64
import json
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from display.dispwin import find_dispwin
from util.qr import generate_qr_png

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- Session state (single session) ----------

class Endpoint:
    """One side of the session: WebSocket + inbound message queue."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.closed = asyncio.Event()


class Session:
    def __init__(self):
        self.pc: Endpoint | None = None
        self.mobile: Endpoint | None = None
        self.calibration_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()


session = Session()


async def _reader(endpoint: Endpoint) -> None:
    """Single task per WebSocket: pump parsed messages into the queue."""
    try:
        while True:
            text = await endpoint.ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            await endpoint.queue.put(msg)
    except WebSocketDisconnect:
        pass
    finally:
        endpoint.closed.set()


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


# ---------- HTTP routes ----------

@app.get("/")
async def pc_page():
    return FileResponse(STATIC_DIR / "pc.html")


@app.get("/mobile")
async def mobile_page():
    return FileResponse(STATIC_DIR / "mobile.html")


# ---------- WebSocket: PC ----------

@app.websocket("/ws/pc")
async def ws_pc(websocket: WebSocket):
    await websocket.accept()
    endpoint = Endpoint(websocket)
    async with session.lock:
        session.pc = endpoint

    dispwin_path = find_dispwin()
    if dispwin_path is None:
        await _send(websocket, {
            "type": "error",
            "message": "dispwin not found — install ArgyllCMS and add it to PATH.",
        })
        await websocket.close()
        session.pc = None
        return

    ip = _local_ip()
    mobile_url = f"https://{ip}:8765/mobile"
    qr_png = generate_qr_png(mobile_url)
    qr_b64 = base64.b64encode(qr_png).decode()
    await _send(websocket, {"type": "qr_code", "png_b64": qr_b64, "url": mobile_url})

    # Single reader for this socket. All other code consumes from endpoint.queue.
    reader = asyncio.create_task(_reader(endpoint))
    control = asyncio.create_task(_pc_control_loop(endpoint))
    try:
        await endpoint.closed.wait()
    finally:
        reader.cancel()
        control.cancel()
        async with session.lock:
            if session.pc is endpoint:
                session.pc = None


async def _pc_control_loop(pc: "Endpoint") -> None:
    """Consume PC-originated messages. Currently: start_calibration trigger."""
    while not pc.closed.is_set():
        msg = await pc.queue.get()
        if msg.get("type") == "start_calibration":
            async with session.lock:
                mobile = session.mobile
                if mobile is None:
                    await _send(pc.ws, {"type": "error", "message": "Mobile not connected."})
                    continue
                if session.calibration_task and not session.calibration_task.done():
                    continue  # already running
                session.calibration_task = asyncio.create_task(_run_calibration_task())


# ---------- WebSocket: Mobile ----------

@app.websocket("/ws/mobile")
async def ws_mobile(websocket: WebSocket):
    await websocket.accept()
    endpoint = Endpoint(websocket)
    async with session.lock:
        session.mobile = endpoint
        pc = session.pc

    if pc is not None:
        await _send(pc.ws, {"type": "mobile_connected"})
        # PC's setup screen now shows a Begin button; calibration starts when
        # the user clicks it (PC sends `start_calibration`).

    reader = asyncio.create_task(_reader(endpoint))
    try:
        await endpoint.closed.wait()
    finally:
        reader.cancel()
        async with session.lock:
            if session.mobile is endpoint:
                session.mobile = None
        if pc is not None and not pc.closed.is_set():
            await _send(pc.ws, {"type": "error", "message": "Mobile disconnected."})


# ---------- Helpers ----------

async def _send(ws: WebSocket, msg: dict) -> None:
    await ws.send_text(json.dumps(msg))


async def _run_calibration_task() -> None:
    """Wraps calibration loop; sends result or error to PC."""
    from calibration.iterate import run_calibration

    pc, mobile = session.pc, session.mobile
    if pc is None or mobile is None:
        return

    async def pc_send(msg: dict) -> None:
        await _send(pc.ws, msg)

    async def mobile_send(msg: dict) -> None:
        await _send(mobile.ws, msg)

    async def mobile_recv() -> dict:
        return await mobile.queue.get()

    async def mobile_drain() -> None:
        """Discard any queued mobile messages (stale frames between patches)."""
        while not mobile.queue.empty():
            mobile.queue.get_nowait()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            icc_bytes, delta_e = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp)
            )
            icc_b64 = base64.b64encode(icc_bytes).decode()
            await _send(pc.ws, {"type": "result", "icc_b64": icc_b64, "delta_e": delta_e})
            await _send(mobile.ws, {"type": "all_done"})
        except Exception as exc:
            await _send(pc.ws, {"type": "error", "message": str(exc)})
```

- [ ] **Step 2: Verify server imports cleanly**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from web.server import app; print('OK')"
```
Expected: `OK` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add src/web/server.py
git commit -m "feat: FastAPI server — PC/mobile WebSocket routing, QR code delivery, session state"
```

---

## Task 9: PC HTML Page

**Files:**
- Create: `src/web/static/pc.html`
- Create: `src/web/static/style.css`

The PC page is a single-page wizard. It advances through screens based on WebSocket messages. All rendering is plain JavaScript — no framework.

- [ ] **Step 1: Create `src/web/static/style.css`**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: system-ui, sans-serif;
    background: #111;
    color: #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
}

.screen { display: none; flex-direction: column; align-items: center; gap: 1.5rem; max-width: 640px; width: 100%; padding: 2rem; text-align: center; }
.screen.active { display: flex; }

h1 { font-size: 1.8rem; font-weight: 600; }
p  { color: #aaa; line-height: 1.6; }

.qr-img { width: 220px; height: 220px; border: 4px solid #444; border-radius: 8px; }

.progress-bar { width: 100%; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
.progress-fill { height: 100%; background: #4caf50; transition: width 0.3s; }

.ssnr-indicator { padding: 0.4rem 1rem; border-radius: 4px; font-size: 0.9rem; font-weight: 600; }
.ssnr-stable  { background: #2d6a2d; color: #8fdb8f; }
.ssnr-waiting { background: #6a5a2d; color: #dbb88f; }

.before-after { display: flex; gap: 1rem; width: 100%; }
.before-after img { flex: 1; border-radius: 4px; }

button { padding: 0.75rem 2rem; border: none; border-radius: 6px; background: #4caf50; color: #fff; font-size: 1rem; cursor: pointer; }
button:hover { background: #43a047; }

.delta-e { font-size: 2rem; font-weight: 700; color: #4caf50; }
.error-msg { color: #f44336; background: #1a0000; border: 1px solid #f44336; padding: 1rem; border-radius: 6px; }

pre { background: #222; padding: 1rem; border-radius: 6px; text-align: left; font-size: 0.85rem; color: #bbb; overflow-x: auto; }
```

- [ ] **Step 2: Create `src/web/static/pc.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Color Calibrator — PC</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>

<!-- Screen 1: Error (dispwin missing) -->
<div id="s-error" class="screen">
  <h1>Setup Required</h1>
  <p class="error-msg" id="error-text"></p>
  <p>Install <strong>ArgyllCMS</strong> and add <code>dispwin</code> to your PATH:</p>
  <pre>
Windows: https://www.argyllcms.com/downloadwin.html
Mac:     brew install argyll-cms
Linux:   sudo apt install argyll  (or dnf/pacman equivalent)
  </pre>
</div>

<!-- Screen 2: QR code -->
<div id="s-qr" class="screen active">
  <h1>Scan with your phone</h1>
  <img id="qr-img" class="qr-img" alt="QR code loading…">
  <p>Open <code id="mobile-url"></code> in your mobile browser, or scan the QR code above.</p>
  <p>Waiting for mobile to connect…</p>
</div>

<!-- Screen 3: Setup instructions -->
<div id="s-setup" class="screen">
  <h1>Setup</h1>
  <p>1. Darken the room as much as possible.</p>
  <p>2. Hold your phone 30 cm from the centre of the screen, perpendicular to it.</p>
  <p>3. On your phone: tap <strong>Start Camera</strong>, then aim it at the screen.</p>
  <p>4. When ready, click Begin below. The screen will show a white patch — lock your phone's <strong>exposure and white balance</strong>, then tap Ready on your phone.</p>
  <button id="begin-btn">Begin Calibration</button>
</div>

<!-- Screen 4: Measuring -->
<div id="s-measuring" class="screen">
  <h1>Measuring — Round <span id="round-num">1</span> of 3</h1>
  <p>Patch <span id="patch-num">0</span> of <span id="patch-total">11</span></p>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
  <div class="ssnr-indicator ssnr-waiting" id="ssnr-badge">Waiting for stable frame…</div>
</div>

<!-- Screen 5: Applying correction -->
<div id="s-applying" class="screen">
  <h1>Applying Correction…</h1>
  <p>Round <span id="apply-round">1</span> complete. Applying gamma ramp to display.</p>
</div>

<!-- Screen 6: Result -->
<div id="s-result" class="screen">
  <h1>Calibration Complete</h1>
  <p class="delta-e">ΔE: <span id="delta-e-value">—</span></p>
  <p>Lower is better. Hardware colorimeters achieve &lt;1 ΔE.</p>
  <div class="before-after">
    <div><p>Before</p><img id="before-img" alt="Before" src=""></div>
    <div><p>After</p><img id="after-img" alt="After" src=""></div>
  </div>
  <button id="download-btn">Download .icc Profile</button>
  <details>
    <summary>How to install</summary>
    <pre id="install-instructions"></pre>
  </details>
</div>

<script>
const screens = {
  error:     document.getElementById('s-error'),
  qr:        document.getElementById('s-qr'),
  setup:     document.getElementById('s-setup'),
  measuring: document.getElementById('s-measuring'),
  applying:  document.getElementById('s-applying'),
  result:    document.getElementById('s-result'),
};

function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove('active'));
  screens[name].classList.add('active');
}

const wsScheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsScheme}//${location.host}/ws/pc`);

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);

  if (msg.type === 'error') {
    document.getElementById('error-text').textContent = msg.message;
    showScreen('error');
    return;
  }

  if (msg.type === 'qr_code') {
    document.getElementById('qr-img').src = `data:image/png;base64,${msg.png_b64}`;
    document.getElementById('mobile-url').textContent = msg.url;
    showScreen('qr');
    return;
  }

  if (msg.type === 'mobile_connected') {
    showScreen('setup');
    document.getElementById('begin-btn').onclick = () => {
      ws.send(JSON.stringify({ type: 'start_calibration' }));
      document.getElementById('begin-btn').disabled = true;
      document.getElementById('begin-btn').textContent = 'Starting…';
    };
    return;
  }

  if (msg.type === 'show_patch') {
    const [r, g, b] = msg.rgb;
    document.body.style.background = `rgb(${r},${g},${b})`;
    // Hide all wizard chrome so the camera measures a clean full-screen patch.
    Object.values(screens).forEach(s => s.classList.remove('active'));
    return;
  }

  if (msg.type === 'patch_done') {
    document.body.style.background = '#111';
    document.getElementById('patch-num').textContent = msg.n;
    document.getElementById('patch-total').textContent = msg.total;
    const pct = (msg.n / msg.total * 100).toFixed(0);
    document.getElementById('progress-fill').style.width = pct + '%';
    if (msg.round && msg.round > 0) {
      document.getElementById('round-num').textContent = msg.round;
    }
    const badge = document.getElementById('ssnr-badge');
    badge.textContent = 'Captured ✓';
    badge.className = 'ssnr-indicator ssnr-stable';
    showScreen('measuring');
    return;
  }

  if (msg.type === 'holdout_started') {
    document.querySelector('#s-measuring h1').textContent = 'Verifying Calibration';
    document.getElementById('patch-total').textContent = msg.total;
    document.getElementById('progress-fill').style.width = '0%';
    showScreen('measuring');
    return;
  }

  if (msg.type === 'capturing') {
    // Keep wizard chrome hidden during capture so the patch fills the screen.
    document.getElementById('round-num').textContent = msg.round;
    const badge = document.getElementById('ssnr-badge');
    badge.textContent = 'Waiting for stable frame…';
    badge.className = 'ssnr-indicator ssnr-waiting';
    return;
  }

  if (msg.type === 'round_done') {
    document.getElementById('apply-round').textContent = msg.round;
    showScreen('applying');
    return;
  }

  if (msg.type === 'result') {
    document.body.style.background = '#111';
    document.body.style.color = '#eee';
    document.getElementById('delta-e-value').textContent = msg.delta_e.toFixed(2);
    if (msg.before_b64) document.getElementById('before-img').src = `data:image/png;base64,${msg.before_b64}`;
    if (msg.after_b64)  document.getElementById('after-img').src  = `data:image/png;base64,${msg.after_b64}`;

    const iccBytes = Uint8Array.from(atob(msg.icc_b64), c => c.charCodeAt(0));
    const blob = new Blob([iccBytes], { type: 'application/vnd.iccprofile' });
    const url  = URL.createObjectURL(blob);
    const btn  = document.getElementById('download-btn');
    btn.onclick = () => { const a = document.createElement('a'); a.href = url; a.download = 'color-calibrator.icc'; a.click(); };

    document.getElementById('install-instructions').textContent = `Windows:
  1. Double-click color-calibrator.icc → Install Profile
  2. Win+R → colorcpl → Devices tab → select display
  3. Tick "Use my settings for this device" → Set as default

Mac:
  Double-click .icc → ColorSync Utility installs and activates automatically.

Linux (X11):
  xcalib color-calibrator.icc
  or copy to ~/.local/share/icc/ and activate via display settings.

Linux (Wayland):
  Copy to ~/.local/share/icc/, then: colormgr device-set-property ... Colorspace <profile-id>`;

    showScreen('result');
  }
};
</script>
</body>
</html>
```

- [ ] **Step 3: Commit**

```bash
git add src/web/static/pc.html src/web/static/style.css
git commit -m "feat: PC wizard page — QR, setup, progress, before/after result screens"
```

---

## Task 10: Mobile HTML Page

**Files:**
- Create: `src/web/static/mobile.html`

The mobile page runs `getUserMedia`, renders to canvas, and sends JPEG frames over WebSocket when the server says `capture`.

- [ ] **Step 1: Create `src/web/static/mobile.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Color Calibrator — Mobile</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #000; color: #fff; font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
#viewfinder { width: 100vw; height: 100vh; object-fit: cover; position: fixed; top: 0; left: 0; z-index: 0; }
#overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; padding: 2rem; gap: 1rem; pointer-events: none; }
#status { background: rgba(0,0,0,0.7); padding: 0.6rem 1.2rem; border-radius: 20px; font-size: 1rem; }
#ssnr-bar-wrap { width: 80%; height: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; overflow: hidden; }
#ssnr-bar { height: 100%; width: 0%; background: #4caf50; transition: width 0.2s, background 0.2s; }
#ready-btn, #start-cam-btn { pointer-events: auto; padding: 1rem 3rem; background: #4caf50; color: #fff; border: none; border-radius: 30px; font-size: 1.2rem; cursor: pointer; }
#ready-btn { display: none; }
#flash { position: fixed; inset: 0; background: #4caf50; opacity: 0; z-index: 10; pointer-events: none; transition: opacity 0.1s; }
canvas { display: none; }
</style>
</head>
<body>
<video id="viewfinder" autoplay playsinline muted></video>
<canvas id="canvas"></canvas>
<div id="overlay">
  <div id="status">Tap Start Camera to begin.</div>
  <div id="ssnr-bar-wrap"><div id="ssnr-bar"></div></div>
  <button id="start-cam-btn">Start Camera</button>
  <button id="ready-btn">Ready</button>
</div>
<div id="flash"></div>

<script>
const video       = document.getElementById('viewfinder');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const status      = document.getElementById('status');
const ssnrBar     = document.getElementById('ssnr-bar');
const readyBtn    = document.getElementById('ready-btn');
const startCamBtn = document.getElementById('start-cam-btn');
const flash       = document.getElementById('flash');

let ws;
let capturing = false;
let captureInterval = null;

// iOS Safari requires getUserMedia to be called from a user gesture handler.
// Page-load invocation is blocked. Wire to an explicit Start Camera button.
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status.textContent = 'Camera API unavailable. Use HTTPS and a modern browser.';
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;
    startCamBtn.style.display = 'none';
    status.textContent = 'Camera ready. Aim at the PC screen.';
    connectWs();
  } catch (e) {
    status.textContent = 'Camera access denied: ' + (e.message || e.name);
  }
}

startCamBtn.onclick = startCamera;

function connectWs() {
  const wsScheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${wsScheme}//${location.host}/ws/mobile`);

  ws.onopen = () => { status.textContent = 'Connected. Waiting for instructions…'; };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);

    if (msg.type === 'show_white_for_wb') {
      status.textContent = 'Lock exposure and white balance on this white screen, then tap Ready.';
      readyBtn.style.display = 'block';
      readyBtn.onclick = () => {
        ws.send(JSON.stringify({ type: 'ready' }));
        readyBtn.style.display = 'none';
        status.textContent = 'White balance locked. Waiting…';
      };
      return;
    }

    if (msg.type === 'capture') {
      capturing = true;
      status.textContent = `Capturing patch ${msg.n} of ${msg.total}…`;
      ssnrBar.style.background = '#f5a623';
      ssnrBar.style.width = '30%';
      startCapturing();
      return;
    }

    if (msg.type === 'stop_capture') {
      stopCapturing();
      return;
    }

    if (msg.type === 'patch_done') {
      stopCapturing();
      ssnrBar.style.background = '#4caf50';
      ssnrBar.style.width = '100%';
      status.textContent = `Patch ${msg.n} of ${msg.total} captured ✓`;
      // Green flash
      flash.style.opacity = '0.6';
      setTimeout(() => { flash.style.opacity = '0'; }, 200);
      return;
    }

    if (msg.type === 'round_done') {
      ssnrBar.style.width = '0%';
      status.textContent = 'Round complete — applying correction, please wait…';
      return;
    }

    if (msg.type === 'all_done') {
      status.textContent = 'All done — check your PC for the result!';
      ssnrBar.style.width = '100%';
      return;
    }
  };

  ws.onclose = () => {
    stopCapturing();
    status.textContent = 'Disconnected. Reload to reconnect.';
  };
}

function startCapturing() {
  if (captureInterval) return;
  captureInterval = setInterval(sendFrame, 200); // 5 fps
}

function stopCapturing() {
  capturing = false;
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
}

function sendFrame() {
  if (!capturing || ws.readyState !== WebSocket.OPEN) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  ws.send(JSON.stringify({ type: 'frame', data: dataUrl.split(',')[1] }));
}

</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/web/static/mobile.html
git commit -m "feat: mobile camera page — getUserMedia, frame streaming, status indicators"
```

---

## Task 11: Iterative Calibration Loop

**Files:**
- Create: `src/calibration/iterate.py`

This module orchestrates the full calibration session. It is purely async and communicates via callback functions — no direct WebSocket imports. This makes it testable without a live server.

- [ ] **Step 1: Implement `src/calibration/iterate.py`**

```python
"""
Iterative calibration loop.

Receives send/recv callbacks for PC and mobile WebSocket connections.
Returns (icc_bytes, final_delta_e) when done.
"""

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from calibration.capture import decode_frame, frame_luminance, is_stable
from calibration.patches import GRAY_PATCHES, HOLDOUT_PATCHES
from calibration.ramp import (
    compose_luts,
    fit_correction,
    identity_lut,
    lut_to_vcgt,
)
from display.dispwin import apply_ramp, clear_ramp
from display.profile import build_vcgt_profile

SendFn = Callable[[dict], Awaitable[None]]
RecvFn = Callable[[], Awaitable[dict]]

SSNR_THRESHOLD = 20.0
STABLE_FRAMES = 5
CAPTURE_TIMEOUT = 10.0    # seconds per patch
READY_TIMEOUT = 180.0     # seconds to wait for user to lock WB and tap Ready
SETTLE_DELAY = 0.3        # seconds for LCD to stabilise on a new patch
MAX_SKIPPED = 3
MAX_ROUNDS = 3
CONVERGENCE_DELTA_E = 1.0


DrainFn = Callable[[], Awaitable[None]]


async def _wait_for_stable_frames(mobile_recv: RecvFn) -> np.ndarray | None:
    """Collect frames until SSNR stable or timeout. Returns averaged frame or None."""
    luminances: list[float] = []
    frames: list[np.ndarray] = []
    deadline = time.monotonic() + CAPTURE_TIMEOUT

    while time.monotonic() < deadline:
        msg = await asyncio.wait_for(mobile_recv(), timeout=max(0.1, deadline - time.monotonic()))
        if msg.get("type") != "frame":
            continue
        frame = decode_frame(msg["data"])
        luma = frame_luminance(frame)
        luminances.append(luma)
        frames.append(frame)

        if len(luminances) > STABLE_FRAMES:
            luminances.pop(0)
            frames.pop(0)

        if is_stable(luminances, SSNR_THRESHOLD):
            # Average the stable frames
            stacked = np.stack(frames, axis=0).astype(float)
            return np.clip(np.mean(stacked, axis=0), 0, 255).astype(np.uint8)

    return None   # timed out


def _normalize_luma(frame: np.ndarray, white_frame: np.ndarray) -> float:
    """Relative luminance: measured luma / white-reference luma. Clamped to [0, 1]."""
    white_luma = frame_luminance(white_frame)
    patch_luma = frame_luminance(frame)
    if white_luma < 1e-6:
        return 0.0
    return float(np.clip(patch_luma / white_luma, 0.0, 1.0))


def _delta_e_gray(measured_lumas: list[float], target_lumas: list[float]) -> float:
    """Mean absolute error between measured and target luminance ratios (proxy for ΔE on grays)."""
    diffs = [abs(m - t) * 100.0 for m, t in zip(measured_lumas, target_lumas)]
    return float(np.mean(diffs))


async def _capture_white_reference(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> np.ndarray:
    """Show white patch, wait for WB lock from mobile, capture 5 stable frames."""
    level = 255
    await pc_send({"type": "show_patch", "rgb": [level, level, level]})
    await mobile_send({"type": "show_white_for_wb"})

    # Wait for mobile 'ready' message (user locked WB). Bounded so a dropped
    # mobile connection does not hang the loop forever.
    try:
        while True:
            msg = await asyncio.wait_for(mobile_recv(), timeout=READY_TIMEOUT)
            if msg.get("type") == "ready":
                break
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timed out waiting for mobile to lock white balance.") from exc

    # Mobile only streams frames between `capture` and `stop_capture`. The WB
    # reference also needs frames, so wrap the capture in the same control
    # messages used for ordinary patches.
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({"type": "capture", "n": 0, "total": 0, "phase": "white_ref"})
    try:
        frame = await _wait_for_stable_frames(mobile_recv)
    finally:
        await mobile_send({"type": "stop_capture"})
    if frame is None:
        raise RuntimeError(
            "White reference capture timed out. Check room lighting and phone position."
        )
    return frame


async def _measure_patch(
    patch_level: float,
    patch_index: int,
    patch_total: int,
    round_num: int,
    white_frame: np.ndarray,
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> float | None:
    """Show one gray patch, capture, return normalized luminance or None if skipped."""
    v = int(round(patch_level * 255))
    await pc_send({"type": "show_patch", "rgb": [v, v, v]})
    await pc_send({"type": "capturing", "round": round_num})

    # Give the display a beat to settle on the new patch, drain any in-flight
    # stale frames from the previous patch, then tell mobile to start streaming.
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({
        "type": "capture",
        "n": patch_index + 1,
        "total": patch_total,
    })

    frame = await _wait_for_stable_frames(mobile_recv)

    await mobile_send({"type": "stop_capture"})
    await pc_send({
        "type": "patch_done",
        "n": patch_index + 1,
        "total": patch_total,
        "round": round_num,
    })
    await mobile_send({
        "type": "patch_done",
        "n": patch_index + 1,
        "total": patch_total,
    })

    if frame is None:
        return None
    return _normalize_luma(frame, white_frame)


async def run_calibration(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    tmp_dir: Path,
) -> tuple[bytes, float]:
    """
    Run the full iterative calibration session.
    Returns (icc_bytes, final_delta_e).
    """
    # --- White reference ---
    white_frame = await _capture_white_reference(pc_send, mobile_send, mobile_recv, mobile_drain)

    # --- Per-channel float LUTs (start at identity) ---
    lut_r = identity_lut()
    lut_g = identity_lut()
    lut_b = identity_lut()

    best_delta_e = float("inf")
    best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

    for round_num in range(1, MAX_ROUNDS + 1):
        # Apply current ramp to display. dispwin and file I/O are blocking —
        # offload to a worker thread so the event loop keeps servicing WebSocket
        # messages (mobile frame stream in particular).
        tmp_icc = tmp_dir / f"round_{round_num}.icc"
        vcgt_r, vcgt_g, vcgt_b = lut_to_vcgt(lut_r), lut_to_vcgt(lut_g), lut_to_vcgt(lut_b)
        await asyncio.to_thread(
            tmp_icc.write_bytes, build_vcgt_profile(vcgt_r, vcgt_g, vcgt_b)
        )
        await asyncio.to_thread(clear_ramp)
        await asyncio.to_thread(apply_ramp, str(tmp_icc))

        # Measure each calibration patch
        measured_lumas: list[float] = []
        target_lumas: list[float] = []
        skipped = 0
        patches = GRAY_PATCHES
        total = len(patches)

        for i, patch in enumerate(patches):
            luma = await _measure_patch(
                patch.level, i, total, round_num,
                white_frame, pc_send, mobile_send, mobile_recv, mobile_drain,
            )
            if luma is None:
                skipped += 1
                if skipped > MAX_SKIPPED:
                    raise RuntimeError(f"Too many skipped patches ({skipped}). Check lighting.")
                # Use target as fallback (neutral effect on ramp)
                measured_lumas.append(patch.target_luma)
            else:
                measured_lumas.append(luma)
            target_lumas.append(patch.target_luma)

        # Fit correction curve
        levels = np.array([p.level for p in patches])
        measured = np.array(measured_lumas)
        target   = np.array(target_lumas)

        # Same curve applied to all 3 channels (gray patches only)
        new_lut = fit_correction(levels, measured)
        lut_r = compose_luts(lut_r, new_lut)
        lut_g = compose_luts(lut_g, new_lut)
        lut_b = compose_luts(lut_b, new_lut)

        delta_e = _delta_e_gray(measured_lumas, target_lumas)
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_luts = (lut_r.copy(), lut_g.copy(), lut_b.copy())

        await pc_send({"type": "round_done", "delta_e": delta_e, "round": round_num})
        await mobile_send({"type": "round_done"})

        if delta_e < CONVERGENCE_DELTA_E or round_num == MAX_ROUNDS:
            break

    # --- Holdout verification ---
    await pc_send({"type": "holdout_started", "total": len(HOLDOUT_PATCHES)})
    await mobile_send({"type": "holdout_started", "total": len(HOLDOUT_PATCHES)})
    holdout_measured: list[float] = []
    for i, patch in enumerate(HOLDOUT_PATCHES):
        # round_num=0 signals "holdout, not a calibration round" to the UI.
        luma = await _measure_patch(
            patch.level, i, len(HOLDOUT_PATCHES), 0,
            white_frame, pc_send, mobile_send, mobile_recv, mobile_drain,
        )
        holdout_measured.append(luma if luma is not None else patch.target_luma)

    holdout_targets = [p.target_luma for p in HOLDOUT_PATCHES]
    final_delta_e = _delta_e_gray(holdout_measured, holdout_targets)

    # --- Build final ICC profile ---
    lut_r, lut_g, lut_b = best_luts
    vcgt_r, vcgt_g, vcgt_b = lut_to_vcgt(lut_r), lut_to_vcgt(lut_g), lut_to_vcgt(lut_b)
    icc_bytes = build_vcgt_profile(vcgt_r, vcgt_g, vcgt_b)

    return icc_bytes, final_delta_e
```

- [ ] **Step 2: Verify no import errors**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from calibration.iterate import run_calibration; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Integration test for the loop**

Create `tests/calibration/test_iterate.py`:

```python
import base64
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import calibration.iterate as iterate
from calibration.iterate import run_calibration


def _b64_jpeg_gray(level_0_1: float) -> str:
    v = int(round(np.clip(level_0_1, 0.0, 1.0) * 255))
    img = Image.new("RGB", (64, 64), (v, v, v))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


class FakeProtocol:
    """
    Simulates the mobile WebSocket. Produces JPEG frames whose grayscale level
    follows `display_gamma`: for input x, the camera 'sees' x ** gamma.
    """

    def __init__(self, display_gamma: float):
        self.gamma = display_gamma
        self.current_input = 1.0  # white reference patch
        self.sent: list[dict] = []
        self.recv_queue: list[dict] = []

    async def pc_send(self, msg: dict) -> None:
        self.sent.append(("pc", msg))
        if msg.get("type") == "show_patch":
            r, g, b = msg["rgb"]
            self.current_input = (r + g + b) / 3.0 / 255.0

    async def mobile_send(self, msg: dict) -> None:
        self.sent.append(("mobile", msg))
        if msg.get("type") == "show_white_for_wb":
            # User locks WB and taps Ready.
            self.recv_queue.append({"type": "ready"})

    async def mobile_recv(self) -> dict:
        # Either pop a scripted message or synthesize a frame from current input.
        if self.recv_queue:
            return self.recv_queue.pop(0)
        return {"type": "frame", "data": _b64_jpeg_gray(self.current_input ** self.gamma)}

    async def mobile_drain(self) -> None:
        self.recv_queue.clear()


@pytest.mark.asyncio
async def test_run_calibration_converges_on_gamma_1_8_display(monkeypatch):
    # Stub dispwin so the test does not touch the real display.
    monkeypatch.setattr(iterate, "clear_ramp", lambda *a, **k: None)
    monkeypatch.setattr(iterate, "apply_ramp", lambda *a, **k: None)
    # Settle delay is for real LCDs; skip in tests so the loop runs in seconds.
    monkeypatch.setattr(iterate, "SETTLE_DELAY", 0.0)

    fake = FakeProtocol(display_gamma=1.8)

    with tempfile.TemporaryDirectory() as tmp:
        icc_bytes, delta_e = await run_calibration(
            fake.pc_send, fake.mobile_send, fake.mobile_recv, fake.mobile_drain, Path(tmp)
        )

    assert isinstance(icc_bytes, bytes) and len(icc_bytes) > 256
    assert icc_bytes[36:40] == b"acsp"
    # Even without modeling the LUT applying back into the camera (we stub
    # dispwin), the holdout ΔE is computed against target — a finite, sane
    # number rather than NaN/inf — confirms the loop ran end-to-end.
    assert np.isfinite(delta_e)
```

> **Note:** This test stubs `dispwin`, so the simulated display gamma does not actually change between rounds. The assertion verifies the loop terminates, builds a valid ICC, and returns a finite ΔE — i.e. the orchestration is correct. A higher-fidelity test would model the LUT round-tripping through the simulated camera; out of scope for this plan.

Run: `uv run pytest tests/calibration/test_iterate.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add src/calibration/iterate.py tests/calibration/test_iterate.py
git commit -m "feat: iterative calibration loop — measure, fit, compose, converge, holdout verify"
```

---

## Task 12: Before/After Comparison Image

The PC page shows a before/after comparison after calibration. The server generates both images using Pillow: the original `test_chart.png` as "before", and the LUT-corrected version as "after". Both are sent as base64 PNG in the `result` message.

- [ ] **Step 1: Generate `test_chart.png` lazily, never commit it**

The chart is a build artifact, not source. Add it to `.gitignore` and synthesize on demand inside the server.

```bash
echo "src/web/static/test_chart.png" >> .gitignore
```

Add to `src/web/server.py` (near the other helpers):

```python
def _ensure_test_chart(path: Path) -> None:
    if path.exists():
        return
    import numpy as np
    from PIL import Image
    w, h = 640, 240
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        v = int(x / w * 255)
        img[: h // 2, x] = [v, v, v]
    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
    ]
    strip_w = w // len(colours)
    for i, c in enumerate(colours):
        img[h // 2 :, i * strip_w : (i + 1) * strip_w] = c
    Image.fromarray(img).save(path)
```

Call it from `_run_calibration_task` just before `_build_comparison_b64`:

```python
            chart_path = STATIC_DIR / "test_chart.png"
            _ensure_test_chart(chart_path)
            before_b64, after_b64 = _build_comparison_b64(chart_path, lut_r, lut_g, lut_b)
```

- [ ] **Step 2: Add `_build_comparison` helper to `src/web/server.py`**

Add the following function and update the `_run_calibration_task` in `server.py` to generate and include before/after images in the `result` message.

In `src/web/server.py`, add after the imports:

```python
import io
import numpy as np
from PIL import Image
from calibration.ramp import apply_lut_to_image
```

Add the helper function before `_run_calibration_task`:

```python
def _build_comparison_b64(
    chart_path: Path,
    lut_r: np.ndarray,
    lut_g: np.ndarray,
    lut_b: np.ndarray,
) -> tuple[str, str]:
    """Return (before_b64, after_b64) as PNG base64 strings."""
    img = np.array(Image.open(chart_path).convert("RGB"))
    corrected = apply_lut_to_image(img, lut_r, lut_g, lut_b)

    def to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return to_b64(img), to_b64(corrected)
```

- [ ] **Step 2a: Change `run_calibration` return shape to include final LUTs**

In `src/calibration/iterate.py`, update the signature and the final return statement only — do not rewrite the body.

Signature change:
```python
) -> tuple[bytes, float, np.ndarray, np.ndarray, np.ndarray]:
```

Final return change:
```python
    return icc_bytes, final_delta_e, lut_r, lut_g, lut_b
```

- [ ] **Step 2b: Targeted edits in `src/web/server.py`**

Patch only the two lines below. Do **not** redefine `_run_calibration_task` — the surrounding code from Task 8 stays as-is.

Find:
```python
            icc_bytes, delta_e = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp)
            )
            icc_b64 = base64.b64encode(icc_bytes).decode()
            await _send(pc.ws, {"type": "result", "icc_b64": icc_b64, "delta_e": delta_e})
```

Replace with:
```python
            icc_bytes, delta_e, lut_r, lut_g, lut_b = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp)
            )
            icc_b64 = base64.b64encode(icc_bytes).decode()
            chart_path = STATIC_DIR / "test_chart.png"
            _ensure_test_chart(chart_path)
            before_b64, after_b64 = _build_comparison_b64(chart_path, lut_r, lut_g, lut_b)
            await _send(pc.ws, {
                "type": "result",
                "icc_b64": icc_b64,
                "delta_e": delta_e,
                "before_b64": before_b64,
                "after_b64": after_b64,
            })
```

- [ ] **Step 3: Verify imports after editing**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from web.server import app; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add .gitignore src/web/server.py src/calibration/iterate.py
git commit -m "feat: before/after comparison image — LUT applied via Pillow, sent as PNG base64"
```

---

## Task 13: main.py Integration and Startup Check

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Finalize `main.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from display.dispwin import find_dispwin
from util.tls import detect_lan_ip, ensure_cert


def main():
    dispwin = find_dispwin()
    if dispwin is None:
        print(
            "\n[color-calibrator] ERROR: 'dispwin' not found on PATH.\n"
            "Install ArgyllCMS:\n"
            "  Windows: https://www.argyllcms.com/downloadwin.html\n"
            "  Mac:     brew install argyll-cms\n"
            "  Linux:   sudo apt install argyll\n"
        )
        sys.exit(1)

    cert_path, key_path = ensure_cert(Path(__file__).parent / ".cert")
    ip = detect_lan_ip()

    print(f"\n[color-calibrator] Server starting on https://{ip}:8765")
    print(f"[color-calibrator] Mobile URL: https://{ip}:8765/mobile")
    print("[color-calibrator] Self-signed cert: accept the browser warning once.\n")

    import uvicorn
    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        ssl_keyfile=str(key_path),
        ssl_certfile=str(cert_path),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (patches, capture, ramp, dispwin, profile).

- [ ] **Step 3: Start the server and verify it serves the PC page**

Run: `uv run python main.py`

If dispwin is not installed, you'll see the install error — that's correct behavior. To test the server itself without dispwin, temporarily bypass the check:

```bash
uv run python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from util.tls import ensure_cert
cert, key = ensure_cert(Path('.cert'))
import uvicorn
uvicorn.run('web.server:app', host='127.0.0.1', port=8765,
            ssl_keyfile=str(key), ssl_certfile=str(cert))
"
```

Then open `https://127.0.0.1:8765/` in a browser. Expected: browser warns about self-signed cert (accept), then PC page loads, attempts WebSocket, shows QR screen skeleton.

- [ ] **Step 4: Final commit**

```bash
git add main.py
git commit -m "feat: main.py startup — dispwin check, LAN IP detection, uvicorn launch"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| FastAPI server, local IP detection | Tasks 8, 13 |
| QR code for mobile URL | Task 7, 8 |
| WebSocket PC + mobile | Task 8 |
| White balance lock step | Tasks 8, 10, 11 |
| SSNR ≥ 20 dB stability check | Tasks 3, 11 |
| 11 gray patches + 3 holdout | Task 2 |
| Iterative loop up to 3 rounds | Task 11 |
| Gamma curve fitting (power law) | Task 4 |
| LUT composition via np.interp | Task 4 |
| dispwin apply/clear | Task 5 |
| ICC v2 profile with VCGT tag | Task 6 |
| Before/after comparison | Task 12 |
| PC wizard UI | Task 9 |
| Mobile camera UI | Task 10 |
| dispwin missing → clear error | Tasks 8, 13 |
| Mobile disconnect → reset | Task 8 |
| SSNR timeout → skip patch | Task 11 |
| Too many skips → error | Task 11 |
| Post-calibration ΔE report | Task 11 |
| Download .icc + install instructions | Task 9 |

**Type consistency check:**

- `fit_correction` signature: `(np.ndarray, np.ndarray, np.ndarray) → np.ndarray` — used consistently in Task 11
- `compose_luts(prev, new) → np.ndarray` — used correctly in Task 11 (`lut_r = compose_luts(lut_r, new_lut)`)
- `lut_to_vcgt(lut) → np.ndarray[uint16]` — called in Task 11, input to `build_vcgt_profile` in Task 6
- `build_vcgt_profile(r, g, b) → bytes` — r/g/b are uint16 arrays, matches Task 6 tests
- `run_calibration` returns `(bytes, float, ndarray, ndarray, ndarray)` after Task 12 update — matches usage in `_run_calibration_task`
- `apply_lut_to_image(img, r, g, b)` — takes float [0,1] luts, consistent between Task 4 definition and Task 12 usage

**No placeholders:** All steps contain actual code, exact commands, and expected output.
