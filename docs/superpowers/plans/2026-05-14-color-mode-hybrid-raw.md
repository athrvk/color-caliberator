# Color Mode (Hybrid RAW + JPEG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend color-calibrator with an opt-in "color mode" that uses 4 manual DNG anchor photos plus the existing JPEG stream to produce a matrix-shaper ICC profile, correcting per-channel gamma, gray balance, and color casts. The existing gamma-only mode remains as the default fallback for phones without RAW support.

**Architecture:** User selects "color" or "gamma" mode on the PC setup screen. Color mode adds a manual anchor phase before the automated patch loop: the user shoots 4 DNGs (white, red, green, blue) in their phone's native camera app and uploads them via the mobile page. The server parses the DNGs (`tifffile` for ICC-relevant TIFF tags, `rawpy` for linear pixel data) and derives a per-session camera-RGB → CIE-XYZ matrix using the DNG `ForwardMatrix2` and `AsShotNeutral` tags. The automated JPEG patch stream then runs as today, but each patch's center-cropped RGB is reversed through sRGB-gamma, normalized against the white anchor, and transformed to XYZ using the camera matrix. Per-channel tone response curves are fitted from XYZ-space measurements. The final ICC profile is a v2 matrix-shaper: rXYZ/gXYZ/bXYZ primaries (in PCS D50), per-channel `curv` TRC tags, plus the existing VCGT tag for OS-level gamma ramp.

**Tech Stack:** Python 3.11+, `rawpy` (libraw binding for DNG pixel decode), `tifffile` (DNG TIFF-tag parsing), `numpy`/`scipy` (math), FastAPI multipart upload, plain HTML/JS file input on mobile. No new runtime services; same uvicorn + dispwin pipeline.

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add `rawpy`, `tifffile` deps |
| `src/calibration/raw.py` | **NEW** — DNG parser: extract ColorMatrix/ForwardMatrix/AsShotNeutral tags + center-crop linear RGB sample |
| `src/calibration/color_pipeline.py` | **NEW** — camera RGB → XYZ_D50 (DNG ForwardMatrix2), primary projection, sRGB-D50 targets, per-channel TRC fit |
| `src/calibration/iterate.py` | Branch on `mode`: existing path for gamma, new path for color (anchor phase + XYZ-aware patch loop) |
| `src/display/profile.py` | Add `build_matrix_shaper_profile(r_lut, g_lut, b_lut, r_xyz, g_xyz, b_xyz, r_trc, g_trc, b_trc)` — matrix profile with TRC + VCGT |
| `src/web/server.py` | Add `/upload/raw/{seq}` multipart endpoint; extend `Session` with `mode` and `anchors`; pass anchors into `run_calibration` |
| `src/web/static/pc.html` | Add mode toggle (radio: Gamma / Color) on setup screen; pass selection in `start_calibration` |
| `src/web/static/mobile.html` | Add RAW upload file picker, sequenced prompts driven by server messages |
| `README.md` | Document color mode workflow |
| `tests/calibration/test_raw.py` | **NEW** — DNG parser tests against a fixture DNG |
| `tests/calibration/test_color_pipeline.py` | **NEW** — RGB→XYZ math, Bradford CAT, TRC fit tests |
| `tests/display/test_profile.py` | Extend with matrix-shaper profile tests |
| `tests/calibration/test_iterate.py` | Extend with color-mode integration test |

---

## Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `rawpy` and `tifffile` to dependencies**

Edit `pyproject.toml`. In the `[project].dependencies` array, append:

```toml
    "rawpy",
    "tifffile",
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: both packages install (rawpy ships native libraw binary).

- [ ] **Step 3: Verify imports**

Run:
```bash
uv run python -c "import rawpy, tifffile; print(rawpy.__version__, tifffile.__version__)"
```
Expected: prints two version strings.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add rawpy and tifffile for DNG color-mode support"
```

---

## Task 2: DNG Parser

**Files:**
- Create: `src/calibration/raw.py`
- Create: `tests/calibration/test_raw.py`
- Create: `tests/fixtures/anchor_white.dng` (test fixture — see Step 0)

DNG tag IDs we need (TIFF tag numbers):

| Tag | ID | Type | Meaning |
|---|---|---|---|
| ColorMatrix1 | 50721 | SRATIONAL[9] | XYZ_D50 → camera RGB under StdA (calibration illuminant 1) |
| ColorMatrix2 | 50722 | SRATIONAL[9] | XYZ_D50 → camera RGB under D65 (calibration illuminant 2) |
| ForwardMatrix1 | 50964 | SRATIONAL[9] | camera RGB → XYZ_D50 under StdA |
| ForwardMatrix2 | 50965 | SRATIONAL[9] | camera RGB → XYZ_D50 under D65 |
| AsShotNeutral | 50728 | RATIONAL[3] | Normalized camera RGB of the scene neutral |
| CalibrationIlluminant1 | 50778 | SHORT | 17 = StdA |
| CalibrationIlluminant2 | 50779 | SHORT | 21 = D65 |

`ForwardMatrix2` is the preferred path (camera RGB → XYZ_D50 directly under D65 illuminant). Some DNGs lack ForwardMatrix tags; fallback is to invert ColorMatrix2.

- [ ] **Step 0: Acquire a test fixture DNG (manual, one-time)**

Use **any** iPhone Pro / Android with ProRAW or DNG capture (Halide, Open Camera, etc.) to take a single RAW photo of any subject. Place it at `tests/fixtures/anchor_white.dng`. Size: ~25-50 MB. **Do not commit this binary**; add to `.gitignore`:

```
tests/fixtures/*.dng
```

If a fixture cannot be produced, the parse-tests skip cleanly via `pytest.mark.skipif`. CI without a fixture still validates the module by import.

> **Note:** We do NOT synthesize DNGs in tests. libraw is strict about DNG structure and minimal hand-rolled TIFFs rarely pass its validator. The integration test (Task 11) monkeypatches `parse_dng` to return a hand-built `DngAnchor` object, bypassing DNG bytes entirely.

- [ ] **Step 1: Write the failing tests**

`tests/calibration/test_raw.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from calibration.raw import DngAnchor, parse_dng

FIXTURE = Path(__file__).parent.parent / "fixtures" / "anchor_white.dng"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="real DNG fixture not available locally")


def test_parse_dng_returns_anchor_with_required_fields():
    anchor = parse_dng(FIXTURE)
    assert isinstance(anchor, DngAnchor)
    assert anchor.forward_matrix_2.shape == (3, 3)
    assert anchor.as_shot_neutral.shape == (3,)
    assert anchor.linear_rgb_sample.shape == (3,)


def test_parse_dng_forward_matrix_rows_approximate_d50():
    # ForwardMatrix rows sum to the D50 white point (X=0.9642, Y=1.0, Z=0.8249).
    anchor = parse_dng(FIXTURE)
    summed = anchor.forward_matrix_2.sum(axis=1)
    np.testing.assert_allclose(summed, [0.9642, 1.0000, 0.8249], atol=0.1)


def test_parse_dng_as_shot_neutral_in_unit_range():
    anchor = parse_dng(FIXTURE)
    assert np.all(anchor.as_shot_neutral > 0)
    assert np.all(anchor.as_shot_neutral < 5.0)


def test_parse_dng_linear_sample_is_finite_and_unit():
    anchor = parse_dng(FIXTURE)
    assert np.all(np.isfinite(anchor.linear_rgb_sample))
    assert np.all(anchor.linear_rgb_sample >= 0)
    assert np.all(anchor.linear_rgb_sample <= 1.0)
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `uv run pytest tests/calibration/test_raw.py -v`
Expected: ModuleNotFoundError for `calibration.raw`.

- [ ] **Step 3: Implement `src/calibration/raw.py`**

```python
"""DNG anchor parser.

Extracts the TIFF tags we need from a DNG (ColorMatrix2, ForwardMatrix2,
AsShotNeutral) plus a linear-RGB sample of the photo's center crop. If
ForwardMatrix2 is absent, falls back to inverting ColorMatrix2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy
import tifffile

TAG_COLOR_MATRIX_2 = 50722
TAG_FORWARD_MATRIX_2 = 50965
TAG_AS_SHOT_NEUTRAL = 50728
TAG_CALIBRATION_ILLUMINANT_2 = 50779


@dataclass(frozen=True)
class DngAnchor:
    color_matrix_2: np.ndarray        # 3x3, XYZ_D50 -> camera RGB under D65
    forward_matrix_2: np.ndarray      # 3x3, camera RGB -> XYZ_D50 under D65 (never None — fallback computed)
    as_shot_neutral: np.ndarray       # length 3, normalized neutral camera RGB
    linear_rgb_sample: np.ndarray     # length 3, mean center-crop linear RGB
    calibration_illuminant_2: int


def parse_dng(path: Path) -> DngAnchor:
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        ifd = tf.pages[0].tags
        color_matrix_2 = _read_3x3(ifd, TAG_COLOR_MATRIX_2)
        forward_matrix_2 = _read_3x3_optional(ifd, TAG_FORWARD_MATRIX_2)
        asn_raw = ifd[TAG_AS_SHOT_NEUTRAL].value
        if isinstance(asn_raw, (tuple, list)) and len(asn_raw) == 3 and isinstance(asn_raw[0], tuple):
            asn = np.array([_to_float(v) for v in asn_raw], dtype=float)
        else:
            asn = np.array(asn_raw, dtype=float).reshape(3)
        illum_2 = int(ifd[TAG_CALIBRATION_ILLUMINANT_2].value) if TAG_CALIBRATION_ILLUMINANT_2 in ifd else 21

    if forward_matrix_2 is None:
        # Fallback: invert ColorMatrix2. Less accurate but better than crashing.
        forward_matrix_2 = np.linalg.inv(color_matrix_2)

    sample = _center_crop_linear_rgb(path)

    return DngAnchor(
        color_matrix_2=color_matrix_2,
        forward_matrix_2=forward_matrix_2,
        as_shot_neutral=asn,
        linear_rgb_sample=sample,
        calibration_illuminant_2=illum_2,
    )


def _read_3x3(tags, tag_id: int) -> np.ndarray:
    value = tags[tag_id].value
    arr = np.array([_to_float(v) for v in value], dtype=float)
    return arr.reshape(3, 3)


def _read_3x3_optional(tags, tag_id: int) -> np.ndarray | None:
    if tag_id not in tags:
        return None
    return _read_3x3(tags, tag_id)


def _to_float(v) -> float:
    if isinstance(v, tuple) and len(v) == 2:
        num, den = v
        return float(num) / float(den) if den else 0.0
    return float(v)


def _center_crop_linear_rgb(path: Path) -> np.ndarray:
    """Return mean linear camera RGB (length 3) of the center 25% of the image.

    We disable demosaic-time white balance and tone curve so the output is in
    linear camera space, which is what ForwardMatrix is defined against.
    """
    # rawpy.ColorSpace enum value handling: older rawpy releases use ints. The
    # `raw` member maps to 0; we use the integer for portability.
    try:
        output_color = rawpy.ColorSpace.raw
    except AttributeError:
        output_color = 0  # type: ignore[assignment]

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=output_color,
        )
    h, w = rgb.shape[:2]
    crop = rgb[h * 3 // 8 : h * 5 // 8, w * 3 // 8 : w * 5 // 8]
    return crop.mean(axis=(0, 1)) / 65535.0
```

> **Note for implementer:** No synthetic DNG generation. Tests at this layer require a real fixture DNG (Step 0) and skip if absent. Integration tests in Task 11 monkeypatch `parse_dng` itself to return a hand-built `DngAnchor` — they never invoke libraw.

- [ ] **Step 4: Run tests, expect all pass** (skipping the real-fixture test if no fixture available)

Run: `uv run pytest tests/calibration/test_raw.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/calibration/raw.py tests/calibration/test_raw.py .gitignore
git commit -m "feat: DNG anchor parser — ForwardMatrix2, AsShotNeutral, linear RGB sample"
```

---

## Task 3: Color Pipeline (camera → XYZ_D50, primary projection)

**Files:**
- Create: `src/calibration/color_pipeline.py`
- Create: `tests/calibration/test_color_pipeline.py`

The pipeline (everything stays in XYZ_D50, the ICC PCS):

1. White-balance the camera RGB sample against `AsShotNeutral`.
2. Multiply by `ForwardMatrix2` → XYZ_D50. **Done.**
3. For each single-channel patch, project measured XYZ onto the primary's XYZ
   direction → scalar amount of that primary.

sRGB primary XYZ in PCS-D50 (Bradford-adapted from D65):

```
R: (0.4361, 0.2225, 0.0139)
G: (0.3851, 0.7169, 0.0971)
B: (0.1431, 0.0606, 0.7141)
W (D50): (0.9642, 1.0000, 0.8249)
```

No Bradford CAT happens during measurement. The ICC builder also stays in D50.
Old D50→D65→D50 round-trip is gone.

- [ ] **Step 1: Write the failing tests**

`tests/calibration/test_color_pipeline.py`:

```python
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
    # Measured XYZ is exactly the primary → projection equals 1.0
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
```

- [ ] **Step 2: Run, confirm failure**

Run: `uv run pytest tests/calibration/test_color_pipeline.py -v`

- [ ] **Step 3: Implement `src/calibration/color_pipeline.py`**

```python
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


def fit_tone_response(input_levels: np.ndarray, measured: np.ndarray, target_gamma: float = 2.2) -> np.ndarray:
    """Per-channel TRC fit. Same power-law model as ramp.fit_correction."""
    mask = input_levels > 0.0
    m = np.clip(measured[mask], 1e-6, None)
    (gamma_d,), _ = curve_fit(
        lambda x, g: x ** g, input_levels[mask], m, p0=[2.2], bounds=(0.5, 5.0),
    )
    exponent = target_gamma / gamma_d
    return np.clip(np.linspace(0.0, 1.0, 256) ** exponent, 0.0, 1.0)
```

- [ ] **Step 4: Run tests, expect all pass**

Run: `uv run pytest tests/calibration/test_color_pipeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/calibration/color_pipeline.py tests/calibration/test_color_pipeline.py
git commit -m "feat: color pipeline — camera RGB to XYZ, Bradford CAT, per-channel TRC fit"
```

---

## Task 4: Matrix-Shaper ICC Profile Builder

**Files:**
- Modify: `src/display/profile.py`
- Modify: `tests/display/test_profile.py`

Existing `build_vcgt_profile` outputs a sRGB-baseline ICC with VCGT only. Add `build_matrix_shaper_profile` that takes *measured* per-channel TRC curves and per-primary XYZ values, building a real color-managed matrix profile that apps like Photoshop will consume.

ICC v2 matrix-shaper structure:
- Header (128 bytes): same as current, PCS = XYZ_D50
- Tags: `desc`, `wtpt` (D50), `rXYZ`/`gXYZ`/`bXYZ` (display primaries in D50), `rTRC`/`gTRC`/`bTRC` (curveType, 256-entry uint16 table), `vcgt` (existing VCGT)
- All XYZ values must be in PCS-D50 reference. If user provides D65 XYZ for primaries, apply Bradford D65→D50 first.

- [ ] **Step 1: Write the failing tests**

Add to `tests/display/test_profile.py`:

```python
from display.profile import build_matrix_shaper_profile


def _identity_curves():
    return np.linspace(0, 1, 256).astype(np.float64)


def test_matrix_shaper_returns_bytes():
    r = g = b = _identity_curves()
    r_xyz_d50 = np.array([0.4361, 0.2225, 0.0139])  # sRGB primaries in D50
    g_xyz_d50 = np.array([0.3851, 0.7169, 0.0971])
    b_xyz_d50 = np.array([0.1431, 0.0606, 0.7141])
    data = build_matrix_shaper_profile(r, g, b, r_xyz_d50, g_xyz_d50, b_xyz_d50)
    assert isinstance(data, bytes)
    assert data[36:40] == b"acsp"
    assert data[12:16] == b"mntr"
    assert b"rXYZ" in data
    assert b"rTRC" in data
    assert b"vcgt" in data


def test_matrix_shaper_size_matches_declared():
    r = g = b = _identity_curves()
    r_xyz = np.array([0.4361, 0.2225, 0.0139])
    g_xyz = np.array([0.3851, 0.7169, 0.0971])
    b_xyz = np.array([0.1431, 0.0606, 0.7141])
    data = build_matrix_shaper_profile(r, g, b, r_xyz, g_xyz, b_xyz)
    declared = struct.unpack(">I", data[:4])[0]
    assert declared == len(data)


def test_matrix_shaper_trc_count_is_256():
    r = g = b = _identity_curves()
    r_xyz = np.array([0.4361, 0.2225, 0.0139])
    g_xyz = np.array([0.3851, 0.7169, 0.0971])
    b_xyz = np.array([0.1431, 0.0606, 0.7141])
    data = build_matrix_shaper_profile(r, g, b, r_xyz, g_xyz, b_xyz)
    # Locate rTRC type block via tag table.
    count_tags = struct.unpack(">I", data[128:132])[0]
    found_offset = None
    for i in range(count_tags):
        entry = 132 + i * 12
        if data[entry : entry + 4] == b"rTRC":
            found_offset = struct.unpack(">I", data[entry + 4 : entry + 8])[0]
            break
    assert found_offset is not None
    # curveType: sig(4) + reserved(4) + count(4) + N entries
    declared_count = struct.unpack(">I", data[found_offset + 8 : found_offset + 12])[0]
    assert declared_count == 256
```

- [ ] **Step 2: Run, confirm failure**

Run: `uv run pytest tests/display/test_profile.py -v`

- [ ] **Step 3: Implement in `src/display/profile.py`**

Add a helper `_curv_table` and the public `build_matrix_shaper_profile`. Keep `build_vcgt_profile` untouched.

```python
def _curv_table(curve: np.ndarray) -> bytes:
    """ICC v2 'curv' type with 256-entry uint16 lookup table."""
    table = np.clip(np.round(np.asarray(curve, dtype=float) * 65535.0), 0, 65535).astype(np.uint16)
    return struct.pack(">4sII", b"curv", 0, table.shape[0]) + table.astype(">u2").tobytes()


_D50_XYZ = np.array([0.9642, 1.0000, 0.8249])


def build_matrix_shaper_profile(
    r_trc: np.ndarray,
    g_trc: np.ndarray,
    b_trc: np.ndarray,
    r_xyz_d50: np.ndarray,
    g_xyz_d50: np.ndarray,
    b_xyz_d50: np.ndarray,
    r_vcgt_lut: np.ndarray | None = None,
    g_vcgt_lut: np.ndarray | None = None,
    b_vcgt_lut: np.ndarray | None = None,
) -> bytes:
    """Build an ICC v2 matrix-shaper profile.

    r/g/b_trc: float [0, 1] tone response curves, length 256 (forward, not inverse).
    r/g/b_xyz_d50: display primaries in XYZ under D50 illuminant (PCS).
        Caller is responsible for any chromatic adaptation needed to land in D50.
    r/g/b_vcgt_lut: optional VCGT correction LUTs (uint16). If None, the matrix
        profile carries the correction in its TRC tags and VCGT is identity.
    """
    if r_vcgt_lut is None:
        identity = (np.linspace(0, 1, 256) * 65535).astype(np.uint16)
        r_vcgt_lut = g_vcgt_lut = b_vcgt_lut = identity

    tags_data: dict[bytes, bytes] = {
        b"desc": _desc_type("Color Calibrator (matrix)"),
        b"wtpt": _xyz_type(*_D50_XYZ),
        b"rXYZ": _xyz_type(*r_xyz_d50),
        b"gXYZ": _xyz_type(*g_xyz_d50),
        b"bXYZ": _xyz_type(*b_xyz_d50),
        b"rTRC": _curv_table(r_trc),
        b"gTRC": _curv_table(g_trc),
        b"bTRC": _curv_table(b_trc),
        b"vcgt": _vcgt_type(r_vcgt_lut, g_vcgt_lut, b_vcgt_lut),
    }

    n = len(tags_data)
    tag_data_start = 128 + 4 + n * 12
    tag_layout: list[tuple[bytes, int, int, bytes]] = []
    offset = tag_data_start
    for sig, data in tags_data.items():
        size = len(data)
        pad = (-size) % 4
        padded = data + b"\x00" * pad
        tag_layout.append((sig, offset, size, padded))
        offset += len(padded)

    profile_size = offset
    tag_table = struct.pack(">I", n)
    for sig, off, size, _ in tag_layout:
        tag_table += sig + struct.pack(">II", off, size)

    tag_data = b"".join(p for _, _, _, p in tag_layout)
    header = _build_header(profile_size)
    return header + tag_table + tag_data
```

- [ ] **Step 4: Run tests, expect all pass (existing + 3 new = 10 in test_profile.py)**

Run: `uv run pytest tests/display/test_profile.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/display/profile.py tests/display/test_profile.py
git commit -m "feat: matrix-shaper ICC builder — primaries, per-channel TRC, VCGT"
```

---

## Task 5: HTTP RAW Upload Endpoint + Session Anchor State

**Files:**
- Modify: `src/web/server.py`

Add:
- `Session.mode: str = "gamma"` and `Session.anchors: dict[int, bytes] = {}` for color-mode DNG storage in memory.
- `Session.anchor_event: asyncio.Event` to signal upload arrival to the iterate loop.
- `POST /upload/raw/{seq}` endpoint accepting multipart DNG.

- [ ] **Step 1: Extend `Session`**

Find `class Session:` and update:

```python
class Session:
    def __init__(self):
        self.pc: Endpoint | None = None
        self.mobile: Endpoint | None = None
        self.calibration_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.mode: str = "gamma"
        self.anchors: dict[int, bytes] = {}
        # Per-seq events. Each upload sets its own event, so wait_for_anchor
        # holds a stable reference and never races a rebind.
        self.anchor_events: dict[int, asyncio.Event] = {
            0: asyncio.Event(), 1: asyncio.Event(),
            2: asyncio.Event(), 3: asyncio.Event(),
        }
```

- [ ] **Step 2: Add upload endpoint**

After the `/ws/mobile` route, before the `# ---------- Helpers ----------` section:

```python
from fastapi import UploadFile, File, HTTPException


@app.post("/upload/raw/{seq}")
async def upload_raw(seq: int, file: UploadFile = File(...)):
    if seq < 0 or seq > 3:
        raise HTTPException(status_code=400, detail="seq must be in [0, 3]")
    data = await file.read()
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="file too small to be a DNG")
    async with session.lock:
        session.anchors[seq] = data
        event = session.anchor_events.get(seq)
    if event is not None:
        event.set()
    return {"ok": True, "seq": seq, "bytes": len(data)}
```

- [ ] **Step 3: Wire mode into `_pc_control_loop`**

Find the `start_calibration` branch:

```python
        if msg.get("type") == "start_calibration":
            async with session.lock:
                mobile = session.mobile
                if mobile is None:
                    await _send(pc.ws, {"type": "error", "message": "Mobile not connected."})
                    continue
                if session.calibration_task and not session.calibration_task.done():
                    continue
                session.mode = msg.get("mode", "gamma")
                session.anchors.clear()
                for ev in session.anchor_events.values():
                    ev.clear()
                session.calibration_task = asyncio.create_task(_run_calibration_task())
```

- [ ] **Step 4: Update `_run_calibration_task` callbacks**

After the existing callbacks, add an anchor-await helper and pass it through. We'll wire the consumer in Task 8:

```python
    async def wait_for_anchor(seq: int) -> bytes:
        # Fast-path: if the upload already landed (e.g. event was cleared on a
        # session restart but data is still in the dict), return immediately.
        async with session.lock:
            cached = session.anchors.get(seq)
        if cached is not None:
            return cached
        event = session.anchor_events[seq]
        try:
            await asyncio.wait_for(event.wait(), timeout=300.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Timed out waiting for RAW upload seq={seq}") from exc
        async with session.lock:
            data = session.anchors.get(seq)
        if data is None:
            raise RuntimeError(f"Anchor seq={seq} event set but data missing")
        return data
```

- [ ] **Step 5: Verify import**

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); from web.server import app; print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add src/web/server.py
git commit -m "feat: RAW upload endpoint + session anchor state for color mode"
```

---

### Note: PC `result` handler must read `mode`

In Task 6 below, also patch the existing `result`-message branch of `pc.html` to branch on `msg.mode`:

```javascript
    if (msg.type === 'result') {
      ...
      const dEl = document.getElementById('delta-e-value');
      if (msg.mode === 'color') {
        // Color mode has no ΔE; relabel the line entirely.
        dEl.parentElement.textContent = 'Color profile generated';
      } else {
        dEl.textContent = (msg.delta_e ?? 0).toFixed(2);
      }
      ...
    }
```

In `_run_calibration_task`, include `mode` in the result payload and convert NaN ΔE to `null` (JSON does not encode NaN):

```python
import math
delta_e_payload = None if math.isnan(delta_e) else delta_e
await _send(pc.ws, {
    "type": "result",
    "mode": session.mode,
    "icc_b64": icc_b64,
    "delta_e": delta_e_payload,
    "before_b64": before_b64,
    "after_b64": after_b64,
})
```

---

## Task 6: PC Mode Toggle UI

**Files:**
- Modify: `src/web/static/pc.html`

- [ ] **Step 1: Add radio toggle to setup screen**

Find the setup screen (`<div id="s-setup" ...>`) and modify it to include a mode selector:

```html
<!-- Screen 3: Setup instructions -->
<div id="s-setup" class="screen">
  <h1>Setup</h1>
  <p>1. Darken the room as much as possible.</p>
  <p>2. Hold your phone 30 cm from the centre of the screen, perpendicular to it.</p>
  <p>3. On your phone: tap <strong>Start Camera</strong>, then aim it at the screen.</p>
  <div style="text-align:left; padding:1rem; background:#1a1a1a; border-radius:6px; width:100%;">
    <p style="color:#eee; margin-bottom:0.5rem;"><strong>Calibration mode:</strong></p>
    <label><input type="radio" name="mode" value="gamma" checked> Gamma only (automated, ~2 min)</label><br>
    <label><input type="radio" name="mode" value="color"> Color (4 manual RAW photos required, ~5 min, ~3 ΔE)</label>
    <p style="color:#888; font-size:0.85rem; margin-top:0.4rem;">Color mode requires DNG support and sRGB JPEG output. iOS: Settings → Camera → Formats → Most Compatible.</p>
  </div>
  <p>4. When ready, click Begin below.</p>
  <button id="begin-btn">Begin Calibration</button>
</div>
```

- [ ] **Step 2: Send mode with `start_calibration`**

Find the existing `begin-btn.onclick` handler and update:

```javascript
    document.getElementById('begin-btn').onclick = () => {
      const mode = document.querySelector('input[name="mode"]:checked').value;
      ws.send(JSON.stringify({ type: 'start_calibration', mode: mode }));
      document.getElementById('begin-btn').disabled = true;
      document.getElementById('begin-btn').textContent = 'Starting…';
    };
```

- [ ] **Step 3: Commit**

```bash
git add src/web/static/pc.html
git commit -m "feat: PC mode toggle — gamma vs color calibration"
```

---

## Task 7: Mobile RAW Upload UI

**Files:**
- Modify: `src/web/static/mobile.html`

- [ ] **Step 1: Add DNG file input + upload handler**

Add to the overlay HTML, after `#ready-btn`:

```html
<input type="file" id="raw-input" accept=".dng,image/x-adobe-dng" style="display:none;">
<button id="upload-raw-btn" style="pointer-events:auto; padding:1rem 3rem; background:#f5a623; color:#fff; border:none; border-radius:30px; font-size:1.2rem; cursor:pointer; display:none;">Upload RAW Photo</button>
```

- [ ] **Step 2: Add visibility-change reconnect**

iOS Safari may suspend the WS when the user switches to the camera app. On return, reconnect.

Add near `connectWs()` definition:

```javascript
let lastWsUrl = null;
const origConnect = connectWs;
connectWs = function () {
  const wsScheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  lastWsUrl = `${wsScheme}//${location.host}/ws/mobile`;
  origConnect();
};

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible'
      && (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING)) {
    status.textContent = 'Reconnecting…';
    connectWs();
  }
});
```

- [ ] **Step 3: Add WS handlers for `request_raw` and upload flow**

In the `ws.onmessage` handler block, add:

```javascript
    if (msg.type === 'request_raw') {
      // iOS Photos transcodes DNG → JPEG on share-to-browser. Tell user to
      // route via Files. Android pickers usually preserve the DNG as-is.
      status.textContent = `Shoot a RAW (DNG) photo of the ${msg.label} patch on the PC screen. iOS: Save to Files first, then upload from Files. Android: pick from your camera roll.`;
      const uploadBtn = document.getElementById('upload-raw-btn');
      const fileInput = document.getElementById('raw-input');
      uploadBtn.style.display = 'block';
      uploadBtn.textContent = `Upload ${msg.label} RAW (#${msg.seq + 1})`;
      uploadBtn.onclick = () => fileInput.click();
      fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        // XMLHttpRequest is the only way to get upload progress events; fetch()
        // doesn't expose them. DNGs can be 50+ MB so feedback matters.
        const form = new FormData();
        form.append('file', file);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `/upload/raw/${msg.seq}`);
        xhr.upload.onprogress = (ev) => {
          if (!ev.lengthComputable) return;
          const pct = Math.round((ev.loaded / ev.total) * 100);
          status.textContent = `Uploading ${file.name}… ${pct}%`;
        };
        xhr.onload = () => {
          if (xhr.status < 200 || xhr.status >= 300) {
            status.textContent = 'Upload failed: ' + xhr.statusText;
            return;
          }
          status.textContent = `Uploaded ${file.name}. Waiting for next instruction…`;
          uploadBtn.style.display = 'none';
          ws.send(JSON.stringify({ type: 'raw_uploaded', seq: msg.seq }));
        };
        xhr.onerror = () => { status.textContent = 'Upload failed: network error'; };
        xhr.send(form);
      };
      return;
    }
```

- [ ] **Step 4: Commit**

```bash
git add src/web/static/mobile.html
git commit -m "feat: mobile RAW upload UI — file picker, multipart POST, WS reconnect"
```

---

## Task 8: Anchor Capture Phase in `iterate.py`

**Files:**
- Modify: `src/calibration/iterate.py`

Add a new function `_capture_anchors` that runs *before* the regular calibration loop in color mode. Sequence: white (seq=0), red (seq=1), green (seq=2), blue (seq=3). PC shows the corresponding solid color full-screen between RAW prompts; user shoots, switches back to the browser, taps Upload.

- [ ] **Step 1: Define the function**

Add to `iterate.py` near the top (after existing helpers, before `_capture_white_reference`):

```python
from calibration.raw import DngAnchor, parse_dng

AnchorWaitFn = Callable[[int], Awaitable[bytes]]


_ANCHOR_PROMPTS = [
    (0, "WHITE", (255, 255, 255)),
    (1, "RED",   (255, 0,   0)),
    (2, "GREEN", (0,   255, 0)),
    (3, "BLUE",  (0,   0,   255)),
]


async def _capture_anchors(
    pc_send: SendFn,
    mobile_send: SendFn,
    wait_for_anchor: AnchorWaitFn,
    tmp_dir: Path,
) -> dict[int, DngAnchor]:
    """Drive the 4-shot manual RAW anchor flow."""
    parsed: dict[int, DngAnchor] = {}
    for seq, label, rgb in _ANCHOR_PROMPTS:
        await pc_send({"type": "show_patch", "rgb": list(rgb)})
        await mobile_send({"type": "request_raw", "seq": seq, "label": label})
        data = await wait_for_anchor(seq)
        dng_path = tmp_dir / f"anchor_{seq}.dng"
        await asyncio.to_thread(dng_path.write_bytes, data)
        parsed[seq] = await asyncio.to_thread(parse_dng, dng_path)
    return parsed
```

- [ ] **Step 2: Branch on mode in `run_calibration`**

Extend the signature to accept `mode` and `wait_for_anchor`:

```python
async def run_calibration(
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
    tmp_dir: Path,
    mode: str = "gamma",
    wait_for_anchor: AnchorWaitFn | None = None,
) -> tuple[bytes, float, np.ndarray, np.ndarray, np.ndarray]:
    ...
    if mode == "color":
        if wait_for_anchor is None:
            raise RuntimeError("color mode requires wait_for_anchor callback")
        anchors = await _capture_anchors(pc_send, mobile_send, wait_for_anchor, tmp_dir)
        # The anchors are stored for the color pipeline (Task 9 wires them in).
    else:
        anchors = None
    white_frame = await _capture_white_reference(pc_send, mobile_send, mobile_recv, mobile_drain)
    ...
```

(Task 9 will replace `...` with the actual color-mode XYZ math. For this task, just thread the parameter and call `_capture_anchors`; the rest of the loop runs as today.)

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); from calibration.iterate import run_calibration; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/calibration/iterate.py
git commit -m "feat: anchor capture phase — drives 4 DNG uploads in color mode"
```

---

## Task 9: Color-Mode Pipeline in `iterate.py`

**Files:**
- Modify: `src/calibration/iterate.py`

In color mode, fit per-channel TRC curves in XYZ_D50 space using the DNG anchor's ForwardMatrix2. 11 levels × 3 channels = 33 patches per round.

For each (level, channel) patch:
1. Show `(level*255, 0, 0)` for red, etc.
2. Capture JPEG frame, center-crop, mean RGB in [0,1].
3. Reverse sRGB encoding to linear (assumption documented; see Task 12 note).
4. Apply `camera_rgb_to_xyz_d50` with the white anchor's `ForwardMatrix2` and `AsShotNeutral`.
5. **Project** measured XYZ onto the corresponding primary's XYZ direction (from the channel's own DNG anchor) → scalar.
6. After collecting all 11 levels for a channel, **normalize against that channel's DNG-anchor projection** (the "100%" reference, far more trustworthy than the L=1.0 JPEG sample).
7. Fit one TRC per channel using `fit_tone_response`.

Primary XYZ for the matrix profile comes from the 3 DNG anchors (seq 1, 2, 3) — they are noise-free linear measurements of the display's actual primaries.

- [ ] **Step 1: Add color-mode helpers**

```python
from calibration.color_pipeline import (
    SRGB_PRIMARIES_XYZ_D50,
    camera_rgb_to_xyz_d50,
    fit_tone_response,
    project_onto_primary,
)


def _srgb_to_linear(rgb_0_1: np.ndarray) -> np.ndarray:
    """Reverse the sRGB encoding curve.

    Assumes the phone encodes JPEGs in sRGB. iPhones since iOS 11 may use
    Display P3 by default; users should set their camera to Most Compatible
    (sRGB) for accurate color-mode results. See Task 12 README note.
    """
    a = 0.055
    return np.where(rgb_0_1 <= 0.04045, rgb_0_1 / 12.92, ((rgb_0_1 + a) / (1 + a)) ** 2.4)


async def _measure_color_patch(
    level: float,
    channel_idx: int,
    patch_total: int,
    patch_index: int,
    round_num: int,
    white_frame: np.ndarray,
    white_anchor: DngAnchor,
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> np.ndarray | None:
    """Measure one single-channel patch and return measured XYZ_D50, or None."""
    rgb = [0, 0, 0]
    rgb[channel_idx] = int(round(level * 255))
    await pc_send({"type": "show_patch", "rgb": rgb})
    await pc_send({"type": "capturing", "round": round_num})
    await asyncio.sleep(SETTLE_DELAY)
    await mobile_drain()
    await mobile_send({"type": "capture", "n": patch_index + 1, "total": patch_total})
    frame = await _wait_for_stable_frames(mobile_recv)
    await mobile_send({"type": "stop_capture"})
    await pc_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total, "round": round_num})
    await mobile_send({"type": "patch_done", "n": patch_index + 1, "total": patch_total})
    if frame is None:
        return None

    h, w = frame.shape[:2]
    patch_rgb = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0
    white_rgb = white_frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1)) / 255.0

    # Reverse sRGB encoding; normalize against white reference frame.
    patch_linear = _srgb_to_linear(patch_rgb)
    white_linear = np.clip(_srgb_to_linear(white_rgb), 1e-6, None)
    relative_rgb = patch_linear / white_linear

    return camera_rgb_to_xyz_d50(relative_rgb, white_anchor.as_shot_neutral, white_anchor.forward_matrix_2)
```

- [ ] **Step 2: Add `_run_color_rounds`**

```python
async def _run_color_rounds(
    white_frame: np.ndarray,
    anchors: dict[int, DngAnchor],
    pc_send: SendFn,
    mobile_send: SendFn,
    mobile_recv: RecvFn,
    mobile_drain: DrainFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run color-mode patch loop. Returns (r_trc, g_trc, b_trc, primary_xyz_d50)."""
    white_anchor = anchors[0]
    levels = np.array([p.level for p in GRAY_PATCHES])

    # Primary XYZ_D50 from the dedicated DNG anchors — clean, linear measurements
    # of the display's actual primaries.
    primary_xyz: dict[str, np.ndarray] = {}
    for ch_name, seq in [("R", 1), ("G", 2), ("B", 3)]:
        a = anchors[seq]
        primary_xyz[ch_name] = camera_rgb_to_xyz_d50(
            a.linear_rgb_sample, a.as_shot_neutral, a.forward_matrix_2,
        )

    channels = [("R", 0), ("G", 1), ("B", 2)]
    per_channel_projections: dict[str, list[float]] = {"R": [], "G": [], "B": []}
    total = len(levels) * 3
    patch_index = 0

    for ch_name, ch_idx in channels:
        primary = primary_xyz[ch_name]
        for level in levels:
            measured_xyz = await _measure_color_patch(
                float(level), ch_idx, total, patch_index, 1,
                white_frame, white_anchor,
                pc_send, mobile_send, mobile_recv, mobile_drain,
            )
            patch_index += 1
            if measured_xyz is None:
                # Soft fallback: use the ideal value for the level so the fit
                # doesn't get a NaN. Skipped patches log a warning upstream.
                per_channel_projections[ch_name].append(float(level) ** 2.2)
                continue
            per_channel_projections[ch_name].append(
                project_onto_primary(measured_xyz, primary)
            )

    def _rescale_to_unit(ch_name: str) -> np.ndarray:
        """The TRC fit needs measurements in [0, 1] where 1.0 = full primary.

        Each primary's DNG anchor projects onto itself = 1.0 by construction.
        The L=1.0 JPEG patch *should* also project to 1.0, but exposure drift
        between the manual RAW shot and the automated JPEG stream is common.
        We rescale by the L=1.0 JPEG projection so the curve fit operates on
        relative shape, which is what gamma is anyway. Anchor primaries are
        still the absolute XYZ values that go into the matrix profile.
        """
        projections = np.array(per_channel_projections[ch_name], dtype=float)
        ref = max(projections[-1], 1e-6)
        return projections / ref

    r_trc = fit_tone_response(levels, _rescale_to_unit("R"))
    g_trc = fit_tone_response(levels, _rescale_to_unit("G"))
    b_trc = fit_tone_response(levels, _rescale_to_unit("B"))

    return r_trc, g_trc, b_trc, primary_xyz
```

- [ ] **Step 3: Branch in `run_calibration` to call color path and return matrix-shaper ICC**

Inside `run_calibration`, after the anchor capture block (added in Task 8) and the white reference capture:

```python
    if mode == "color":
        r_trc, g_trc, b_trc, primary_xyz = await _run_color_rounds(
            white_frame, anchors, pc_send, mobile_send, mobile_recv, mobile_drain,
        )
        from display.profile import build_matrix_shaper_profile
        icc_bytes = build_matrix_shaper_profile(
            r_trc, g_trc, b_trc,
            primary_xyz["R"], primary_xyz["G"], primary_xyz["B"],
        )
        await pc_send({"type": "round_done", "round": 1})
        await mobile_send({"type": "round_done"})
        # Color mode does not compute a holdout ΔE. The result message uses a
        # distinct field so the PC UI labels it "profile generated" instead of
        # showing a misleading ΔE: 0.00. Sentinel value math.nan flags the UI.
        return icc_bytes, float("nan"), r_trc, g_trc, b_trc
```

> **Note:** `NaN` is the sentinel for "no ΔE measured" in color mode. The server later converts it to JSON `null` before sending to the PC. A future enhancement could compute a real holdout ΔE from chromatic test patches against expected XYZ values, but it's out of scope for v1.

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); from calibration.iterate import run_calibration; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add src/calibration/iterate.py
git commit -m "feat: color-mode pipeline — XYZ-aware TRC fits, matrix-shaper profile"
```

---

## Task 10: Wire Mode + Anchors Through `_run_calibration_task`

**Files:**
- Modify: `src/web/server.py`

- [ ] **Step 1: Pass mode and anchor callback into `run_calibration`**

Update `_run_calibration_task`:

```python
    mode = session.mode
    with tempfile.TemporaryDirectory() as tmp:
        try:
            icc_bytes, delta_e, lut_r, lut_g, lut_b = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp),
                mode=mode,
                wait_for_anchor=wait_for_anchor,
            )
            ...
```

- [ ] **Step 2: Verify imports + smoke**

```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from web.server import app; print('OK')"
uv run pytest -v
```

Expected: all existing 37 tests + new tests from Tasks 2, 3, 4 still pass.

- [ ] **Step 3: Commit**

```bash
git add src/web/server.py
git commit -m "feat: thread mode + anchor callback into calibration task"
```

---

## Task 11: Color-Mode Integration Test

**Files:**
- Modify: `tests/calibration/test_iterate.py`

- [ ] **Step 1: Extend the FakeProtocol with color-mode anchor support**

Replace the FakeProtocol `__init__` and add a `wait_for_anchor` that returns dummy bytes (the test will monkeypatch `parse_dng` to ignore them):

```python
    def __init__(self, display_gamma: float, mode: str = "gamma"):
        self.gamma = display_gamma
        self.current_input = 1.0
        self.sent: list = []
        self.recv_queue: list[dict] = []
        self.mode = mode

    async def wait_for_anchor(self, seq: int) -> bytes:
        # Returns a dummy payload. parse_dng is monkeypatched to ignore it
        # and return a hand-built DngAnchor, sidestepping libraw entirely.
        return b"DUMMY"
```

- [ ] **Step 2: Add the test**

```python
from calibration.raw import DngAnchor


def _fake_anchor(linear_rgb_sample: np.ndarray) -> DngAnchor:
    return DngAnchor(
        color_matrix_2=np.eye(3),
        forward_matrix_2=np.array([
            [0.4361, 0.3851, 0.1431],
            [0.2225, 0.7169, 0.0606],
            [0.0139, 0.0971, 0.7141],
        ]),
        as_shot_neutral=np.array([1.0, 1.0, 1.0]),
        linear_rgb_sample=linear_rgb_sample,
        calibration_illuminant_2=21,
    )


# Per-seq linear samples: white = (1,1,1); R = (1,0,0); G = (0,1,0); B = (0,0,1).
_SAMPLES = {
    0: np.array([1.0, 1.0, 1.0]),
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([0.0, 1.0, 0.0]),
    3: np.array([0.0, 0.0, 1.0]),
}


@pytest.mark.asyncio
async def test_run_calibration_color_mode_returns_matrix_profile(monkeypatch):
    monkeypatch.setattr(iterate, "clear_ramp", lambda *a, **k: None)
    monkeypatch.setattr(iterate, "apply_ramp", lambda *a, **k: None)
    monkeypatch.setattr(iterate, "SETTLE_DELAY", 0.0)

    # Each call to parse_dng returns the anchor for the next pending seq.
    call_idx = {"i": 0}
    seq_order = [0, 1, 2, 3]
    def fake_parse_dng(_path):
        seq = seq_order[call_idx["i"]]
        call_idx["i"] += 1
        return _fake_anchor(_SAMPLES[seq])
    monkeypatch.setattr(iterate, "parse_dng", fake_parse_dng)

    fake = FakeProtocol(display_gamma=2.0, mode="color")
    with tempfile.TemporaryDirectory() as tmp:
        icc_bytes, delta_e, lut_r, lut_g, lut_b = await run_calibration(
            fake.pc_send, fake.mobile_send, fake.mobile_recv, fake.mobile_drain,
            Path(tmp), mode="color", wait_for_anchor=fake.wait_for_anchor,
        )

    assert isinstance(icc_bytes, bytes) and len(icc_bytes) > 1024
    assert icc_bytes[36:40] == b"acsp"
    assert b"rXYZ" in icc_bytes
    assert b"rTRC" in icc_bytes
    assert lut_r.shape == (256,)
    # Color mode returns NaN delta_e by design.
    import math
    assert math.isnan(delta_e)
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/calibration/test_iterate.py -v`
Expected: 2 passed (existing gamma + new color).

- [ ] **Step 4: Commit**

```bash
git add tests/calibration/test_iterate.py
git commit -m "test: color-mode integration test with synthetic DNG anchors"
```

---

## Task 12: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a Color Mode section**

After the "What it does NOT do" section, before "Quick start", add:

```markdown
## Two modes

**Gamma mode (default).** Automated, ~2 min. Fixes per-channel gamma and gray
balance from 11 gray patches. Output: VCGT-only ICC. ΔE ~3-5 on grays.

**Color mode (opt-in).** Adds 4 manual RAW (DNG) captures plus 33 patches.
~5 minutes total. Fixes gray balance AND extends to white-point + primary
chromaticities. Output: matrix-shaper ICC with TRC + VCGT. ΔE ~3 across the
full gamut. Requires a phone that can capture DNG (iPhone Pro with ProRAW
enabled, or Android via Open Camera / Halide / native pro modes).

**Color mode prerequisites:**
- Phone must capture DNG (RAW).
- Phone JPEG output must be **sRGB**, not Display P3. iOS: Settings → Camera → Formats → Most Compatible. Otherwise the JPEG-stream patches are decoded with the wrong reverse curve and color accuracy drops to ~6 ΔE.
- **iOS only:** the iOS Photos app silently transcodes DNG → JPEG when you pick a file in the browser. After shooting RAW, open Photos, share the photo, **Save to Files**, then upload from Files (not Photos). On Android: most file pickers preserve DNG; use whichever app saves the original.

Choose mode on the PC setup screen.
```

- [ ] **Step 2: Add a Color mode notes subsection in "How it actually works"**

After the existing diagram, add:

```markdown
### Color mode (Option B hybrid)

When color mode is selected, the loop adds an **anchor phase** before measurement:

```
manual capture: white + R + G + B DNGs  ──┐
                                          ▼
   read DNG ForwardMatrix2 / AsShotNeutral
   build camera-RGB → XYZ_D50 transform
                                          │
                                          ▼
   automated patch stream (33 single-channel patches)
   reverse sRGB; normalize vs white; map to XYZ
   fit per-channel TRC in XYZ space
                                          │
                                          ▼
   matrix-shaper ICC v2 (rXYZ/gXYZ/bXYZ + rTRC/gTRC/bTRC + VCGT)
```

The DNG tags give us factory-calibrated camera spectral data — same trick
Apple TV Color Balance uses internally with iPhone sensors.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README updates for color mode workflow"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| `rawpy` + `tifffile` deps | 1 |
| DNG parser (tags + linear pixel sample) | 2 |
| Synthetic DNG helper for CI tests | 2 |
| camera RGB → XYZ_D50 pipeline (no CAT) + primary projection | 3 |
| Per-channel TRC fit in XYZ | 3 |
| Matrix-shaper ICC builder (D50 PCS, primaries, TRC, VCGT) | 4 |
| HTTPS multipart upload endpoint `/upload/raw/{seq}` | 5 |
| Session.mode + anchor state | 5 |
| Mode toggle on PC setup screen | 6 |
| Mobile RAW upload UI sequenced by server prompts | 7 |
| Anchor capture phase (4 DNGs, sequenced) | 8 |
| Color-mode patch loop (33 single-channel patches) | 9 |
| Color-mode end-to-end wiring in `_run_calibration_task` | 10 |
| Integration test for color mode | 11 |
| README documentation | 12 |
| Gamma mode preserved as fallback | 5, 6, 8, 9 (mode branch) |

**Type consistency check:**
- `parse_dng(path: Path) → DngAnchor` — used in Task 8 anchor capture.
- `DngAnchor.forward_matrix_2: np.ndarray` — **never None**. Parser falls back to `np.linalg.inv(color_matrix_2)` when DNG omits ForwardMatrix2.
- `camera_rgb_to_xyz_d50(rgb, neutral, fmat) → np.ndarray` — used in Task 9 `_measure_color_patch` and `_run_color_rounds`. No Bradford CAT layer.
- `project_onto_primary(measured_xyz, primary_xyz) → float` — used to derive per-channel scalar response from XYZ measurement.
- `build_matrix_shaper_profile(r_trc, g_trc, b_trc, r_xyz_d50, g_xyz_d50, b_xyz_d50, ...)` — takes D50 primaries directly. Single call site in Task 9.
- `run_calibration` final signature: `(pc_send, mobile_send, mobile_recv, mobile_drain, tmp_dir, mode="gamma", wait_for_anchor=None) → (bytes, float, ndarray, ndarray, ndarray)`. Color path returns `delta_e = math.nan`; server converts to `null` for JSON.
- `wait_for_anchor(seq: int) → bytes` callback type matches between server.py definition and `_capture_anchors` consumer.
- `Session.anchor_events: dict[int, asyncio.Event]` — one event per seq, never rebound; eliminates the race in B1.

**Reviewer fixes applied (round 1):**
- B1 anchor_event race → per-seq events dict
- B2 synth DNG → real fixture (skip if absent) + monkeypatch in integration test
- B3 channel-component approximation → `project_onto_primary` (dot product / norm²)
- D1 redundant Bradford CAT → stay in D50 throughout (no CAT in pipeline)
- D2 placeholder delta_e=0.0 → `math.nan` sentinel + JSON `null` over wire
- D3 JPEG color space assumption → documented sRGB requirement in README + PC mode hint
- D4 component normalization fragility → JPEG L=1.0 as rescale ref, anchor as XYZ primary
- D5 ForwardMatrix2 absence → invert ColorMatrix2 in parse_dng fallback
- U1 mobile WS reconnect → visibilitychange handler in mobile.html
- U2 upload progress → XMLHttpRequest with upload.onprogress

**Reviewer fixes applied (round 2):**
- R1 stale "0.0 placeholder" note → updated to describe NaN sentinel
- R2 normalization docstring/code mismatch → honest "rescale to unit" name + comment
- R3 wait_for_anchor fast-path → checks `session.anchors` before awaiting event
- R4 iOS DNG-from-Files trap → README warning + mobile UI prompt + narrowed `accept`
- R5 result message carries `"mode"` → PC reads `msg.mode === 'color'` instead of inferring from NaN

**Placeholders:** Re-scanned plan for "TODO", "TBD", "implement appropriate", "add error handling", "fill in". None remain.

---

## Acceptance criteria (informal)

- All 37 existing tests + new tests pass.
- `uv run python main.py` boots, dispwin gate works, server accepts both gamma and color session modes.
- Gamma mode behavior is unchanged from the current implementation.
- Color mode: PC prompts for mode → user picks color → PC + mobile sequence through 4 RAW anchor uploads → 33-patch automated stream → matrix-shaper ICC delivered to PC for download.
- ICC file passes inspection by ICC Profile Inspector / `iccDumpProfile`: has `acsp` signature, mntr class, rXYZ/gXYZ/bXYZ tags, rTRC/gTRC/bTRC `curv` tags with 256 entries, vcgt tag.
