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
            asn_flat = np.array(asn_raw, dtype=float)
            if asn_flat.size == 6:
                # Flat (num, den) interleaved → fold into 3 rationals.
                nums = asn_flat[0::2]
                dens = asn_flat[1::2]
                asn = np.where(dens == 0, 0.0, nums / np.where(dens == 0, 1.0, dens))
            else:
                asn = asn_flat.reshape(3)
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
    # tifffile sometimes returns 9 (num, den) tuples → 9 floats.
    # Other versions return 18 flat ints (num, den interleaved) → 18 floats.
    # Fold pairs back into rationals when we got the flat form.
    if arr.size == 18:
        nums = arr[0::2]
        dens = arr[1::2]
        arr = np.where(dens == 0, 0.0, nums / np.where(dens == 0, 1.0, dens))
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
