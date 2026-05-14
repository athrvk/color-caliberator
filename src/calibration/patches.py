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
