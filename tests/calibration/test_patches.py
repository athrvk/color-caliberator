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
