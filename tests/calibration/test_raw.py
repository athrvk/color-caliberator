from pathlib import Path

import numpy as np
import pytest

from calibration.raw import DngAnchor, _read_3x3, parse_dng

FIXTURE = Path(__file__).parent.parent / "fixtures" / "anchor_white.dng"
needs_fixture = pytest.mark.skipif(not FIXTURE.exists(), reason="real DNG fixture not available locally")


class _FakeTag:
    def __init__(self, value):
        self.value = value


def test_read_3x3_paired_rationals():
    tags = {1: _FakeTag(((1, 2), (1, 4), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)))}
    m = _read_3x3(tags, 1)
    assert m.shape == (3, 3)
    assert m[0, 0] == pytest.approx(0.5)
    assert m[0, 1] == pytest.approx(0.25)


def test_read_3x3_flat_interleaved_rationals():
    """iPhone DNG via some tifffile versions returns 18 flat ints (num/den pairs)."""
    flat = (1, 2, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    tags = {1: _FakeTag(flat)}
    m = _read_3x3(tags, 1)
    assert m.shape == (3, 3)
    assert m[0, 0] == pytest.approx(0.5)
    assert m[0, 1] == pytest.approx(0.25)


@needs_fixture
def test_parse_dng_returns_anchor_with_required_fields():
    anchor = parse_dng(FIXTURE)
    assert isinstance(anchor, DngAnchor)
    assert anchor.forward_matrix_2.shape == (3, 3)
    assert anchor.as_shot_neutral.shape == (3,)
    assert anchor.linear_rgb_sample.shape == (3,)


@needs_fixture
def test_parse_dng_forward_matrix_rows_approximate_d50():
    # ForwardMatrix rows sum to the D50 white point (X=0.9642, Y=1.0, Z=0.8249).
    anchor = parse_dng(FIXTURE)
    summed = anchor.forward_matrix_2.sum(axis=1)
    np.testing.assert_allclose(summed, [0.9642, 1.0000, 0.8249], atol=0.1)


@needs_fixture
def test_parse_dng_as_shot_neutral_in_unit_range():
    anchor = parse_dng(FIXTURE)
    assert np.all(anchor.as_shot_neutral > 0)
    assert np.all(anchor.as_shot_neutral < 5.0)


@needs_fixture
def test_parse_dng_linear_sample_is_finite_and_unit():
    anchor = parse_dng(FIXTURE)
    assert np.all(np.isfinite(anchor.linear_rgb_sample))
    assert np.all(anchor.linear_rgb_sample >= 0)
    assert np.all(anchor.linear_rgb_sample <= 1.0)
