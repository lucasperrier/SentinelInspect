import pytest
from pydantic import ValidationError

from sentinelinspect.inference.contracts import (
    CLASS_NAMES,
    POSITIVE_INDEX,
    ReviewPolicy,
    ReviewReason,
)


def test_positive_index_matches_the_datamodule_label_encoding():
    """datamodule maps crack -> 1, non_crack -> 0. If CLASS_NAMES and
    POSITIVE_INDEX ever disagree with that, every prediction inverts silently."""
    assert CLASS_NAMES[POSITIVE_INDEX] == "crack"
    assert CLASS_NAMES[0] == "no_crack"


@pytest.mark.parametrize(
    "probability,expected",
    [(0.05, False), (0.34, False), (0.35, True), (0.50, True), (0.65, True), (0.66, False), (0.99, False)],
)
def test_review_band_is_inclusive_at_both_edges(probability, expected):
    flagged, reason = ReviewPolicy(lower=0.35, upper=0.65).evaluate(probability)
    assert flagged is expected
    assert (reason == ReviewReason.LOW_CONFIDENCE) is expected


def test_band_is_two_sided():
    """0.48 and 0.52 are equally uncertain. A one-sided floor would flag only
    one of them, which is the mistake this policy exists to avoid."""
    policy = ReviewPolicy(lower=0.4, upper=0.6)
    assert policy.evaluate(0.48)[0] is True
    assert policy.evaluate(0.52)[0] is True


def test_inverted_band_is_rejected():
    with pytest.raises((ValidationError, ValueError), match="lower"):
        ReviewPolicy(lower=0.7, upper=0.3)


def test_probabilities_outside_zero_one_are_rejected():
    with pytest.raises(ValidationError):
        ReviewPolicy(lower=-0.1, upper=0.5)


def test_zero_width_band_reviews_nothing_but_the_exact_boundary():
    policy = ReviewPolicy(lower=0.5, upper=0.5)
    assert policy.width == 0.0
    assert policy.evaluate(0.5)[0] is True
    assert policy.evaluate(0.5001)[0] is False
