import math

import pytest

from predlog import scoring


@pytest.mark.parametrize(
    ("probability", "outcome", "expected"),
    [
        (0.70, 1, 0.09),
        (0.70, 0, 0.49),
        (0.50, 1, 0.25),
        (0.90, 1, 0.01),
        (0.90, 0, 0.81),
        (0.00, 0, 0.00),
        (1.00, 1, 0.00),
    ],
)
def test_brier_score(probability, outcome, expected):
    """Brier score is squared error between probability and outcome."""

    assert scoring.brier_score(probability, outcome) == pytest.approx(expected)


@pytest.mark.parametrize("probability", [-0.01, 1.01, math.inf, math.nan])
def test_brier_score_rejects_invalid_probability(probability):
    """Binary probability must be finite and between 0 and 1."""

    with pytest.raises(ValueError):
        scoring.brier_score(probability, 1)


@pytest.mark.parametrize("outcome", [-1, 2, math.inf, math.nan])
def test_brier_score_rejects_invalid_outcome(outcome):
    """Binary outcome must be finite and encoded as 0 or 1."""

    with pytest.raises(ValueError):
        scoring.brier_score(0.50, outcome)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (5.0, True),
        (9.5, True),
        (12.0, True),
        (4.9, False),
        (12.1, False),
    ],
)
def test_contained_is_inclusive(actual, expected):
    """Range containment includes values exactly on either interval bound."""

    assert scoring.contained(5.0, 12.0, actual) is expected


def test_interval_width():
    """Interval width is the distance from lower bound to upper bound."""

    assert scoring.interval_width(5.0, 12.0) == pytest.approx(7.0)


@pytest.mark.parametrize(("lower", "upper"), [(5.0, 5.0), (12.0, 5.0)])
def test_interval_functions_reject_invalid_bounds(lower, upper):
    """Intervals must have a lower bound that is strictly less than upper."""

    with pytest.raises(ValueError):
        scoring.contained(lower, upper, 6.0)

    with pytest.raises(ValueError):
        scoring.interval_width(lower, upper)

    with pytest.raises(ValueError):
        scoring.relative_interval_width(lower, upper)

    with pytest.raises(ValueError):
        scoring.winkler_score(lower, upper, 0.80, 6.0)


def test_relative_interval_width():
    """Relative interval width divides width by absolute interval midpoint."""

    assert scoring.relative_interval_width(80.0, 120.0) == pytest.approx(0.40)
    assert scoring.relative_interval_width(-120.0, -80.0) == pytest.approx(0.40)


def test_relative_interval_width_returns_none_near_zero_midpoint():
    """Relative width is unavailable when the midpoint is too close to zero."""

    assert scoring.relative_interval_width(-1.0, 1.0) is None


def test_winkler_score_inside_interval():
    """Inside the interval, Winkler score is just interval width."""

    assert scoring.winkler_score(5.0, 12.0, 0.80, 9.5) == pytest.approx(7.0)


def test_winkler_score_below_interval():
    """Below the interval, Winkler score adds the lower-bound miss penalty."""

    assert scoring.winkler_score(5.0, 12.0, 0.80, 3.0) == pytest.approx(27.0)


def test_winkler_score_above_interval():
    """Above the interval, Winkler score adds the upper-bound miss penalty."""

    assert scoring.winkler_score(5.0, 12.0, 0.80, 14.0) == pytest.approx(27.0)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, math.inf, math.nan])
def test_winkler_score_rejects_invalid_confidence(confidence):
    """Range confidence must be finite, greater than 0, and less than 1."""

    with pytest.raises(ValueError):
        scoring.winkler_score(5.0, 12.0, confidence, 9.5)


@pytest.mark.parametrize("actual", [math.inf, math.nan])
def test_range_functions_reject_non_finite_actual(actual):
    """Resolved actual values must be finite."""

    with pytest.raises(ValueError):
        scoring.contained(5.0, 12.0, actual)

    with pytest.raises(ValueError):
        scoring.winkler_score(5.0, 12.0, 0.80, actual)
