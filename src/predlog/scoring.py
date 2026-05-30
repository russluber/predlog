"""Pure scoring functions for resolved Predlog predictions.

This module contains the mathematical core of Predlog. It does not read from
SQLite, write files, parse CLI arguments, or print output. Keeping scoring pure
makes it easy to test and lets the CLI, stats, and plotting layers reuse the
same formulas consistently.
"""

from __future__ import annotations

import math

RELATIVE_WIDTH_MIN_MIDPOINT = 1e-12
"""Smallest midpoint magnitude treated as safe for relative interval width."""


def brier_score(probability: float, outcome: int) -> float:
    """Return the Brier score for a resolved binary prediction.

    Parameters:
        probability: The forecast probability as a decimal from ``0.0`` to
            ``1.0``. For example, a 70 percent forecast is stored as ``0.70``.
        outcome: The resolved outcome, where ``1`` means the event happened and
            ``0`` means it did not happen.

    Returns:
        The squared error between the forecast probability and the outcome.
        Lower scores are better, and ``0.0`` is perfect.

    Raises:
        ValueError: If ``probability`` is outside ``[0.0, 1.0]`` or ``outcome``
            is not ``0`` or ``1``.
    """

    _validate_probability(probability)
    _validate_outcome(outcome)
    return (probability - outcome) ** 2


def contained(lower: float, upper: float, actual: float) -> bool:
    """Return whether an actual value falls inside a forecast interval.

    Interval containment is inclusive. An actual value exactly equal to either
    bound counts as contained, matching the Winkler interval-score formula in
    the architecture document.

    Raises:
        ValueError: If the interval bounds are invalid.
    """

    _validate_interval(lower, upper)
    _validate_finite("actual", actual)
    return lower <= actual <= upper


def interval_width(lower: float, upper: float) -> float:
    """Return the width of a numerical forecast interval.

    The width is ``upper - lower``. Predlog v0.1 requires strict intervals, so
    equal bounds and reversed bounds are rejected.

    Raises:
        ValueError: If the interval bounds are invalid.
    """

    _validate_interval(lower, upper)
    return upper - lower


def relative_interval_width(lower: float, upper: float) -> float | None:
    """Return interval width relative to the absolute interval midpoint.

    Relative width is useful for comparing interval tightness across different
    scales. For example, a width of ``10`` around ``1000`` is much tighter than
    a width of ``10`` around ``12``.

    The formula is:

    ``(upper - lower) / abs((lower + upper) / 2)``

    Returns ``None`` when the midpoint is too close to zero, because dividing by
    a near-zero midpoint creates an unstable and misleading ratio.

    Raises:
        ValueError: If the interval bounds are invalid.
    """

    _validate_interval(lower, upper)
    midpoint = (lower + upper) / 2
    if abs(midpoint) <= RELATIVE_WIDTH_MIN_MIDPOINT:
        return None
    return interval_width(lower, upper) / abs(midpoint)


def winkler_score(lower: float, upper: float, confidence: float, actual: float) -> float:
    """Return the Winkler interval score for a resolved range prediction.

    Parameters:
        lower: Lower bound of the forecast interval.
        upper: Upper bound of the forecast interval.
        confidence: Stated interval confidence as a decimal greater than
            ``0.0`` and less than ``1.0``. For example, an 80 percent interval
            is stored as ``0.80``.
        actual: The resolved numerical value.

    Returns:
        The Winkler interval score. Lower scores are better. If ``actual`` is
        inside the interval, the score is the interval width. If ``actual`` is
        outside the interval, the score adds a miss penalty based on
        ``alpha = 1 - confidence``.

    Raises:
        ValueError: If the interval bounds are invalid, confidence is outside
            ``(0.0, 1.0)``, or ``actual`` is not finite.
    """

    width = interval_width(lower, upper)
    _validate_confidence(confidence)
    _validate_finite("actual", actual)

    alpha = 1 - confidence
    if actual < lower:
        return width + (2 / alpha) * (lower - actual)
    if actual > upper:
        return width + (2 / alpha) * (actual - upper)
    return width


def _validate_probability(probability: float) -> None:
    """Raise ValueError unless a binary probability is finite and in range."""

    _validate_finite("probability", probability)
    if not 0 <= probability <= 1:
        msg = "probability must be between 0.0 and 1.0"
        raise ValueError(msg)


def _validate_outcome(outcome: int) -> None:
    """Raise ValueError unless a binary outcome is 0 or 1."""

    _validate_finite("outcome", outcome)
    if outcome not in {0, 1}:
        msg = "outcome must be 0 or 1"
        raise ValueError(msg)


def _validate_confidence(confidence: float) -> None:
    """Raise ValueError unless range confidence is finite and strictly valid."""

    _validate_finite("confidence", confidence)
    if not 0 < confidence < 1:
        msg = "confidence must be greater than 0.0 and less than 1.0"
        raise ValueError(msg)


def _validate_interval(lower: float, upper: float) -> None:
    """Raise ValueError unless interval bounds are finite and lower < upper."""

    _validate_finite("lower", lower)
    _validate_finite("upper", upper)
    if lower >= upper:
        msg = "interval lower bound must be less than upper bound"
        raise ValueError(msg)


def _validate_finite(name: str, value: float) -> None:
    """Raise ValueError unless a numeric value is finite."""

    if not math.isfinite(value):
        msg = f"{name} must be finite"
        raise ValueError(msg)
