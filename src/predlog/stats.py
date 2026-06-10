"""Summary statistics for resolved Predlog predictions.

This module turns prediction models into calibration and scoring summaries. It
does not read SQLite, print terminal output, or generate plots. Those layers can
reuse the summary dataclasses here so Predlog's terminal stats and plots agree.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import math

from predlog import config, scoring
from predlog.models import AnyPrediction, BinaryPrediction, RangePrediction


CALIBRATION_FEEDBACK_TOLERANCE = 0.05
"""Maximum calibration gap that still counts as about right."""


@dataclass(frozen=True, kw_only=True)
class BinaryCalibrationBucket:
    """Calibration summary for binary predictions near one probability bucket.

    Attributes:
        bucket: Bucket label as a percentage from ``10`` through ``90``.
        count: Number of resolved binary predictions in the bucket.
        event_rate: Fraction of predictions in the bucket that resolved yes.
        calibration_gap: Difference between the observed event rate and bucket
            probability.
        feedback: Short interpretation of the calibration gap.
    """

    bucket: int
    count: int
    event_rate: float
    calibration_gap: float
    feedback: str


@dataclass(frozen=True, kw_only=True)
class RangeCalibrationBucket:
    """Calibration summary for range predictions near one confidence bucket.

    Attributes:
        bucket: Bucket label as a percentage from ``10`` through ``90``.
        count: Number of resolved range predictions in the bucket.
        containment_rate: Fraction of actual values contained in the forecast
            intervals for this bucket.
        mean_confidence: Mean stated confidence for predictions in the bucket,
            stored as a decimal from ``0.0`` to ``1.0``.
        calibration_gap: Difference between containment rate and mean stated
            confidence.
        median_range_factor: Median multiplicative spread of prediction
            intervals in this bucket. Smaller values mean sharper intervals.
        feedback: Short interpretation of the calibration gap.
    """

    bucket: int
    count: int
    containment_rate: float
    mean_confidence: float
    calibration_gap: float
    median_range_factor: float
    feedback: str


@dataclass(frozen=True, kw_only=True)
class BinaryStats:
    """Aggregate stats for resolved binary predictions."""

    resolved_count: int
    mean_brier_score: float | None
    directional_count: int
    directional_hit_rate: float | None
    calibration_buckets: tuple[BinaryCalibrationBucket, ...]


@dataclass(frozen=True, kw_only=True)
class RangeStats:
    """Aggregate stats for resolved numerical range predictions."""

    resolved_count: int
    mean_winkler_score: float | None
    containment_rate: float | None
    median_range_factor: float | None
    confidence_buckets: tuple[RangeCalibrationBucket, ...]


@dataclass(frozen=True, kw_only=True)
class PredictionStats:
    """Combined binary and range statistics for Predlog."""

    binary: BinaryStats
    range: RangeStats


def summarize_predictions(predictions: Iterable[AnyPrediction]) -> PredictionStats:
    """Return binary and range stats for resolved predictions.

    Open predictions are ignored. This lets callers safely pass mixed lists from
    storage or tests without accidentally counting unresolved forecasts.
    """

    prediction_list = list(predictions)
    binary_predictions = [
        prediction
        for prediction in prediction_list
        if isinstance(prediction, BinaryPrediction)
    ]
    range_predictions = [
        prediction
        for prediction in prediction_list
        if isinstance(prediction, RangePrediction)
    ]
    return PredictionStats(
        binary=summarize_binary(binary_predictions),
        range=summarize_range(range_predictions),
    )


def summarize_binary(predictions: Iterable[BinaryPrediction]) -> BinaryStats:
    """Return stats for resolved binary predictions.

    The directional hit rate excludes 50 percent forecasts because they do not
    express a yes/no lean.
    """

    resolved = _resolved_binary_predictions(predictions)
    if not resolved:
        return BinaryStats(
            resolved_count=0,
            mean_brier_score=None,
            directional_count=0,
            directional_hit_rate=None,
            calibration_buckets=(),
        )

    brier_scores = [
        scoring.brier_score(prediction.probability, prediction.outcome)
        for prediction in resolved
    ]
    directional_hits = [
        _binary_direction_was_hit(prediction)
        for prediction in resolved
        if prediction.probability != 0.5
    ]

    return BinaryStats(
        resolved_count=len(resolved),
        mean_brier_score=_mean(brier_scores),
        directional_count=len(directional_hits),
        directional_hit_rate=_mean(directional_hits),
        calibration_buckets=_binary_calibration_buckets(resolved),
    )


def summarize_range(predictions: Iterable[RangePrediction]) -> RangeStats:
    """Return stats for resolved numerical range predictions."""

    resolved = _resolved_range_predictions(predictions)
    if not resolved:
        return RangeStats(
            resolved_count=0,
            mean_winkler_score=None,
            containment_rate=None,
            median_range_factor=None,
            confidence_buckets=(),
        )

    winkler_scores = [
        scoring.winkler_score(
            prediction.lower,
            prediction.upper,
            prediction.confidence,
            prediction.actual,
        )
        for prediction in resolved
    ]
    containment_results = [
        scoring.contained(prediction.lower, prediction.upper, prediction.actual)
        for prediction in resolved
    ]
    range_factors = [
        scoring.range_factor(prediction.lower, prediction.upper)
        for prediction in resolved
    ]

    return RangeStats(
        resolved_count=len(resolved),
        mean_winkler_score=_mean(winkler_scores),
        containment_rate=_mean(containment_results),
        median_range_factor=_median(range_factors),
        confidence_buckets=_range_calibration_buckets(resolved),
    )


def nearest_probability_bucket(decimal_value: float) -> int:
    """Return the configured bucket for a valid Predlog probability."""

    return _forecast_decimal_to_bucket(decimal_value)


def nearest_confidence_bucket(decimal_value: float) -> int:
    """Return the configured bucket for a valid Predlog confidence value."""

    return _forecast_decimal_to_bucket(decimal_value)


def _resolved_binary_predictions(
    predictions: Iterable[BinaryPrediction],
) -> list[BinaryPrediction]:
    """Return binary predictions that have outcomes."""

    return [
        prediction
        for prediction in predictions
        if prediction.is_resolved and prediction.outcome is not None
    ]


def _resolved_range_predictions(
    predictions: Iterable[RangePrediction],
) -> list[RangePrediction]:
    """Return range predictions that have actual values."""

    return [
        prediction
        for prediction in predictions
        if prediction.is_resolved and prediction.actual is not None
    ]


def _binary_calibration_buckets(
    predictions: Iterable[BinaryPrediction],
) -> tuple[BinaryCalibrationBucket, ...]:
    """Return non-empty binary calibration buckets."""

    grouped: dict[int, list[BinaryPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[nearest_probability_bucket(prediction.probability)].append(prediction)

    buckets = []
    for bucket in config.BINARY_CALIBRATION_BUCKETS:
        bucket_predictions = grouped[bucket]
        if not bucket_predictions:
            continue
        outcomes = [prediction.outcome for prediction in bucket_predictions]
        event_rate = _mean(outcomes)
        assert event_rate is not None
        bucket_probability = bucket / 100
        calibration_gap = event_rate - bucket_probability
        buckets.append(
            BinaryCalibrationBucket(
                bucket=bucket,
                count=len(bucket_predictions),
                event_rate=event_rate,
                calibration_gap=calibration_gap,
                feedback=_binary_calibration_feedback(
                    len(bucket_predictions),
                    calibration_gap,
                ),
            )
        )
    return tuple(buckets)


def _range_calibration_buckets(
    predictions: Iterable[RangePrediction],
) -> tuple[RangeCalibrationBucket, ...]:
    """Return non-empty range confidence calibration buckets."""

    grouped: dict[int, list[RangePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[nearest_confidence_bucket(prediction.confidence)].append(prediction)

    buckets = []
    for bucket in config.RANGE_CONFIDENCE_BUCKETS:
        bucket_predictions = grouped[bucket]
        if not bucket_predictions:
            continue
        containment_results = [
            scoring.contained(prediction.lower, prediction.upper, prediction.actual)
            for prediction in bucket_predictions
        ]
        confidences = [
            prediction.confidence for prediction in bucket_predictions
        ]
        range_factors = [
            scoring.range_factor(prediction.lower, prediction.upper)
            for prediction in bucket_predictions
        ]
        containment_rate = _mean(containment_results)
        mean_confidence = _mean(confidences)
        median_range_factor = _median(range_factors)
        assert containment_rate is not None
        assert mean_confidence is not None
        assert median_range_factor is not None
        calibration_gap = containment_rate - mean_confidence
        buckets.append(
            RangeCalibrationBucket(
                bucket=bucket,
                count=len(bucket_predictions),
                containment_rate=containment_rate,
                mean_confidence=mean_confidence,
                calibration_gap=calibration_gap,
                median_range_factor=median_range_factor,
                feedback=_range_calibration_feedback(
                    len(bucket_predictions),
                    calibration_gap,
                ),
            )
        )
    return tuple(buckets)


def _binary_direction_was_hit(prediction: BinaryPrediction) -> bool:
    """Return whether a non-50-percent binary forecast got the direction right."""

    predicted_outcome = 1 if prediction.probability > 0.5 else 0
    return prediction.outcome == predicted_outcome


def _forecast_decimal_to_bucket(decimal_value: float) -> int:
    """Return the 10-point percentage bucket for a valid forecast decimal."""

    percent = decimal_value * 100
    for bucket in config.FORECAST_PERCENTAGES:
        if math.isclose(percent, bucket, rel_tol=0.0, abs_tol=1e-9):
            return bucket
    msg = "forecast must be one of 10, 20, ..., 90 percent"
    raise ValueError(msg)


def _binary_calibration_feedback(count: int, calibration_gap: float) -> str:
    """Return plain-English feedback for a binary calibration bucket."""

    if count < config.CALIBRATION_MIN_EVIDENCE_COUNT:
        return "Not enough data"
    if abs(calibration_gap) <= CALIBRATION_FEEDBACK_TOLERANCE:
        return "About right"
    if calibration_gap < 0:
        return "Predicted too high"
    return "Predicted too low"


def _range_calibration_feedback(count: int, calibration_gap: float) -> str:
    """Return plain-English feedback for a range calibration bucket."""

    if count < config.CALIBRATION_MIN_EVIDENCE_COUNT:
        return "Not enough data"
    if abs(calibration_gap) <= CALIBRATION_FEEDBACK_TOLERANCE:
        return "About right"
    if calibration_gap < 0:
        return "Too narrow"
    return "Too wide"


def _mean(values: Iterable[float | int | bool | None]) -> float | None:
    """Return the arithmetic mean of non-None values, or None for empty input."""

    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def _median(values: Iterable[float | int | bool | None]) -> float | None:
    """Return the median of non-None values, or None for empty input."""

    clean_values = sorted(float(value) for value in values if value is not None)
    if not clean_values:
        return None

    midpoint = len(clean_values) // 2
    if len(clean_values) % 2 == 1:
        return clean_values[midpoint]
    return (clean_values[midpoint - 1] + clean_values[midpoint]) / 2
