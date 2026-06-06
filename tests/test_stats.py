from datetime import datetime, timezone

import pytest

from predlog import stats
from predlog.models import OPEN_STATUS, RESOLVED_STATUS, BinaryPrediction, RangePrediction


CREATED_AT = datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)
RESOLVED_AT = datetime(2026, 5, 31, 9, 0, 0, tzinfo=timezone.utc)


def binary_prediction(probability, outcome=None, status=RESOLVED_STATUS):
    """Build a binary prediction for stats tests."""

    return BinaryPrediction(
        id=1,
        question="Binary?",
        created_at=CREATED_AT,
        status=status,
        resolved_at=RESOLVED_AT if status == RESOLVED_STATUS else None,
        probability=probability,
        outcome=outcome,
    )


def range_prediction(
    lower,
    upper,
    confidence,
    actual=None,
    status=RESOLVED_STATUS,
):
    """Build a range prediction for stats tests."""

    return RangePrediction(
        id=2,
        question="Range?",
        created_at=CREATED_AT,
        status=status,
        resolved_at=RESOLVED_AT if status == RESOLVED_STATUS else None,
        lower=lower,
        upper=upper,
        confidence=confidence,
        actual=actual,
    )


def test_empty_prediction_stats():
    """Empty input returns counts of zero and no fake summary values."""

    summary = stats.summarize_predictions([])

    assert summary.binary.resolved_count == 0
    assert summary.binary.mean_brier_score is None
    assert summary.binary.directional_count == 0
    assert summary.binary.directional_hit_rate is None
    assert summary.binary.calibration_buckets == ()

    assert summary.range.resolved_count == 0
    assert summary.range.mean_winkler_score is None
    assert summary.range.containment_rate is None
    assert summary.range.median_range_factor is None
    assert summary.range.confidence_buckets == ()


def test_binary_mean_brier_score():
    """Binary stats include mean Brier score for resolved predictions."""

    summary = stats.summarize_binary(
        [
            binary_prediction(0.70, 1),
            binary_prediction(0.60, 0),
        ]
    )

    assert summary.resolved_count == 2
    assert summary.mean_brier_score == pytest.approx((0.09 + 0.36) / 2)


def test_binary_directional_hit_rate_excludes_fifty_percent_predictions():
    """Directional hit rate ignores forecasts that do not lean yes or no."""

    summary = stats.summarize_binary(
        [
            binary_prediction(0.70, 1),
            binary_prediction(0.40, 0),
            binary_prediction(0.60, 0),
            binary_prediction(0.50, 1),
        ]
    )

    assert summary.directional_count == 3
    assert summary.directional_hit_rate == pytest.approx(2 / 3)


def test_binary_calibration_bucket_calculations():
    """Binary calibration buckets use nearest 10-point probability buckets."""

    summary = stats.summarize_binary(
        [
            binary_prediction(0.01, 0),
            binary_prediction(0.73, 1),
            binary_prediction(0.75, 0),
            binary_prediction(0.99, 1),
        ]
    )

    buckets = {bucket.bucket: bucket for bucket in summary.calibration_buckets}

    assert set(buckets) == {10, 70, 80, 90}
    assert buckets[10].count == 1
    assert buckets[10].event_rate == pytest.approx(0.0)
    assert buckets[10].mean_probability == pytest.approx(0.01)
    assert buckets[10].calibration_gap == pytest.approx(-0.01)
    assert buckets[10].feedback == "Not enough data"
    assert buckets[70].event_rate == pytest.approx(1.0)
    assert buckets[70].mean_probability == pytest.approx(0.73)
    assert buckets[70].calibration_gap == pytest.approx(0.27)
    assert buckets[80].event_rate == pytest.approx(0.0)
    assert buckets[80].mean_probability == pytest.approx(0.75)
    assert buckets[80].calibration_gap == pytest.approx(-0.75)
    assert buckets[90].event_rate == pytest.approx(1.0)
    assert buckets[90].mean_probability == pytest.approx(0.99)
    assert buckets[90].calibration_gap == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("decimal_value", "expected_bucket"),
    [
        (0.01, 10),
        (0.14, 10),
        (0.15, 20),
        (0.73, 70),
        (0.75, 80),
        (0.84, 80),
        (0.85, 90),
        (0.99, 90),
    ],
)
def test_nearest_probability_bucket(decimal_value, expected_bucket):
    """Probability bucket assignment matches the calibration policy."""

    assert stats.nearest_probability_bucket(decimal_value) == expected_bucket


def test_range_containment_rate_and_mean_winkler_score():
    """Range stats include containment rate and mean Winkler score."""

    summary = stats.summarize_range(
        [
            range_prediction(5.0, 12.0, 0.80, 9.5),
            range_prediction(5.0, 12.0, 0.80, 14.0),
        ]
    )

    assert summary.resolved_count == 2
    assert summary.containment_rate == pytest.approx(0.5)
    assert summary.mean_winkler_score == pytest.approx((7.0 + 27.0) / 2)


def test_range_factor_summaries():
    """Range stats summarize multiplicative interval sharpness."""

    summary = stats.summarize_range(
        [
            range_prediction(80.0, 120.0, 0.80, 100.0),
            range_prediction(90.0, 110.0, 0.80, 100.0),
            range_prediction(50.0, 100.0, 0.80, 75.0),
        ]
    )

    assert summary.median_range_factor == pytest.approx(1.5)


def test_range_confidence_bucket_calculations():
    """Range calibration buckets use nearest 10-point confidence buckets."""

    summary = stats.summarize_range(
        [
            range_prediction(1.0, 10.0, 0.73, 5.0),
            range_prediction(1.0, 10.0, 0.75, 12.0),
            range_prediction(1.0, 10.0, 0.99, 5.0),
        ]
    )

    buckets = {bucket.bucket: bucket for bucket in summary.confidence_buckets}

    assert set(buckets) == {70, 80, 90}
    assert buckets[70].count == 1
    assert buckets[70].containment_rate == pytest.approx(1.0)
    assert buckets[70].mean_confidence == pytest.approx(0.73)
    assert buckets[70].calibration_gap == pytest.approx(0.27)
    assert buckets[70].median_range_factor == pytest.approx(10.0)
    assert buckets[70].feedback == "Not enough data"
    assert buckets[80].containment_rate == pytest.approx(0.0)
    assert buckets[80].mean_confidence == pytest.approx(0.75)
    assert buckets[80].calibration_gap == pytest.approx(-0.75)
    assert buckets[80].median_range_factor == pytest.approx(10.0)
    assert buckets[90].containment_rate == pytest.approx(1.0)
    assert buckets[90].mean_confidence == pytest.approx(0.99)
    assert buckets[90].calibration_gap == pytest.approx(0.01)
    assert buckets[90].median_range_factor == pytest.approx(10.0)


def test_calibration_feedback_requires_enough_evidence():
    """Calibration feedback waits for enough bucket evidence."""

    sparse_binary = stats.summarize_binary([binary_prediction(0.80, 0)])
    enough_binary = stats.summarize_binary(
        [
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 0),
        ]
    )
    enough_range = stats.summarize_range(
        [
            range_prediction(90.0, 110.0, 0.80, 100.0),
            range_prediction(90.0, 110.0, 0.80, 101.0),
            range_prediction(90.0, 110.0, 0.80, 99.0),
            range_prediction(90.0, 110.0, 0.80, 102.0),
            range_prediction(90.0, 110.0, 0.80, 98.0),
        ]
    )

    assert sparse_binary.calibration_buckets[0].feedback == "Not enough data"
    assert enough_binary.calibration_buckets[0].feedback == "About right"
    assert enough_range.confidence_buckets[0].feedback == "Too wide"


@pytest.mark.parametrize(
    ("decimal_value", "expected_bucket"),
    [
        (0.01, 10),
        (0.14, 10),
        (0.15, 20),
        (0.73, 70),
        (0.75, 80),
        (0.84, 80),
        (0.85, 90),
        (0.99, 90),
    ],
)
def test_nearest_confidence_bucket(decimal_value, expected_bucket):
    """Confidence bucket assignment matches the calibration policy."""

    assert stats.nearest_confidence_bucket(decimal_value) == expected_bucket


def test_summarize_predictions_ignores_open_predictions():
    """Mixed inputs count only resolved predictions."""

    summary = stats.summarize_predictions(
        [
            binary_prediction(0.70, 1),
            binary_prediction(0.60, status=OPEN_STATUS),
            range_prediction(5.0, 12.0, 0.80, 9.5),
            range_prediction(5.0, 12.0, 0.80, status=OPEN_STATUS),
        ]
    )

    assert summary.binary.resolved_count == 1
    assert summary.range.resolved_count == 1
