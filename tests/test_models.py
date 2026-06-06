from dataclasses import FrozenInstanceError
from datetime import datetime
import math

import pytest

from predlog.models import (
    BINARY_KIND,
    OPEN_STATUS,
    RANGE_KIND,
    RESOLVED_STATUS,
    BinaryPrediction,
    RangePrediction,
)


CREATED_AT = datetime(2026, 5, 30, 9, 0, 0)
RESOLVED_AT = datetime(2026, 5, 31, 9, 0, 0)


def make_open_binary(**overrides):
    """Build a valid open binary prediction for model tests."""

    values = {
        "id": 1,
        "question": "Will it rain tomorrow?",
        "created_at": CREATED_AT,
        "status": OPEN_STATUS,
        "probability": 0.30,
    }
    values.update(overrides)
    return BinaryPrediction(**values)


def make_resolved_binary(**overrides):
    """Build a valid resolved binary prediction for model tests."""

    values = {
        "id": 1,
        "question": "Will it rain tomorrow?",
        "created_at": CREATED_AT,
        "status": RESOLVED_STATUS,
        "resolved_at": RESOLVED_AT,
        "probability": 0.30,
        "outcome": 1,
    }
    values.update(overrides)
    return BinaryPrediction(**values)


def make_open_range(**overrides):
    """Build a valid open range prediction for model tests."""

    values = {
        "id": 2,
        "question": "How many hours will this project take?",
        "created_at": CREATED_AT,
        "status": OPEN_STATUS,
        "lower": 5.0,
        "upper": 12.0,
        "confidence": 0.80,
    }
    values.update(overrides)
    return RangePrediction(**values)


def make_resolved_range(**overrides):
    """Build a valid resolved range prediction for model tests."""

    values = {
        "id": 2,
        "question": "How many hours will this project take?",
        "created_at": CREATED_AT,
        "status": RESOLVED_STATUS,
        "resolved_at": RESOLVED_AT,
        "lower": 5.0,
        "upper": 12.0,
        "confidence": 0.80,
        "actual": 9.5,
    }
    values.update(overrides)
    return RangePrediction(**values)


def test_open_binary_prediction_fields():
    """Open binary predictions carry probability but no outcome."""

    prediction = make_open_binary()

    assert prediction.id == 1
    assert prediction.kind == BINARY_KIND
    assert prediction.question == "Will it rain tomorrow?"
    assert prediction.created_at == CREATED_AT
    assert prediction.status == OPEN_STATUS
    assert prediction.resolved_at is None
    assert prediction.probability == pytest.approx(0.30)
    assert prediction.outcome is None
    assert prediction.is_open is True
    assert prediction.is_resolved is False


def test_resolved_binary_prediction_fields():
    """Resolved binary predictions carry outcome and resolution timestamp."""

    prediction = make_resolved_binary()

    assert prediction.kind == BINARY_KIND
    assert prediction.status == RESOLVED_STATUS
    assert prediction.resolved_at == RESOLVED_AT
    assert prediction.outcome == 1
    assert prediction.is_open is False
    assert prediction.is_resolved is True


def test_open_range_prediction_fields():
    """Open range predictions carry interval and confidence but no actual."""

    prediction = make_open_range()

    assert prediction.id == 2
    assert prediction.kind == RANGE_KIND
    assert prediction.question == "How many hours will this project take?"
    assert prediction.created_at == CREATED_AT
    assert prediction.status == OPEN_STATUS
    assert prediction.resolved_at is None
    assert prediction.lower == pytest.approx(5.0)
    assert prediction.upper == pytest.approx(12.0)
    assert prediction.confidence == pytest.approx(0.80)
    assert prediction.actual is None
    assert prediction.is_open is True
    assert prediction.is_resolved is False


def test_resolved_range_prediction_fields():
    """Resolved range predictions carry actual value and resolution timestamp."""

    prediction = make_resolved_range()

    assert prediction.kind == RANGE_KIND
    assert prediction.status == RESOLVED_STATUS
    assert prediction.resolved_at == RESOLVED_AT
    assert prediction.actual == pytest.approx(9.5)
    assert prediction.is_open is False
    assert prediction.is_resolved is True


def test_models_are_immutable():
    """Prediction models are frozen after creation."""

    prediction = make_open_binary()

    with pytest.raises(FrozenInstanceError):
        prediction.question = "A different question"


@pytest.mark.parametrize("prediction_id", [0, -1, 1.5, True])
def test_common_fields_reject_invalid_id(prediction_id):
    """Prediction IDs must be positive integers."""

    with pytest.raises(ValueError):
        make_open_binary(id=prediction_id)


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_common_fields_reject_invalid_question(question):
    """Prediction questions must be non-empty text."""

    with pytest.raises(ValueError):
        make_open_binary(question=question)


@pytest.mark.parametrize("created_at", ["2026-05-30", None])
def test_common_fields_reject_invalid_created_at(created_at):
    """Prediction creation timestamp must be a datetime."""

    with pytest.raises(ValueError):
        make_open_binary(created_at=created_at)


def test_common_fields_reject_invalid_resolved_at():
    """Resolution timestamp must be a datetime when present."""

    with pytest.raises(ValueError):
        make_resolved_binary(resolved_at="2026-05-31")


def test_common_fields_reject_unknown_status():
    """Prediction status must match the v0.1 lifecycle labels."""

    with pytest.raises(ValueError):
        make_open_binary(status="pending")


def test_open_predictions_cannot_have_resolved_at():
    """Open predictions cannot carry a resolution timestamp."""

    with pytest.raises(ValueError):
        make_open_binary(resolved_at=RESOLVED_AT)


def test_resolved_predictions_must_have_resolved_at():
    """Resolved predictions must carry a resolution timestamp."""

    with pytest.raises(ValueError):
        make_resolved_binary(resolved_at=None)


@pytest.mark.parametrize("probability", [-0.01, 1.01, math.inf, math.nan, True])
def test_binary_prediction_rejects_invalid_probability(probability):
    """Binary probability must be finite and between 0 and 1."""

    with pytest.raises(ValueError):
        make_open_binary(probability=probability)


@pytest.mark.parametrize("outcome", [-1, 2, 1.5, True])
def test_binary_prediction_rejects_invalid_outcome(outcome):
    """Binary outcome must be an integer encoded as 0 or 1."""

    with pytest.raises(ValueError):
        make_resolved_binary(outcome=outcome)


def test_open_binary_prediction_cannot_have_outcome():
    """Open binary predictions cannot already have an outcome."""

    with pytest.raises(ValueError):
        make_open_binary(outcome=1)


def test_resolved_binary_prediction_must_have_outcome():
    """Resolved binary predictions must have an outcome."""

    with pytest.raises(ValueError):
        make_resolved_binary(outcome=None)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (5.0, 5.0),
        (12.0, 5.0),
        (0.0, 12.0),
        (-1.0, 12.0),
        (math.nan, 12.0),
        (5.0, math.inf),
    ],
)
def test_range_prediction_rejects_invalid_interval(lower, upper):
    """Range intervals must be finite, positive-scale, and strict."""

    with pytest.raises(ValueError):
        make_open_range(lower=lower, upper=upper)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, math.inf, math.nan])
def test_range_prediction_rejects_invalid_confidence(confidence):
    """Range confidence must be finite, greater than 0, and less than 1."""

    with pytest.raises(ValueError):
        make_open_range(confidence=confidence)


@pytest.mark.parametrize("actual", [math.inf, math.nan, True])
def test_range_prediction_rejects_invalid_actual(actual):
    """Resolved actual values must be finite numbers."""

    with pytest.raises(ValueError):
        make_resolved_range(actual=actual)


def test_open_range_prediction_cannot_have_actual():
    """Open range predictions cannot already have an actual value."""

    with pytest.raises(ValueError):
        make_open_range(actual=9.5)


def test_resolved_range_prediction_must_have_actual():
    """Resolved range predictions must have an actual value."""

    with pytest.raises(ValueError):
        make_resolved_range(actual=None)
