"""Data models shared across Predlog modules.

The models in this module are deliberately lightweight. They describe the
Python shape of a prediction after it has been read from storage or passed
between internal layers. They do not know how to query SQLite, print terminal
tables, generate plots, or compute scores.

Predlog v0.1 has two concrete prediction types:

* :class:`BinaryPrediction` for yes/no events with a probability.
* :class:`RangePrediction` for numerical intervals with a confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Literal, TypeAlias

BINARY_KIND = "binary"
"""Storage and model label for binary yes/no predictions."""

RANGE_KIND = "range"
"""Storage and model label for numerical range predictions."""

OPEN_STATUS = "open"
"""Status label for predictions that do not have an outcome yet."""

RESOLVED_STATUS = "resolved"
"""Status label for predictions with a recorded outcome or actual value."""

PredictionKind: TypeAlias = Literal["binary", "range"]
"""Allowed prediction kind values."""

PredictionStatus: TypeAlias = Literal["open", "resolved"]
"""Allowed prediction lifecycle status values."""


@dataclass(frozen=True, kw_only=True)
class Prediction:
    """Common fields shared by every Predlog prediction.

    This base class captures the lifecycle metadata that both binary and range
    predictions need. In normal application code, use :class:`BinaryPrediction`
    or :class:`RangePrediction` rather than instantiating this class directly.

    Attributes:
        id: Permanent internal database ID.
        question: User-written prediction question.
        created_at: Timestamp when the prediction was logged.
        status: Either ``"open"`` or ``"resolved"``.
        resolved_at: Timestamp when the prediction was resolved, or ``None`` if
            it is still open.
    """

    id: int
    question: str
    created_at: datetime
    status: PredictionStatus
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate common prediction fields after dataclass initialization."""

        _validate_id(self.id)
        _validate_question(self.question)
        _validate_datetime("created_at", self.created_at)
        _validate_status(self.status)
        _validate_resolution_timestamp(self.status, self.resolved_at)

    @property
    def is_open(self) -> bool:
        """Return ``True`` when this prediction is still awaiting resolution."""

        return self.status == OPEN_STATUS

    @property
    def is_resolved(self) -> bool:
        """Return ``True`` when this prediction has been resolved."""

        return self.status == RESOLVED_STATUS


@dataclass(frozen=True, kw_only=True)
class BinaryPrediction(Prediction):
    """A yes/no prediction with a probability forecast.

    Attributes:
        kind: Always ``"binary"``.
        probability: Forecast probability as a decimal from ``0.0`` to ``1.0``.
        outcome: Resolved outcome, where ``1`` means yes and ``0`` means no.
            Open binary predictions must have ``outcome=None``.
    """

    kind: Literal["binary"] = field(default=BINARY_KIND, init=False)
    probability: float
    outcome: int | None = None

    def __post_init__(self) -> None:
        """Validate binary-specific fields and common prediction fields."""

        super().__post_init__()
        _validate_probability(self.probability)
        _validate_binary_resolution(self.status, self.outcome)


@dataclass(frozen=True, kw_only=True)
class RangePrediction(Prediction):
    """A numerical interval prediction with a confidence forecast.

    Attributes:
        kind: Always ``"range"``.
        lower: Lower bound of the forecast interval. Predlog range intervals
            must be strictly positive so range sharpness can be compared with
            multiplicative range factors.
        upper: Upper bound of the forecast interval.
        confidence: Stated interval confidence as a decimal greater than
            ``0.0`` and less than ``1.0``.
        actual: Resolved numerical value. Open range predictions must have
            ``actual=None``.
    """

    kind: Literal["range"] = field(default=RANGE_KIND, init=False)
    lower: float
    upper: float
    confidence: float
    actual: float | None = None

    def __post_init__(self) -> None:
        """Validate range-specific fields and common prediction fields."""

        super().__post_init__()
        _validate_interval(self.lower, self.upper)
        _validate_confidence(self.confidence)
        _validate_range_resolution(self.status, self.actual)


AnyPrediction: TypeAlias = BinaryPrediction | RangePrediction
"""Union type for code that can work with either concrete prediction model."""


def _validate_id(prediction_id: int) -> None:
    """Raise ValueError unless a prediction ID is a positive integer."""

    if isinstance(prediction_id, bool) or not isinstance(prediction_id, int):
        msg = "prediction id must be a positive integer"
        raise ValueError(msg)
    if prediction_id < 1:
        msg = "prediction id must be a positive integer"
        raise ValueError(msg)


def _validate_question(question: str) -> None:
    """Raise ValueError unless a prediction question is non-empty text."""

    if not isinstance(question, str) or not question.strip():
        msg = "question must be non-empty text"
        raise ValueError(msg)


def _validate_datetime(name: str, value: datetime) -> None:
    """Raise ValueError unless a timestamp value is a datetime object."""

    if not isinstance(value, datetime):
        msg = f"{name} must be a datetime"
        raise ValueError(msg)


def _validate_status(status: PredictionStatus) -> None:
    """Raise ValueError unless a prediction status is known."""

    if status not in {OPEN_STATUS, RESOLVED_STATUS}:
        msg = "status must be 'open' or 'resolved'"
        raise ValueError(msg)


def _validate_resolution_timestamp(
    status: PredictionStatus, resolved_at: datetime | None
) -> None:
    """Raise ValueError unless status and resolved_at agree."""

    if resolved_at is not None:
        _validate_datetime("resolved_at", resolved_at)
    if status == OPEN_STATUS and resolved_at is not None:
        msg = "open predictions cannot have resolved_at"
        raise ValueError(msg)
    if status == RESOLVED_STATUS and resolved_at is None:
        msg = "resolved predictions must have resolved_at"
        raise ValueError(msg)


def _validate_binary_resolution(
    status: PredictionStatus, outcome: int | None
) -> None:
    """Raise ValueError unless binary status and outcome agree."""

    if status == OPEN_STATUS and outcome is not None:
        msg = "open binary predictions cannot have an outcome"
        raise ValueError(msg)
    if status == RESOLVED_STATUS and outcome is None:
        msg = "resolved binary predictions must have an outcome"
        raise ValueError(msg)
    if outcome is not None:
        _validate_outcome(outcome)


def _validate_range_resolution(
    status: PredictionStatus, actual: float | None
) -> None:
    """Raise ValueError unless range status and actual value agree."""

    if status == OPEN_STATUS and actual is not None:
        msg = "open range predictions cannot have an actual value"
        raise ValueError(msg)
    if status == RESOLVED_STATUS and actual is None:
        msg = "resolved range predictions must have an actual value"
        raise ValueError(msg)
    if actual is not None:
        _validate_finite_number("actual", actual)


def _validate_probability(probability: float) -> None:
    """Raise ValueError unless a binary probability is finite and in range."""

    _validate_finite_number("probability", probability)
    if not 0 <= probability <= 1:
        msg = "probability must be between 0.0 and 1.0"
        raise ValueError(msg)


def _validate_outcome(outcome: int) -> None:
    """Raise ValueError unless a binary outcome is exactly 0 or 1."""

    if isinstance(outcome, bool) or not isinstance(outcome, int):
        msg = "outcome must be 0 or 1"
        raise ValueError(msg)
    if outcome not in {0, 1}:
        msg = "outcome must be 0 or 1"
        raise ValueError(msg)


def _validate_interval(lower: float, upper: float) -> None:
    """Raise ValueError unless interval bounds are finite, positive, and strict."""

    _validate_finite_number("lower", lower)
    _validate_finite_number("upper", upper)
    if lower <= 0:
        msg = "range lower bound must be greater than zero"
        raise ValueError(msg)
    if lower >= upper:
        msg = "interval lower bound must be less than upper bound"
        raise ValueError(msg)


def _validate_confidence(confidence: float) -> None:
    """Raise ValueError unless range confidence is finite and strictly valid."""

    _validate_finite_number("confidence", confidence)
    if not 0 < confidence < 1:
        msg = "confidence must be greater than 0.0 and less than 1.0"
        raise ValueError(msg)


def _validate_finite_number(name: str, value: float) -> None:
    """Raise ValueError unless a value is a finite int or float."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be a finite number"
        raise ValueError(msg)
    if not math.isfinite(value):
        msg = f"{name} must be a finite number"
        raise ValueError(msg)
