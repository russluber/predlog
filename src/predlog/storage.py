"""SQLite persistence for Predlog predictions.

This module owns Predlog's local database. It creates the v0.1 schema, inserts
new predictions, lists predictions, resolves predictions, and converts database
rows into the dataclasses from :mod:`predlog.models`.

The rest of the application should not need to know SQL details. CLI, stats,
and plotting code can work with :class:`predlog.models.BinaryPrediction` and
:class:`predlog.models.RangePrediction` objects instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from predlog import config
from predlog.models import (
    BINARY_KIND,
    OPEN_STATUS,
    RANGE_KIND,
    RESOLVED_STATUS,
    AnyPrediction,
    BinaryPrediction,
    PredictionStatus,
    RangePrediction,
)

CREATE_PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('binary', 'range')),
    question TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    probability REAL,
    outcome INTEGER,
    lower REAL,
    upper REAL,
    confidence REAL,
    actual REAL
)
"""


class StorageError(Exception):
    """Base class for Predlog storage errors."""


class PredictionNotFoundError(StorageError):
    """Raised when a prediction ID does not exist in the database."""

    def __init__(self, prediction_id: int) -> None:
        """Create an error for a missing prediction ID."""

        super().__init__(f"prediction {prediction_id} was not found")
        self.prediction_id = prediction_id


class PredictionAlreadyResolvedError(StorageError):
    """Raised when code tries to resolve a prediction more than once."""

    def __init__(self, prediction_id: int) -> None:
        """Create an error for an already-resolved prediction ID."""

        super().__init__(f"prediction {prediction_id} is already resolved")
        self.prediction_id = prediction_id


class PredictionKindMismatchError(StorageError):
    """Raised when a prediction is resolved with the wrong resolver."""

    def __init__(self, prediction_id: int, expected: str, actual: str) -> None:
        """Create an error for a prediction kind mismatch."""

        super().__init__(
            f"prediction {prediction_id} is {actual}, not {expected}"
        )
        self.prediction_id = prediction_id
        self.expected = expected
        self.actual = actual


def initialize_database(db_path: Path | str | None = None) -> Path:
    """Create the Predlog SQLite database and predictions table if needed.

    Args:
        db_path: Optional explicit database path. When omitted, Predlog uses
            :func:`predlog.config.get_database_path`.

    Returns:
        The resolved database path that was initialized.

    The schema intentionally matches the v0.1 architecture: one ``predictions``
    table and no score, tag, note, or due-date columns.
    """

    database_path = _resolve_database_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(database_path) as connection:
        connection.execute(CREATE_PREDICTIONS_TABLE_SQL)
    return database_path


def add_binary_prediction(
    question: str,
    probability: float,
    *,
    db_path: Path | str | None = None,
    created_at: datetime | None = None,
) -> BinaryPrediction:
    """Insert and return a new open binary prediction.

    Args:
        question: User-written prediction question.
        probability: Forecast probability as a decimal from ``0.0`` to ``1.0``.
        db_path: Optional explicit SQLite database path.
        created_at: Optional creation timestamp. Naive timestamps are treated
            as UTC.

    Returns:
        The persisted :class:`BinaryPrediction`, including its database ID.
    """

    created_at_utc = _normalize_timestamp(created_at)
    BinaryPrediction(
        id=1,
        question=question,
        created_at=created_at_utc,
        status=OPEN_STATUS,
        probability=probability,
    )

    database_path = initialize_database(db_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                kind, question, created_at, resolved_at, status, probability,
                outcome, lower, upper, confidence, actual
            )
            VALUES (?, ?, ?, NULL, ?, ?, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                BINARY_KIND,
                question,
                _timestamp_to_text(created_at_utc),
                OPEN_STATUS,
                probability,
            ),
        )
        return _get_prediction(connection, cursor.lastrowid, BinaryPrediction)


def add_range_prediction(
    question: str,
    lower: float,
    upper: float,
    confidence: float,
    *,
    db_path: Path | str | None = None,
    created_at: datetime | None = None,
) -> RangePrediction:
    """Insert and return a new open range prediction.

    Args:
        question: User-written prediction question.
        lower: Lower bound of the forecast interval.
        upper: Upper bound of the forecast interval.
        confidence: Stated interval confidence as a decimal greater than
            ``0.0`` and less than ``1.0``.
        db_path: Optional explicit SQLite database path.
        created_at: Optional creation timestamp. Naive timestamps are treated
            as UTC.

    Returns:
        The persisted :class:`RangePrediction`, including its database ID.
    """

    created_at_utc = _normalize_timestamp(created_at)
    RangePrediction(
        id=1,
        question=question,
        created_at=created_at_utc,
        status=OPEN_STATUS,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )

    database_path = initialize_database(db_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                kind, question, created_at, resolved_at, status, probability,
                outcome, lower, upper, confidence, actual
            )
            VALUES (?, ?, ?, NULL, ?, NULL, NULL, ?, ?, ?, NULL)
            """,
            (
                RANGE_KIND,
                question,
                _timestamp_to_text(created_at_utc),
                OPEN_STATUS,
                lower,
                upper,
                confidence,
            ),
        )
        return _get_prediction(connection, cursor.lastrowid, RangePrediction)


def list_predictions(
    status: PredictionStatus | None = None,
    *,
    db_path: Path | str | None = None,
) -> list[AnyPrediction]:
    """Return predictions, optionally filtered by lifecycle status.

    Results are ordered by ``created_at`` and then ``id`` so command output and
    tests stay deterministic.
    """

    if status is not None:
        _validate_status(status)

    database_path = initialize_database(db_path)
    with _connect(database_path) as connection:
        if status is None:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                WHERE status = ?
                ORDER BY created_at ASC, id ASC
                """,
                (status,),
            ).fetchall()
    return [_row_to_prediction(row) for row in rows]


def list_open_predictions(
    *,
    db_path: Path | str | None = None,
) -> list[AnyPrediction]:
    """Return all open predictions."""

    return list_predictions(OPEN_STATUS, db_path=db_path)


def list_resolved_predictions(
    *,
    db_path: Path | str | None = None,
) -> list[AnyPrediction]:
    """Return all resolved predictions."""

    return list_predictions(RESOLVED_STATUS, db_path=db_path)


def load_resolved_predictions(
    *,
    db_path: Path | str | None = None,
) -> list[AnyPrediction]:
    """Return resolved predictions for stats and plot generation."""

    return list_resolved_predictions(db_path=db_path)


def resolve_binary_prediction(
    prediction_id: int,
    outcome: int,
    *,
    db_path: Path | str | None = None,
    resolved_at: datetime | None = None,
) -> BinaryPrediction:
    """Resolve an open binary prediction and return the updated model.

    Raises:
        PredictionNotFoundError: If ``prediction_id`` is not in the database.
        PredictionKindMismatchError: If the prediction is not binary.
        PredictionAlreadyResolvedError: If the prediction is already resolved.
        ValueError: If ``outcome`` is not ``0`` or ``1``.
    """

    _validate_prediction_id(prediction_id)
    resolved_at_utc = _normalize_timestamp(resolved_at)

    database_path = initialize_database(db_path)
    with _connect(database_path) as connection:
        prediction = _get_prediction(connection, prediction_id)
        _ensure_kind(prediction, prediction_id, BINARY_KIND)
        _ensure_open(prediction, prediction_id)
        assert isinstance(prediction, BinaryPrediction)

        BinaryPrediction(
            id=prediction.id,
            question=prediction.question,
            created_at=prediction.created_at,
            status=RESOLVED_STATUS,
            resolved_at=resolved_at_utc,
            probability=prediction.probability,
            outcome=outcome,
        )

        connection.execute(
            """
            UPDATE predictions
            SET status = ?, resolved_at = ?, outcome = ?
            WHERE id = ?
            """,
            (
                RESOLVED_STATUS,
                _timestamp_to_text(resolved_at_utc),
                outcome,
                prediction_id,
            ),
        )
        return _get_prediction(connection, prediction_id, BinaryPrediction)


def resolve_range_prediction(
    prediction_id: int,
    actual: float,
    *,
    db_path: Path | str | None = None,
    resolved_at: datetime | None = None,
) -> RangePrediction:
    """Resolve an open range prediction and return the updated model.

    Raises:
        PredictionNotFoundError: If ``prediction_id`` is not in the database.
        PredictionKindMismatchError: If the prediction is not range-based.
        PredictionAlreadyResolvedError: If the prediction is already resolved.
        ValueError: If ``actual`` is not a finite number.
    """

    _validate_prediction_id(prediction_id)
    resolved_at_utc = _normalize_timestamp(resolved_at)

    database_path = initialize_database(db_path)
    with _connect(database_path) as connection:
        prediction = _get_prediction(connection, prediction_id)
        _ensure_kind(prediction, prediction_id, RANGE_KIND)
        _ensure_open(prediction, prediction_id)
        assert isinstance(prediction, RangePrediction)

        RangePrediction(
            id=prediction.id,
            question=prediction.question,
            created_at=prediction.created_at,
            status=RESOLVED_STATUS,
            resolved_at=resolved_at_utc,
            lower=prediction.lower,
            upper=prediction.upper,
            confidence=prediction.confidence,
            actual=actual,
        )

        connection.execute(
            """
            UPDATE predictions
            SET status = ?, resolved_at = ?, actual = ?
            WHERE id = ?
            """,
            (
                RESOLVED_STATUS,
                _timestamp_to_text(resolved_at_utc),
                actual,
                prediction_id,
            ),
        )
        return _get_prediction(connection, prediction_id, RangePrediction)


def _resolve_database_path(db_path: Path | str | None) -> Path:
    """Return the active database path."""

    if db_path is None:
        return config.get_database_path()
    return Path(db_path).expanduser()


@contextmanager
def _connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection and close it after the transaction."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _get_prediction(
    connection: sqlite3.Connection,
    prediction_id: int,
    expected_type: type[BinaryPrediction] | type[RangePrediction] | None = None,
) -> AnyPrediction:
    """Load a prediction by ID and convert it to a model."""

    row = connection.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (prediction_id,),
    ).fetchone()
    if row is None:
        raise PredictionNotFoundError(prediction_id)

    prediction = _row_to_prediction(row)
    if expected_type is not None and not isinstance(prediction, expected_type):
        raise PredictionKindMismatchError(
            prediction_id,
            _kind_for_model_type(expected_type),
            prediction.kind,
        )
    return prediction


def _row_to_prediction(row: sqlite3.Row) -> AnyPrediction:
    """Convert a SQLite row from the predictions table into a model."""

    created_at = _timestamp_from_text(row["created_at"])
    resolved_at = _optional_timestamp_from_text(row["resolved_at"])

    if row["kind"] == BINARY_KIND:
        return BinaryPrediction(
            id=row["id"],
            question=row["question"],
            created_at=created_at,
            resolved_at=resolved_at,
            status=row["status"],
            probability=row["probability"],
            outcome=row["outcome"],
        )
    if row["kind"] == RANGE_KIND:
        return RangePrediction(
            id=row["id"],
            question=row["question"],
            created_at=created_at,
            resolved_at=resolved_at,
            status=row["status"],
            lower=row["lower"],
            upper=row["upper"],
            confidence=row["confidence"],
            actual=row["actual"],
        )

    msg = f"unknown prediction kind: {row['kind']}"
    raise StorageError(msg)


def _kind_for_model_type(
    model_type: type[BinaryPrediction] | type[RangePrediction],
) -> str:
    """Return the prediction kind label for a concrete model class."""

    if model_type is BinaryPrediction:
        return BINARY_KIND
    return RANGE_KIND


def _ensure_kind(
    prediction: AnyPrediction,
    prediction_id: int,
    expected_kind: str,
) -> None:
    """Raise unless a prediction has the expected kind."""

    if prediction.kind != expected_kind:
        raise PredictionKindMismatchError(
            prediction_id,
            expected_kind,
            prediction.kind,
        )


def _ensure_open(prediction: AnyPrediction, prediction_id: int) -> None:
    """Raise unless a prediction is open."""

    if prediction.status == RESOLVED_STATUS:
        raise PredictionAlreadyResolvedError(prediction_id)


def _validate_prediction_id(prediction_id: int) -> None:
    """Raise ValueError unless a prediction ID is a positive integer."""

    if isinstance(prediction_id, bool) or not isinstance(prediction_id, int):
        msg = "prediction id must be a positive integer"
        raise ValueError(msg)
    if prediction_id < 1:
        msg = "prediction id must be a positive integer"
        raise ValueError(msg)


def _validate_status(status: PredictionStatus) -> None:
    """Raise ValueError unless a prediction status is known."""

    if status not in {OPEN_STATUS, RESOLVED_STATUS}:
        msg = "status must be 'open' or 'resolved'"
        raise ValueError(msg)


def _normalize_timestamp(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC timestamp.

    A missing timestamp is generated from the current time. A naive supplied
    timestamp is treated as UTC so tests and callers can pass simple datetime
    values without inheriting local machine timezone behavior.
    """

    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        msg = "timestamp must be a datetime"
        raise ValueError(msg)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp_to_text(value: datetime) -> str:
    """Serialize a timestamp as UTC ISO text for SQLite."""

    return _normalize_timestamp(value).isoformat()


def _timestamp_from_text(value: str) -> datetime:
    """Parse UTC ISO timestamp text from SQLite."""

    return _normalize_timestamp(datetime.fromisoformat(value))


def _optional_timestamp_from_text(value: str | None) -> datetime | None:
    """Parse optional UTC ISO timestamp text from SQLite."""

    if value is None:
        return None
    return _timestamp_from_text(value)
