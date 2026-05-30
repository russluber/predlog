from datetime import datetime, timezone
import math
import sqlite3

import pytest

from predlog import config, storage
from predlog.models import (
    BINARY_KIND,
    OPEN_STATUS,
    RANGE_KIND,
    RESOLVED_STATUS,
    BinaryPrediction,
    RangePrediction,
)


CREATED_BINARY = datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)
CREATED_RANGE = datetime(2026, 5, 30, 10, 0, 0, tzinfo=timezone.utc)
RESOLVED_AT = datetime(2026, 5, 31, 9, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def db_path(tmp_path):
    """Return a nested test database path."""

    return tmp_path / "nested" / "predlog.db"


def fetch_rows(db_path):
    """Fetch all raw prediction rows from a test database."""

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT * FROM predictions ORDER BY id ASC"
        ).fetchall()


def count_rows(db_path):
    """Count raw prediction rows in a test database."""

    with sqlite3.connect(db_path) as connection:
        return connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def test_initialize_database_creates_parent_directory_and_table(db_path):
    """Database initialization creates the parent directory and schema."""

    initialized_path = storage.initialize_database(db_path)

    assert initialized_path == db_path
    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        table_name = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'predictions'
            """
        ).fetchone()
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        ]

    assert table_name == ("predictions",)
    assert columns == [
        "id",
        "kind",
        "question",
        "created_at",
        "resolved_at",
        "status",
        "probability",
        "outcome",
        "lower",
        "upper",
        "confidence",
        "actual",
    ]


def test_add_binary_prediction_returns_model_and_persists_columns(db_path):
    """Binary inserts return a model and write only binary-specific columns."""

    prediction = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.30,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )

    assert isinstance(prediction, BinaryPrediction)
    assert prediction.id == 1
    assert prediction.kind == BINARY_KIND
    assert prediction.question == "Will it rain tomorrow?"
    assert prediction.created_at == CREATED_BINARY
    assert prediction.status == OPEN_STATUS
    assert prediction.resolved_at is None
    assert prediction.probability == pytest.approx(0.30)
    assert prediction.outcome is None

    row = fetch_rows(db_path)[0]
    assert row["kind"] == BINARY_KIND
    assert row["question"] == "Will it rain tomorrow?"
    assert row["created_at"] == CREATED_BINARY.isoformat()
    assert row["resolved_at"] is None
    assert row["status"] == OPEN_STATUS
    assert row["probability"] == pytest.approx(0.30)
    assert row["outcome"] is None
    assert row["lower"] is None
    assert row["upper"] is None
    assert row["confidence"] is None
    assert row["actual"] is None


def test_add_range_prediction_returns_model_and_persists_columns(db_path):
    """Range inserts return a model and write only range-specific columns."""

    prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )

    assert isinstance(prediction, RangePrediction)
    assert prediction.id == 1
    assert prediction.kind == RANGE_KIND
    assert prediction.question == "How many hours will this project take?"
    assert prediction.created_at == CREATED_RANGE
    assert prediction.status == OPEN_STATUS
    assert prediction.resolved_at is None
    assert prediction.lower == pytest.approx(5.0)
    assert prediction.upper == pytest.approx(12.0)
    assert prediction.confidence == pytest.approx(0.80)
    assert prediction.actual is None

    row = fetch_rows(db_path)[0]
    assert row["kind"] == RANGE_KIND
    assert row["question"] == "How many hours will this project take?"
    assert row["created_at"] == CREATED_RANGE.isoformat()
    assert row["resolved_at"] is None
    assert row["status"] == OPEN_STATUS
    assert row["probability"] is None
    assert row["outcome"] is None
    assert row["lower"] == pytest.approx(5.0)
    assert row["upper"] == pytest.approx(12.0)
    assert row["confidence"] == pytest.approx(0.80)
    assert row["actual"] is None


def test_list_predictions_filters_and_orders_deterministically(db_path):
    """Listing supports all/open/resolved filters with stable ordering."""

    later_binary = storage.add_binary_prediction(
        "Later binary?",
        0.60,
        db_path=db_path,
        created_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
    )
    range_prediction = storage.add_range_prediction(
        "Middle range?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
    )
    earlier_binary = storage.add_binary_prediction(
        "Earlier binary?",
        0.40,
        db_path=db_path,
        created_at=datetime(2026, 5, 30, 8, 0, tzinfo=timezone.utc),
    )

    storage.resolve_binary_prediction(
        later_binary.id,
        1,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    assert [item.id for item in storage.list_predictions(db_path=db_path)] == [
        earlier_binary.id,
        range_prediction.id,
        later_binary.id,
    ]
    assert [item.id for item in storage.list_open_predictions(db_path=db_path)] == [
        earlier_binary.id,
        range_prediction.id,
    ]
    assert [
        item.id for item in storage.list_resolved_predictions(db_path=db_path)
    ] == [later_binary.id]


def test_resolve_binary_prediction_records_outcome_and_timestamp(db_path):
    """Resolving a binary prediction updates status, outcome, and timestamp."""

    prediction = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )

    resolved = storage.resolve_binary_prediction(
        prediction.id,
        1,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    assert isinstance(resolved, BinaryPrediction)
    assert resolved.id == prediction.id
    assert resolved.status == RESOLVED_STATUS
    assert resolved.resolved_at == RESOLVED_AT
    assert resolved.outcome == 1

    row = fetch_rows(db_path)[0]
    assert row["status"] == RESOLVED_STATUS
    assert row["resolved_at"] == RESOLVED_AT.isoformat()
    assert row["outcome"] == 1


def test_resolve_range_prediction_records_actual_and_timestamp(db_path):
    """Resolving a range prediction updates status, actual, and timestamp."""

    prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )

    resolved = storage.resolve_range_prediction(
        prediction.id,
        9.5,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    assert isinstance(resolved, RangePrediction)
    assert resolved.id == prediction.id
    assert resolved.status == RESOLVED_STATUS
    assert resolved.resolved_at == RESOLVED_AT
    assert resolved.actual == pytest.approx(9.5)

    row = fetch_rows(db_path)[0]
    assert row["status"] == RESOLVED_STATUS
    assert row["resolved_at"] == RESOLVED_AT.isoformat()
    assert row["actual"] == pytest.approx(9.5)


def test_resolved_predictions_move_between_open_and_resolved_lists(db_path):
    """Resolved predictions disappear from open lists and appear in resolved lists."""

    binary = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )
    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )

    assert [item.id for item in storage.list_open_predictions(db_path=db_path)] == [
        binary.id,
        range_prediction.id,
    ]

    storage.resolve_binary_prediction(
        binary.id,
        0,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    assert [item.id for item in storage.list_open_predictions(db_path=db_path)] == [
        range_prediction.id
    ]
    assert [
        item.id for item in storage.list_resolved_predictions(db_path=db_path)
    ] == [binary.id]


def test_load_resolved_predictions_aliases_resolved_list(db_path):
    """Stats and plots can load the same resolved set."""

    binary = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )
    storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )
    storage.resolve_binary_prediction(
        binary.id,
        1,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    assert storage.load_resolved_predictions(
        db_path=db_path
    ) == storage.list_resolved_predictions(db_path=db_path)


def test_missing_prediction_raises_custom_error(db_path):
    """Resolving a valid but unknown ID raises PredictionNotFoundError."""

    storage.initialize_database(db_path)

    with pytest.raises(storage.PredictionNotFoundError):
        storage.resolve_binary_prediction(1, 1, db_path=db_path)


def test_already_resolved_prediction_raises_custom_error(db_path):
    """Predictions cannot be resolved twice."""

    prediction = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )
    storage.resolve_binary_prediction(
        prediction.id,
        1,
        db_path=db_path,
        resolved_at=RESOLVED_AT,
    )

    with pytest.raises(storage.PredictionAlreadyResolvedError):
        storage.resolve_binary_prediction(prediction.id, 0, db_path=db_path)


def test_wrong_kind_resolution_raises_custom_error(db_path):
    """Binary and range resolvers reject the opposite prediction kind."""

    binary = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )
    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )

    with pytest.raises(storage.PredictionKindMismatchError):
        storage.resolve_range_prediction(binary.id, 9.5, db_path=db_path)

    with pytest.raises(storage.PredictionKindMismatchError):
        storage.resolve_binary_prediction(range_prediction.id, 1, db_path=db_path)


def test_invalid_insert_input_does_not_add_rows(db_path):
    """Invalid insert input raises before any row is added."""

    storage.initialize_database(db_path)

    with pytest.raises(ValueError):
        storage.add_binary_prediction(
            "Invalid probability",
            1.01,
            db_path=db_path,
            created_at=CREATED_BINARY,
        )

    with pytest.raises(ValueError):
        storage.add_range_prediction(
            "Invalid interval",
            12.0,
            5.0,
            0.80,
            db_path=db_path,
            created_at=CREATED_RANGE,
        )

    assert count_rows(db_path) == 0


def test_invalid_resolve_input_does_not_update_row(db_path):
    """Invalid resolution input raises before the stored row changes."""

    binary = storage.add_binary_prediction(
        "Will it rain tomorrow?",
        0.70,
        db_path=db_path,
        created_at=CREATED_BINARY,
    )
    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
        db_path=db_path,
        created_at=CREATED_RANGE,
    )

    with pytest.raises(ValueError):
        storage.resolve_binary_prediction(binary.id, 2, db_path=db_path)

    with pytest.raises(ValueError):
        storage.resolve_range_prediction(range_prediction.id, math.inf, db_path=db_path)

    assert [item.status for item in storage.list_predictions(db_path=db_path)] == [
        OPEN_STATUS,
        OPEN_STATUS,
    ]


def test_naive_timestamps_are_treated_as_utc(db_path):
    """Naive timestamps are stored and returned as UTC-aware values."""

    naive_created_at = datetime(2026, 5, 30, 9, 0, 0)
    naive_resolved_at = datetime(2026, 5, 31, 9, 0, 0)

    prediction = storage.add_binary_prediction(
        "Will this timestamp be UTC?",
        0.50,
        db_path=db_path,
        created_at=naive_created_at,
    )
    resolved = storage.resolve_binary_prediction(
        prediction.id,
        1,
        db_path=db_path,
        resolved_at=naive_resolved_at,
    )

    assert prediction.created_at == naive_created_at.replace(tzinfo=timezone.utc)
    assert resolved.resolved_at == naive_resolved_at.replace(tzinfo=timezone.utc)


def test_default_path_respects_predlog_home(monkeypatch, tmp_path):
    """Omitting db_path uses config and therefore respects PREDLOG_HOME."""

    predlog_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(predlog_home))

    prediction = storage.add_binary_prediction(
        "Will default storage work?",
        0.80,
        created_at=CREATED_BINARY,
    )

    assert prediction.id == 1
    assert config.get_database_path() == predlog_home / "predlog.db"
    assert (predlog_home / "predlog.db").exists()
    assert count_rows(predlog_home / "predlog.db") == 1


def test_list_predictions_rejects_unknown_status(db_path):
    """Status filters must match the v0.1 lifecycle labels."""

    with pytest.raises(ValueError):
        storage.list_predictions("pending", db_path=db_path)


@pytest.mark.parametrize("prediction_id", [0, -1, True, 1.5])
def test_resolve_rejects_invalid_prediction_id(db_path, prediction_id):
    """Resolve operations require positive integer IDs."""

    with pytest.raises(ValueError):
        storage.resolve_binary_prediction(prediction_id, 1, db_path=db_path)
