from pathlib import Path

import pytest
from typer.testing import CliRunner

from predlog import config, storage
from predlog.cli import app
from predlog.models import OPEN_STATUS, RESOLVED_STATUS, BinaryPrediction, RangePrediction


runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_predlog_home(monkeypatch, tmp_path):
    """Point every CLI test at an isolated Predlog data directory."""

    predlog_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(predlog_home))
    return predlog_home


def test_binary_command_creates_open_binary_prediction():
    """The binary command stores a probability entered as a percentage."""

    result = runner.invoke(
        app,
        ["binary", "Will it rain tomorrow?", "--prob", "30"],
    )

    assert result.exit_code == 0, result.output
    assert "Logged binary prediction" in result.output
    assert "Will it rain tomorrow?" in result.output
    assert "30 percent" in result.output

    predictions = storage.list_open_predictions()
    assert len(predictions) == 1
    prediction = predictions[0]
    assert isinstance(prediction, BinaryPrediction)
    assert prediction.question == "Will it rain tomorrow?"
    assert prediction.probability == pytest.approx(0.30)
    assert prediction.status == OPEN_STATUS


def test_range_command_creates_open_range_prediction():
    """The range command stores confidence entered as a percentage."""

    result = runner.invoke(
        app,
        [
            "range",
            "How many hours will this project take?",
            "--low",
            "5",
            "--high",
            "12",
            "--conf",
            "80",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Logged range prediction" in result.output
    assert "How many hours will this project take?" in result.output
    assert "[5, 12]" in result.output
    assert "80 percent" in result.output

    predictions = storage.list_open_predictions()
    assert len(predictions) == 1
    prediction = predictions[0]
    assert isinstance(prediction, RangePrediction)
    assert prediction.lower == pytest.approx(5.0)
    assert prediction.upper == pytest.approx(12.0)
    assert prediction.confidence == pytest.approx(0.80)
    assert prediction.status == OPEN_STATUS


def test_binary_command_rejects_invalid_probability():
    """Binary probabilities must be greater than 0 and less than 100 percent."""

    for probability in ["0", "100"]:
        result = runner.invoke(
            app,
            ["binary", "Invalid probability?", "--prob", probability],
        )

        assert result.exit_code == 1
        assert (
            "probability must be greater than 0 and less than 100 percent"
            in result.output
        )
    assert storage.list_predictions() == []


def test_range_command_rejects_invalid_confidence():
    """Range confidence must be greater than 0 and less than 100 percent."""

    result = runner.invoke(
        app,
        [
            "range",
            "Invalid confidence?",
            "--low",
            "5",
            "--high",
            "12",
            "--conf",
            "100",
        ],
    )

    assert result.exit_code == 1
    assert "confidence must be greater than 0 and less than 100 percent" in result.output
    assert storage.list_predictions() == []


def test_range_command_rejects_invalid_bounds():
    """Range lower bounds must be strictly less than upper bounds."""

    result = runner.invoke(
        app,
        [
            "range",
            "Invalid interval?",
            "--low",
            "12",
            "--high",
            "5",
            "--conf",
            "80",
        ],
    )

    assert result.exit_code == 1
    assert "interval lower bound must be less than upper bound" in result.output
    assert storage.list_predictions() == []


def test_range_command_rejects_non_positive_lower_bound():
    """Range predictions must stay on a positive scale."""

    result = runner.invoke(
        app,
        [
            "range",
            "Invalid positive-scale interval?",
            "--low",
            "0",
            "--high",
            "5",
            "--conf",
            "80",
        ],
    )

    assert result.exit_code == 1
    assert "range lower bound must be greater than zero" in result.output
    assert storage.list_predictions() == []


def test_list_command_shows_all_open_and_resolved_predictions():
    """The list command can show all, open, or resolved predictions."""

    binary = storage.add_binary_prediction("Open binary?", 0.60)
    resolved_range = storage.add_range_prediction("Resolved range?", 5.0, 12.0, 0.80)
    storage.resolve_range_prediction(resolved_range.id, 9.5)

    all_result = runner.invoke(app, ["list"])
    open_result = runner.invoke(app, ["list", "open"])
    resolved_result = runner.invoke(app, ["list", "resolved"])

    assert all_result.exit_code == 0, all_result.output
    assert "Open binary?" in all_result.output
    assert "Resolved range?" in all_result.output
    assert str(binary.id) in all_result.output

    assert open_result.exit_code == 0, open_result.output
    assert "Open binary?" in open_result.output
    assert "Resolved range?" not in open_result.output

    assert resolved_result.exit_code == 0, resolved_result.output
    assert "Resolved range?" in resolved_result.output
    assert "Open binary?" not in resolved_result.output
    assert RESOLVED_STATUS in resolved_result.output


def test_list_command_empty_states():
    """The list command prints friendly empty messages."""

    all_result = runner.invoke(app, ["list"])
    open_result = runner.invoke(app, ["list", "open"])
    resolved_result = runner.invoke(app, ["list", "resolved"])

    assert all_result.exit_code == 0
    assert "No predictions yet." in all_result.output
    assert open_result.exit_code == 0
    assert "No open predictions." in open_result.output
    assert resolved_result.exit_code == 0
    assert "No resolved predictions." in resolved_result.output


def test_list_command_rejects_unknown_status():
    """The list command only accepts open and resolved filters."""

    result = runner.invoke(app, ["list", "pending"])

    assert result.exit_code == 1
    assert "Status must be 'open' or 'resolved'." in result.output


def test_delete_command_removes_open_prediction():
    """The delete command removes open predictions by ID."""

    prediction = storage.add_binary_prediction("Delete this?", 0.60)
    storage.add_range_prediction("Keep this?", 5.0, 12.0, 0.80)

    result = runner.invoke(app, ["delete", str(prediction.id)])

    assert result.exit_code == 0, result.output
    assert "Deleted prediction" in result.output
    assert f"ID: {prediction.id}" in result.output
    assert "Status: open" in result.output
    assert "Type: binary" in result.output
    assert "Delete this?" in result.output
    assert [item.question for item in storage.list_predictions()] == ["Keep this?"]


def test_delete_command_requires_force_for_resolved_prediction():
    """Resolved predictions require --force for deletion."""

    prediction = storage.add_binary_prediction("Protect this resolved prediction?", 0.70)
    storage.resolve_binary_prediction(prediction.id, 1)

    result = runner.invoke(app, ["delete", str(prediction.id)])

    assert result.exit_code == 1
    assert "use --force to delete it" in result.output
    assert len(storage.list_predictions()) == 1


def test_delete_command_removes_resolved_prediction_with_force():
    """The delete command removes resolved predictions when forced."""

    prediction = storage.add_range_prediction(
        "Delete this resolved range prediction?",
        5.0,
        12.0,
        0.80,
    )
    storage.resolve_range_prediction(prediction.id, 9.5)

    result = runner.invoke(app, ["delete", str(prediction.id), "--force"])

    assert result.exit_code == 0, result.output
    assert "Deleted prediction" in result.output
    assert "Status: resolved" in result.output
    assert "Type: range" in result.output
    assert storage.list_predictions() == []


def test_delete_command_reports_missing_prediction():
    """The delete command reports unknown IDs cleanly."""

    result = runner.invoke(app, ["delete", "1"])

    assert result.exit_code == 1
    assert "prediction 1 was not found" in result.output


def test_resolve_command_resolves_binary_prediction_interactively():
    """Interactive binary resolution prints immediate Brier feedback."""

    prediction = storage.add_binary_prediction("Will it rain tomorrow?", 0.70)

    result = runner.invoke(app, ["resolve"], input="1\nyes\n")

    assert result.exit_code == 0, result.output
    assert "[1] Will it rain tomorrow?" in result.output
    assert "Outcome: yes" in result.output
    assert "Brier score: 0.090" in result.output

    resolved_predictions = storage.list_resolved_predictions()
    assert len(resolved_predictions) == 1
    resolved = resolved_predictions[0]
    assert resolved.id == prediction.id
    assert isinstance(resolved, BinaryPrediction)
    assert resolved.outcome == 1
    assert resolved.status == RESOLVED_STATUS


def test_resolve_command_resolves_range_prediction_interactively():
    """Interactive range resolution prints containment and Winkler feedback."""

    prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
    )

    result = runner.invoke(app, ["resolve"], input="1\n9.5\n")

    assert result.exit_code == 0, result.output
    assert "[1] How many hours will this project take?" in result.output
    assert "Actual value: 9.5" in result.output
    assert "Contained in interval: yes" in result.output
    assert "Winkler score: 7.000" in result.output

    resolved_predictions = storage.list_resolved_predictions()
    assert len(resolved_predictions) == 1
    resolved = resolved_predictions[0]
    assert resolved.id == prediction.id
    assert isinstance(resolved, RangePrediction)
    assert resolved.actual == pytest.approx(9.5)
    assert resolved.status == RESOLVED_STATUS


def test_resolve_command_resolves_binary_prediction_directly_with_yes():
    """Direct binary resolution avoids the interactive menu."""

    prediction = storage.add_binary_prediction("Will it rain tomorrow?", 0.70)

    result = runner.invoke(app, ["resolve", str(prediction.id), "--yes"])

    assert result.exit_code == 0, result.output
    assert "Resolved: Will it rain tomorrow?" in result.output
    assert "Outcome: yes" in result.output
    assert "Brier score: 0.090" in result.output
    assert "[1] Will it rain tomorrow?" not in result.output

    resolved = storage.list_resolved_predictions()[0]
    assert isinstance(resolved, BinaryPrediction)
    assert resolved.id == prediction.id
    assert resolved.outcome == 1


def test_resolve_command_resolves_binary_prediction_directly_with_no():
    """Direct binary resolution supports no outcomes."""

    prediction = storage.add_binary_prediction("Will it rain tomorrow?", 0.30)

    result = runner.invoke(app, ["resolve", str(prediction.id), "--no"])

    assert result.exit_code == 0, result.output
    assert "Outcome: no" in result.output
    assert "Brier score: 0.090" in result.output

    resolved = storage.list_resolved_predictions()[0]
    assert isinstance(resolved, BinaryPrediction)
    assert resolved.outcome == 0


def test_resolve_command_resolves_range_prediction_directly_with_actual():
    """Direct range resolution accepts an actual value."""

    prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
    )

    result = runner.invoke(app, ["resolve", str(prediction.id), "--actual", "9.5"])

    assert result.exit_code == 0, result.output
    assert "Resolved: How many hours will this project take?" in result.output
    assert "Actual value: 9.5" in result.output
    assert "Contained in interval: yes" in result.output
    assert "Winkler score: 7.000" in result.output

    resolved = storage.list_resolved_predictions()[0]
    assert isinstance(resolved, RangePrediction)
    assert resolved.id == prediction.id
    assert resolved.actual == pytest.approx(9.5)


def test_resolve_command_rejects_direct_options_without_id():
    """Fast resolve options require an explicit prediction ID."""

    result = runner.invoke(app, ["resolve", "--yes"])

    assert result.exit_code == 1
    assert "Pass a prediction ID" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["resolve", "1"],
        ["resolve", "1", "--yes", "--no"],
        ["resolve", "1", "--yes", "--actual", "9.5"],
    ],
)
def test_resolve_command_rejects_invalid_direct_option_combinations(args):
    """Direct resolve mode rejects ambiguous or incomplete options."""

    result = runner.invoke(app, args)

    assert result.exit_code == 1


def test_resolve_command_rejects_wrong_direct_resolver_for_prediction_kind():
    """Direct resolve mode reports kind mismatches cleanly."""

    binary = storage.add_binary_prediction("Will it rain tomorrow?", 0.70)
    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        5.0,
        12.0,
        0.80,
    )

    binary_result = runner.invoke(app, ["resolve", str(binary.id), "--actual", "9.5"])
    range_result = runner.invoke(app, ["resolve", str(range_prediction.id), "--yes"])

    assert binary_result.exit_code == 1
    assert "is binary, not range" in binary_result.output
    assert range_result.exit_code == 1
    assert "is range, not binary" in range_result.output


def test_resolve_command_handles_no_open_predictions():
    """Resolve exits cleanly when there are no open predictions."""

    result = runner.invoke(app, ["resolve"])

    assert result.exit_code == 0
    assert "No open predictions to resolve." in result.output


def test_stats_command_handles_no_resolved_predictions():
    """Stats exits cleanly when there is no resolved data yet."""

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    assert "No resolved binary predictions yet." in result.output
    assert "No resolved range predictions yet." in result.output
    assert "Binary calibration" not in result.output
    assert "Range calibration" not in result.output


def test_stats_command_summarizes_resolved_predictions():
    """Stats prints binary and range summaries from resolved predictions."""

    binary = storage.add_binary_prediction("Will it rain tomorrow?", 0.70)
    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        80.0,
        120.0,
        0.80,
    )
    storage.resolve_binary_prediction(binary.id, 1)
    storage.resolve_range_prediction(range_prediction.id, 100.0)

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0, result.output
    assert "Binary predictions" in result.output
    assert "Number of resolved predictions: 1" in result.output
    assert "Mean Brier score: 0.090" in result.output
    assert "Correct lean rate: 100.0 percent" in result.output
    assert "Binary calibration" in result.output
    assert "Bucket" in result.output
    assert "Average predicted chance" in result.output
    assert "Actual event rate" in result.output
    assert "Calibration gap" in result.output
    assert "Feedback" in result.output
    assert "70%" in result.output
    assert "70.0%" in result.output
    assert "100.0%" in result.output
    assert "+30.0%" in result.output
    assert "sparse" in result.output
    assert "Not enough data" in result.output
    assert "Range predictions" in result.output
    assert "Mean Winkler score: 40.000" in result.output
    assert "Containment rate: 100.0 percent" in result.output
    assert "Range sharpness" in result.output
    assert "Median range factor: 1.50x" in result.output
    assert "Average width" not in result.output
    assert "Average relative width" not in result.output
    assert "Range calibration" in result.output
    assert "Confidence" in result.output
    assert "Inside range rate" in result.output
    assert "Median range factor" in result.output
    assert "1.50x" in result.output
    assert "80.0%" in result.output


def test_stats_command_marks_calibration_buckets_with_enough_evidence():
    """Stats labels buckets with five or more resolved predictions as enough."""

    binary_outcomes = [1, 1, 1, 1, 0]
    for outcome in binary_outcomes:
        prediction = storage.add_binary_prediction("Will the bucket fill?", 0.80)
        storage.resolve_binary_prediction(prediction.id, outcome)

    range_actuals = [100.0, 102.0, 98.0, 101.0, 99.0]
    for actual in range_actuals:
        prediction = storage.add_range_prediction(
            "Will the actual land in range?",
            90.0,
            110.0,
            0.80,
        )
        storage.resolve_range_prediction(prediction.id, actual)

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0, result.output
    assert "Binary calibration" in result.output
    assert "Range calibration" in result.output
    assert "80%" in result.output
    assert "80.0%" in result.output
    assert "100.0%" in result.output
    assert "+0.0%" in result.output
    assert "+20.0%" in result.output
    assert "enough" in result.output
    assert "About right" in result.output
    assert "Too wide" in result.output


def test_plot_binary_command_creates_default_plot():
    """The binary plot command saves the default calibration PNG."""

    binary = storage.add_binary_prediction("Will it rain tomorrow?", 0.70)
    storage.resolve_binary_prediction(binary.id, 1)

    result = runner.invoke(app, ["plot", "binary"])

    plot_path = config.get_binary_plot_path()
    assert result.exit_code == 0, result.output
    assert "Saved binary calibration plot" in result.output
    assert str(plot_path) in result.output
    assert plot_path.exists()


def test_plot_range_command_creates_default_plot():
    """The range plot command saves the default diagnostics PNG."""

    range_prediction = storage.add_range_prediction(
        "How many hours will this project take?",
        80.0,
        120.0,
        0.80,
    )
    storage.resolve_range_prediction(range_prediction.id, 100.0)

    result = runner.invoke(app, ["plot", "range"])

    plot_path = config.get_range_plot_path()
    assert result.exit_code == 0, result.output
    assert "Saved range calibration and sharpness plot" in result.output
    assert str(plot_path) in result.output
    assert plot_path.exists()


def test_plot_commands_handle_no_resolved_data():
    """Plot commands fail cleanly when their resolved data is missing."""

    binary_result = runner.invoke(app, ["plot", "binary"])
    range_result = runner.invoke(app, ["plot", "range"])

    assert binary_result.exit_code == 1
    assert "No resolved binary predictions to plot." in binary_result.output
    assert range_result.exit_code == 1
    assert "No resolved range predictions to plot." in range_result.output


def test_where_command_respects_predlog_home(isolated_predlog_home):
    """The where command shows paths derived from PREDLOG_HOME."""

    result = runner.invoke(app, ["where"])

    assert result.exit_code == 0, result.output
    assert str(isolated_predlog_home) in result.output
    assert str(isolated_predlog_home / "predlog.db") in result.output
    assert str(isolated_predlog_home / "plots") in result.output
    assert "binary_calibration.png" in result.output
    assert "range_calibration_sharpness.png" in result.output


def test_cli_help_runs():
    """The Typer app exposes command help."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "prediction journal" in result.output


def test_where_command_does_not_create_database(isolated_predlog_home):
    """Showing paths does not initialize local storage."""

    result = runner.invoke(app, ["where"])

    assert result.exit_code == 0, result.output
    assert not Path(isolated_predlog_home / "predlog.db").exists()
