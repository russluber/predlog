from datetime import datetime, timezone

import pytest

from predlog import config, plots, stats
from predlog.models import OPEN_STATUS, RESOLVED_STATUS, BinaryPrediction, RangePrediction


CREATED_AT = datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)
RESOLVED_AT = datetime(2026, 5, 31, 9, 0, 0, tzinfo=timezone.utc)


def binary_prediction(probability, outcome):
    """Build a resolved binary prediction for plot tests."""

    return BinaryPrediction(
        id=1,
        question="Binary?",
        created_at=CREATED_AT,
        status=RESOLVED_STATUS,
        resolved_at=RESOLVED_AT,
        probability=probability,
        outcome=outcome,
    )


def range_prediction(lower, upper, confidence, actual):
    """Build a resolved range prediction for plot tests."""

    return RangePrediction(
        id=2,
        question="Range?",
        created_at=CREATED_AT,
        status=RESOLVED_STATUS,
        resolved_at=RESOLVED_AT,
        lower=lower,
        upper=upper,
        confidence=confidence,
        actual=actual,
    )


def open_binary_prediction():
    """Build an open binary prediction for no-data plot tests."""

    return BinaryPrediction(
        id=3,
        question="Open?",
        created_at=CREATED_AT,
        status=OPEN_STATUS,
        probability=0.70,
    )


def assert_png(path):
    """Assert that a path exists and starts with a PNG signature."""

    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_binary_plot_creates_png_at_explicit_path(tmp_path):
    """Binary calibration plot writes a PNG to an explicit destination."""

    output_path = tmp_path / "binary.png"

    saved_path = plots.plot_binary_calibration(
        [
            binary_prediction(0.70, 1),
            binary_prediction(0.80, 0),
            range_prediction(5.0, 12.0, 0.80, 9.0),
        ],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert_png(saved_path)


def test_binary_plot_uses_default_config_path(monkeypatch, tmp_path):
    """Binary plot uses config paths and creates the plots directory."""

    predlog_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(predlog_home))

    saved_path = plots.plot_binary_calibration([binary_prediction(0.70, 1)])

    assert saved_path == predlog_home / "plots" / "binary_calibration.png"
    assert_png(saved_path)


def test_binary_plot_handles_sparse_and_filled_bucket_markers(tmp_path):
    """Binary plot handles sparse and filled bucket marker styles."""

    output_path = tmp_path / "binary-top-edge.png"

    saved_path = plots.plot_binary_calibration(
        [
            binary_prediction(0.60, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
            binary_prediction(0.80, 1),
        ],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert_png(saved_path)


def test_binary_plot_side_panel_includes_correct_lean_rate():
    """Binary plot side-panel text mirrors the binary stats summary."""

    binary_stats = stats.summarize_binary(
        [
            binary_prediction(0.70, 1),
            binary_prediction(0.30, 0),
            binary_prediction(0.60, 0),
        ]
    )

    text = plots._binary_stats_text(binary_stats)

    assert "Number resolved: 3" in text
    assert "Mean Brier score:" in text
    assert "Correct lean rate: 66.7%" in text


def test_binary_plot_raises_with_no_resolved_binary_predictions(tmp_path):
    """Binary plot needs at least one resolved binary prediction."""

    with pytest.raises(ValueError, match="No resolved binary predictions"):
        plots.plot_binary_calibration(
            [open_binary_prediction(), range_prediction(5.0, 12.0, 0.80, 9.0)],
            output_path=tmp_path / "binary.png",
        )


def test_range_plot_creates_png_at_explicit_path(tmp_path):
    """Range diagnostics plot writes a PNG to an explicit destination."""

    output_path = tmp_path / "range.png"

    saved_path = plots.plot_range_diagnostics(
        [
            range_prediction(80.0, 120.0, 0.80, 100.0),
            range_prediction(5.0, 12.0, 0.70, 14.0),
            binary_prediction(0.70, 1),
        ],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert_png(saved_path)


def test_range_plot_handles_many_predictions_per_confidence_bucket(tmp_path):
    """Range sharpness diagnostics stay compact with many resolved predictions."""

    output_path = tmp_path / "range-many.png"
    predictions = [
        range_prediction(40.0, 60.0, bucket / 100, 50.0)
        for bucket in config.RANGE_CONFIDENCE_BUCKETS
        for _ in range(10)
    ]

    saved_path = plots.plot_range_diagnostics(predictions, output_path=output_path)

    assert saved_path == output_path
    assert_png(saved_path)


def test_range_sharpness_scale_uses_median_range_factor():
    """Range marker colors use median range factor from stats output."""

    range_stats = stats.summarize_range(
        [
            range_prediction(80.0, 120.0, 0.80, 100.0),
            range_prediction(90.0, 110.0, 0.80, 100.0),
            range_prediction(50.0, 100.0, 0.80, 75.0),
            range_prediction(100.0, 200.0, 0.70, 150.0),
        ]
    )
    sharpness_scale = plots._range_sharpness_scale(range_stats)

    assert sharpness_scale.bucket_values == pytest.approx({70: 2.0, 80: 1.5})
    assert sharpness_scale.label == "Median range factor"


def test_range_factor_color_scale_is_log_scaled_and_capped():
    """Range factor colors use a stable capped log scale."""

    norm = plots._sharpness_color_norm([1.2, 150.0])

    assert norm.vmin == 1
    assert norm.vmax == 20
    assert norm.clip is True
    assert norm(20.0) == pytest.approx(1.0)
    assert norm(150.0) == pytest.approx(1.0)
    assert plots._format_range_factor_tick(1.5, None) == "1.5x"
    assert plots._format_range_factor_tick(20, None) == ">=20x"


def test_range_plot_side_panel_excludes_sharpness_metric():
    """Range side-panel text keeps sharpness out of the summary box."""

    range_stats = stats.summarize_range(
        [
            range_prediction(80.0, 120.0, 0.80, 100.0),
            range_prediction(90.0, 110.0, 0.80, 100.0),
        ]
    )

    text = plots._range_stats_text(range_stats)

    assert "Number resolved: 2" in text
    assert "Mean Winkler score:" in text
    assert "Inside range rate: 100.0%" in text
    assert "Median range factor" not in text


def test_range_plot_uses_default_config_path(monkeypatch, tmp_path):
    """Range plot uses config paths and creates the plots directory."""

    predlog_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(predlog_home))

    saved_path = plots.plot_range_diagnostics(
        [range_prediction(80.0, 120.0, 0.80, 100.0)]
    )

    assert saved_path == predlog_home / "plots" / "range_calibration_sharpness.png"
    assert_png(saved_path)


def test_range_plot_raises_with_no_resolved_range_predictions(tmp_path):
    """Range plot needs at least one resolved range prediction."""

    with pytest.raises(ValueError, match="No resolved range predictions"):
        plots.plot_range_diagnostics(
            [binary_prediction(0.70, 1), open_binary_prediction()],
            output_path=tmp_path / "range.png",
        )
