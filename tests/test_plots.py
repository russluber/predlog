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


def test_range_sharpness_values_use_typical_uncertainty():
    """Range sharpness uses half of relative width to match stats output."""

    bucket_labels, values, y_label, title = plots._range_sharpness_bucket_values(
        [
            range_prediction(80.0, 120.0, 0.80, 100.0),
            range_prediction(90.0, 110.0, 0.80, 100.0),
            range_prediction(40.0, 60.0, 0.70, 50.0),
        ]
    )

    assert bucket_labels == [70, 80]
    assert values == pytest.approx([20.0, 15.0])
    assert y_label == "Typical uncertainty (%)"
    assert title == "Sharpness"


def test_range_plot_uses_default_config_path(monkeypatch, tmp_path):
    """Range plot uses config paths and creates the plots directory."""

    predlog_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(predlog_home))

    saved_path = plots.plot_range_diagnostics(
        [range_prediction(80.0, 120.0, 0.80, 100.0)]
    )

    assert saved_path == predlog_home / "plots" / "range_calibration_sharpness.png"
    assert_png(saved_path)


def test_range_plot_falls_back_to_raw_width_when_relative_width_is_unavailable(
    tmp_path,
):
    """Range plot still works when every interval midpoint is near zero."""

    output_path = tmp_path / "range-raw-width.png"

    saved_path = plots.plot_range_diagnostics(
        [
            range_prediction(-1.0, 1.0, 0.80, 0.0),
            range_prediction(-2.0, 2.0, 0.70, 0.0),
        ],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert_png(saved_path)


def test_range_sharpness_values_fall_back_to_raw_width():
    """Range sharpness falls back to raw width when midpoint scale is unusable."""

    bucket_labels, values, y_label, title = plots._range_sharpness_bucket_values(
        [
            range_prediction(-1.0, 1.0, 0.80, 0.0),
            range_prediction(-2.0, 2.0, 0.70, 0.0),
        ]
    )

    assert bucket_labels == [70, 80]
    assert values == pytest.approx([4.0, 2.0])
    assert y_label == "Average raw width"
    assert title == "Raw Width by Confidence"


def test_range_plot_raises_with_no_resolved_range_predictions(tmp_path):
    """Range plot needs at least one resolved range prediction."""

    with pytest.raises(ValueError, match="No resolved range predictions"):
        plots.plot_range_diagnostics(
            [binary_prediction(0.70, 1), open_binary_prediction()],
            output_path=tmp_path / "range.png",
        )
