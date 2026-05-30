"""PNG plot generation for Predlog calibration diagnostics.

This module owns matplotlib usage for Predlog. It accepts prediction models,
reuses the summary calculations from :mod:`predlog.stats`, saves PNG files, and
returns the saved path to callers. It does not read SQLite or print terminal
output.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from predlog import config, scoring, stats
from predlog.models import AnyPrediction, RangePrediction


def plot_binary_calibration(
    predictions: Iterable[AnyPrediction],
    output_path: Path | str | None = None,
) -> Path:
    """Create and save the binary calibration plot.

    Args:
        predictions: Prediction models. Open and non-binary predictions are
            ignored by the stats layer.
        output_path: Optional explicit destination. When omitted, Predlog writes
            to ``~/.predlog/plots/binary_calibration.png`` or the equivalent
            ``PREDLOG_HOME`` location.

    Returns:
        The path of the saved PNG file.

    Raises:
        ValueError: If there are no resolved binary predictions to plot.
    """

    binary_stats = stats.summarize_predictions(predictions).binary
    if binary_stats.resolved_count == 0:
        msg = "No resolved binary predictions to plot."
        raise ValueError(msg)

    saved_path = _resolve_output_path(output_path, config.get_binary_plot_path())
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot([0, 100], [0, 100], color="0.45", linestyle="--", label="Perfect calibration")

    if binary_stats.calibration_buckets:
        bucket_labels = [bucket.bucket for bucket in binary_stats.calibration_buckets]
        event_rates = [
            bucket.event_rate * 100 for bucket in binary_stats.calibration_buckets
        ]
        point_sizes = [
            45 + (bucket.count * 14) for bucket in binary_stats.calibration_buckets
        ]
        ax.scatter(bucket_labels, event_rates, s=point_sizes, color="#2563eb", zorder=3)
        for bucket in binary_stats.calibration_buckets:
            _annotate_bucket_count(
                ax,
                label=f"n={bucket.count}",
                x_value=bucket.bucket,
                y_value=bucket.event_rate * 100,
            )

    ax.set_title("Binary Calibration")
    ax.set_xlabel("Predicted probability bucket (%)")
    ax.set_ylabel("Actual event frequency (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.BINARY_CALIBRATION_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    ax.text(
        0.03,
        0.97,
        _binary_stats_text(binary_stats),
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(saved_path)
    plt.close(fig)
    return saved_path


def plot_range_diagnostics(
    predictions: Iterable[AnyPrediction],
    output_path: Path | str | None = None,
) -> Path:
    """Create and save the range diagnostics plot.

    Args:
        predictions: Prediction models. Open and non-range predictions are
            ignored by the stats layer.
        output_path: Optional explicit destination. When omitted, Predlog writes
            to ``~/.predlog/plots/range_diagnostics.png`` or the equivalent
            ``PREDLOG_HOME`` location.

    Returns:
        The path of the saved PNG file.

    Raises:
        ValueError: If there are no resolved range predictions to plot.
    """

    prediction_list = list(predictions)
    range_stats = stats.summarize_predictions(prediction_list).range
    resolved_ranges = _resolved_range_predictions(prediction_list)
    if range_stats.resolved_count == 0:
        msg = "No resolved range predictions to plot."
        raise ValueError(msg)

    saved_path = _resolve_output_path(output_path, config.get_range_plot_path())
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    calibration_ax, width_ax = axes
    _draw_range_calibration_panel(calibration_ax, range_stats)
    _draw_range_width_panel(width_ax, resolved_ranges)

    fig.suptitle("Range Prediction Diagnostics")
    fig.text(
        0.5,
        0.02,
        _range_stats_text(range_stats),
        ha="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(saved_path)
    plt.close(fig)
    return saved_path


def _draw_range_calibration_panel(ax, range_stats: stats.RangeStats) -> None:
    """Draw stated-confidence vs observed-containment calibration."""

    ax.plot([0, 100], [0, 100], color="0.45", linestyle="--", label="Perfect calibration")
    if range_stats.confidence_buckets:
        bucket_labels = [bucket.bucket for bucket in range_stats.confidence_buckets]
        containment_rates = [
            bucket.containment_rate * 100 for bucket in range_stats.confidence_buckets
        ]
        point_sizes = [
            45 + (bucket.count * 14) for bucket in range_stats.confidence_buckets
        ]
        ax.scatter(bucket_labels, containment_rates, s=point_sizes, color="#16a34a", zorder=3)
        for bucket in range_stats.confidence_buckets:
            _annotate_bucket_count(
                ax,
                label=f"n={bucket.count}",
                x_value=bucket.bucket,
                y_value=bucket.containment_rate * 100,
            )

    ax.set_title("Containment Calibration")
    ax.set_xlabel("Stated confidence bucket (%)")
    ax.set_ylabel("Actual containment rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.RANGE_CONFIDENCE_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")


def _draw_range_width_panel(
    ax,
    predictions: list[RangePrediction],
) -> None:
    """Draw interval width diagnostics, preferring relative width when possible."""

    relative_widths = [
        width
        for width in (
            scoring.relative_interval_width(prediction.lower, prediction.upper)
            for prediction in predictions
        )
        if width is not None
    ]

    if relative_widths:
        values = [width * 100 for width in relative_widths]
        y_label = "Relative width (%)"
        title = "Interval Tightness"
    else:
        values = [
            scoring.interval_width(prediction.lower, prediction.upper)
            for prediction in predictions
        ]
        y_label = "Raw width"
        title = "Interval Tightness (Raw Width)"

    x_values = list(range(1, len(values) + 1))
    ax.bar(x_values, values, color="#f59e0b")
    ax.set_title(title)
    ax.set_xlabel("Resolved range prediction")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    ax.grid(True, axis="y", alpha=0.25)


def _resolved_range_predictions(
    predictions: Iterable[AnyPrediction],
) -> list[RangePrediction]:
    """Return resolved range predictions with actual values."""

    return [
        prediction
        for prediction in predictions
        if isinstance(prediction, RangePrediction)
        and prediction.is_resolved
        and prediction.actual is not None
    ]


def _resolve_output_path(output_path: Path | str | None, default_path: Path) -> Path:
    """Return the destination path for a plot."""

    if output_path is None:
        return default_path
    return Path(output_path).expanduser()


def _annotate_bucket_count(ax, *, label: str, x_value: float, y_value: float) -> None:
    """Annotate a calibration bucket count without crossing the plot boundary."""

    if y_value >= 95:
        offset = (0, -10)
        vertical_alignment = "top"
    else:
        offset = (0, 8)
        vertical_alignment = "bottom"

    ax.annotate(
        label,
        (x_value, y_value),
        textcoords="offset points",
        xytext=offset,
        ha="center",
        va=vertical_alignment,
        fontsize=8,
    )


def _binary_stats_text(binary_stats: stats.BinaryStats) -> str:
    """Return text-box content for the binary calibration plot."""

    return (
        f"Resolved: {binary_stats.resolved_count}\n"
        f"Mean Brier: {_format_optional_score(binary_stats.mean_brier_score)}"
    )


def _range_stats_text(range_stats: stats.RangeStats) -> str:
    """Return text-box content for the range diagnostics plot."""

    return (
        f"Resolved: {range_stats.resolved_count}   "
        f"Mean Winkler: {_format_optional_score(range_stats.mean_winkler_score)}   "
        f"Containment: {_format_optional_rate(range_stats.containment_rate)}"
    )


def _format_optional_score(value: float | None) -> str:
    """Format an optional score for plot annotation."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_optional_rate(value: float | None) -> str:
    """Format an optional decimal rate for plot annotation."""

    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"
