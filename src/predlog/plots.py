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
from matplotlib.lines import Line2D

from predlog import config, scoring, stats
from predlog.models import AnyPrediction, RangePrediction


MIN_FILLED_BUCKET_COUNT = 5
CALIBRATION_MARKER_SIZE = 72
MAIN_PLOT_RIGHT_EDGE = 0.74
SIDE_PANEL_X = 0.78
SIDE_PANEL_STATS_Y = 0.88
SIDE_PANEL_LEGEND_Y = 0.64


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

    fig, ax = plt.subplots(figsize=(8.8, 5), dpi=150)
    ax.plot([0, 100], [0, 100], color="0.45", linestyle="--", label="Perfect calibration")

    if binary_stats.calibration_buckets:
        bucket_labels = [bucket.bucket for bucket in binary_stats.calibration_buckets]
        event_rates = [
            bucket.event_rate * 100 for bucket in binary_stats.calibration_buckets
        ]
        bucket_counts = [bucket.count for bucket in binary_stats.calibration_buckets]
        _draw_bucket_markers(
            ax,
            x_values=bucket_labels,
            y_values=event_rates,
            counts=bucket_counts,
            color="#2563eb",
        )

    ax.set_title("Binary Calibration")
    ax.set_xlabel("Predicted probability bucket (%)")
    ax.set_ylabel("Actual event frequency (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.BINARY_CALIBRATION_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, MAIN_PLOT_RIGHT_EDGE, 1))
    _add_side_panel(fig, stats_text=_binary_stats_text(binary_stats), color="#2563eb")
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    calibration_ax, width_ax = axes
    _draw_range_calibration_panel(calibration_ax, range_stats)
    _draw_range_width_panel(width_ax, resolved_ranges)

    fig.suptitle("Range Prediction Diagnostics")
    fig.tight_layout(rect=(0, 0.02, MAIN_PLOT_RIGHT_EDGE, 0.95))
    _add_side_panel(fig, stats_text=_range_stats_text(range_stats), color="#16a34a")
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
        bucket_counts = [bucket.count for bucket in range_stats.confidence_buckets]
        _draw_bucket_markers(
            ax,
            x_values=bucket_labels,
            y_values=containment_rates,
            counts=bucket_counts,
            color="#16a34a",
        )

    ax.set_title("Containment Calibration")
    ax.set_xlabel("Stated confidence bucket (%)")
    ax.set_ylabel("Actual containment rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.RANGE_CONFIDENCE_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)


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


def _add_side_panel(fig, *, stats_text: str, color: str) -> None:
    """Draw calibration summary and marker legend outside the plot axes."""

    fig.text(
        SIDE_PANEL_X,
        SIDE_PANEL_STATS_Y,
        stats_text,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.legend(
        handles=_calibration_legend_handles(color),
        loc="upper left",
        bbox_to_anchor=(SIDE_PANEL_X, SIDE_PANEL_LEGEND_Y),
        borderaxespad=0,
        frameon=True,
    )


def _draw_bucket_markers(
    ax,
    *,
    x_values: list[int],
    y_values: list[float],
    counts: list[int],
    color: str,
) -> None:
    """Draw fixed-size calibration markers, hollowing buckets with little data."""

    sparse_points = [
        (x_value, y_value)
        for x_value, y_value, count in zip(x_values, y_values, counts, strict=True)
        if count < MIN_FILLED_BUCKET_COUNT
    ]
    filled_points = [
        (x_value, y_value)
        for x_value, y_value, count in zip(x_values, y_values, counts, strict=True)
        if count >= MIN_FILLED_BUCKET_COUNT
    ]

    if sparse_points:
        ax.scatter(
            [point[0] for point in sparse_points],
            [point[1] for point in sparse_points],
            s=CALIBRATION_MARKER_SIZE,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            zorder=3,
        )
    if filled_points:
        ax.scatter(
            [point[0] for point in filled_points],
            [point[1] for point in filled_points],
            s=CALIBRATION_MARKER_SIZE,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
            zorder=3,
        )


def _calibration_legend_handles(color: str) -> list[Line2D]:
    """Return legend handles for calibration reference and bucket evidence."""

    return [
        Line2D(
            [],
            [],
            color="0.45",
            linestyle="--",
            label="Perfect calibration",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.8,
            linestyle="None",
            label=f"<{MIN_FILLED_BUCKET_COUNT} in bucket",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.8,
            linestyle="None",
            label=f">={MIN_FILLED_BUCKET_COUNT} in bucket",
        ),
    ]


def _binary_stats_text(binary_stats: stats.BinaryStats) -> str:
    """Return text-box content for the binary calibration plot."""

    return (
        f"Resolved: {binary_stats.resolved_count}\n"
        f"Mean Brier: {_format_optional_score(binary_stats.mean_brier_score)}"
    )


def _range_stats_text(range_stats: stats.RangeStats) -> str:
    """Return text-box content for the range diagnostics plot."""

    return (
        f"Resolved: {range_stats.resolved_count}\n"
        f"Mean Winkler: {_format_optional_score(range_stats.mean_winkler_score)}\n"
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
