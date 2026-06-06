"""PNG plot generation for Predlog calibration diagnostics.

This module owns matplotlib usage for Predlog. It accepts prediction models,
reuses the summary calculations from :mod:`predlog.stats`, saves PNG files, and
returns the saved path to callers. It does not read SQLite or print terminal
output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from predlog import config, stats
from predlog.models import AnyPrediction


CALIBRATION_MARKER_SIZE = 72
MAIN_PLOT_RIGHT_EDGE = 0.74
SIDE_PANEL_X = 0.78
SIDE_PANEL_STATS_Y = 0.88
SIDE_PANEL_LEGEND_Y = 0.64
RANGE_SHARPNESS_COLORMAP = "viridis"
RANGE_FACTOR_COLOR_CAP = 20
RANGE_FACTOR_COLOR_TICKS = (1, 1.5, 2, 3, 5, 10, RANGE_FACTOR_COLOR_CAP)


@dataclass(frozen=True)
class _RangeSharpnessScale:
    """Bucket-level range factors used to color range calibration markers."""

    bucket_values: dict[int, float]
    label: str


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

    ax.set_title("Calibration")
    ax.set_xlabel("Predicted probability bucket (%)")
    ax.set_ylabel("Actual event rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.BINARY_CALIBRATION_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)

    fig.suptitle("Binary Predictions", fontweight="bold")
    fig.tight_layout(rect=(0, 0, MAIN_PLOT_RIGHT_EDGE, 0.92))
    _add_side_panel(fig, stats_text=_binary_stats_text(binary_stats), color="#2563eb")
    fig.savefig(saved_path)
    plt.close(fig)
    return saved_path


def plot_range_diagnostics(
    predictions: Iterable[AnyPrediction],
    output_path: Path | str | None = None,
) -> Path:
    """Create and save the range calibration and sharpness plot.

    Args:
        predictions: Prediction models. Open and non-range predictions are
            ignored by the stats layer.
        output_path: Optional explicit destination. When omitted, Predlog writes
            to ``~/.predlog/plots/range_calibration_sharpness.png`` or the
            equivalent ``PREDLOG_HOME`` location.

    Returns:
        The path of the saved PNG file.

    Raises:
        ValueError: If there are no resolved range predictions to plot.
    """

    prediction_list = list(predictions)
    range_stats = stats.summarize_predictions(prediction_list).range
    if range_stats.resolved_count == 0:
        msg = "No resolved range predictions to plot."
        raise ValueError(msg)

    saved_path = _resolve_output_path(output_path, config.get_range_plot_path())
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 5.2), dpi=150)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(4.5, 1.45),
        left=0.08,
        right=0.96,
        bottom=0.14,
        top=0.82,
        wspace=0.32,
    )
    calibration_ax = fig.add_subplot(grid[0, 0])
    side_ax = fig.add_subplot(grid[0, 1])
    side_ax.axis("off")

    sharpness_scale = _range_sharpness_scale(range_stats)
    sharpness_mappable = _draw_range_calibration_panel(
        calibration_ax,
        range_stats,
        sharpness_scale=sharpness_scale,
    )

    fig.suptitle("Range Predictions", fontweight="bold")
    if sharpness_mappable is not None:
        _add_range_sharpness_colorbar(
            fig,
            calibration_ax,
            sharpness_mappable,
            sharpness_scale,
        )
    _add_side_panel_to_axis(
        side_ax,
        stats_text=_range_stats_text(range_stats),
        color="0.25",
    )
    fig.savefig(saved_path)
    plt.close(fig)
    return saved_path


def _draw_range_calibration_panel(
    ax,
    range_stats: stats.RangeStats,
    *,
    sharpness_scale: _RangeSharpnessScale,
) -> ScalarMappable | None:
    """Draw stated-confidence vs observed-containment calibration.

    Marker position shows calibration. Marker color shows median range factor
    for that confidence bucket, so sharpness reads as a refinement of the same
    calibration view instead of a separate chart.
    """

    ax.plot([0, 100], [0, 100], color="0.45", linestyle="--", label="Perfect calibration")
    sharpness_mappable = None
    if range_stats.confidence_buckets:
        bucket_labels = [bucket.bucket for bucket in range_stats.confidence_buckets]
        containment_rates = [
            bucket.containment_rate * 100 for bucket in range_stats.confidence_buckets
        ]
        bucket_counts = [bucket.count for bucket in range_stats.confidence_buckets]
        sharpness_mappable = _draw_sharpness_bucket_markers(
            ax,
            x_values=bucket_labels,
            y_values=containment_rates,
            counts=bucket_counts,
            sharpness_scale=sharpness_scale,
        )

    ax.set_title("Calibration & Sharpness")
    ax.set_xlabel("Confidence bucket (%)")
    ax.set_ylabel("Inside range rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_xticks(config.RANGE_CONFIDENCE_BUCKETS)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.25)
    return sharpness_mappable


def _range_sharpness_scale(range_stats: stats.RangeStats) -> _RangeSharpnessScale:
    """Return bucketed median range factors for coloring range markers."""

    return _RangeSharpnessScale(
        bucket_values={
            bucket.bucket: bucket.median_range_factor
            for bucket in range_stats.confidence_buckets
        },
        label="Median range factor",
    )


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


def _add_side_panel_to_axis(ax, *, stats_text: str, color: str) -> None:
    """Draw calibration summary and marker legend inside a side-panel axis."""

    ax.text(
        0,
        1,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(
        handles=_calibration_legend_handles(color),
        loc="upper left",
        bbox_to_anchor=(0, 0.64),
        borderaxespad=0,
        frameon=True,
    )


def _add_range_sharpness_colorbar(
    fig,
    ax,
    mappable: ScalarMappable,
    sharpness_scale: _RangeSharpnessScale,
) -> None:
    """Draw the range-factor color scale for the range calibration plot."""

    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.04)
    colorbar.ax.set_title(sharpness_scale.label, pad=8)
    colorbar.set_ticks(RANGE_FACTOR_COLOR_TICKS)
    colorbar.ax.yaxis.set_major_formatter(_range_factor_formatter)


def _draw_sharpness_bucket_markers(
    ax,
    *,
    x_values: list[int],
    y_values: list[float],
    counts: list[int],
    sharpness_scale: _RangeSharpnessScale,
) -> ScalarMappable | None:
    """Draw range calibration markers colored by bucket median range factor."""

    colored_points = [
        (x_value, y_value, count, sharpness_scale.bucket_values[x_value])
        for x_value, y_value, count in zip(x_values, y_values, counts, strict=True)
        if x_value in sharpness_scale.bucket_values
    ]
    uncolored_points = [
        (x_value, y_value, count)
        for x_value, y_value, count in zip(x_values, y_values, counts, strict=True)
        if x_value not in sharpness_scale.bucket_values
    ]

    if uncolored_points:
        _draw_bucket_markers(
            ax,
            x_values=[point[0] for point in uncolored_points],
            y_values=[point[1] for point in uncolored_points],
            counts=[point[2] for point in uncolored_points],
            color="0.45",
        )

    if not colored_points:
        return None

    values = [point[3] for point in colored_points]
    norm = _sharpness_color_norm(values)
    cmap = plt.get_cmap(RANGE_SHARPNESS_COLORMAP)
    sparse_points = [
        point
        for point in colored_points
        if point[2] < config.CALIBRATION_MIN_EVIDENCE_COUNT
    ]
    filled_points = [
        point
        for point in colored_points
        if point[2] >= config.CALIBRATION_MIN_EVIDENCE_COUNT
    ]

    if sparse_points:
        ax.scatter(
            [point[0] for point in sparse_points],
            [point[1] for point in sparse_points],
            s=CALIBRATION_MARKER_SIZE,
            facecolors="none",
            edgecolors=[cmap(norm(point[3])) for point in sparse_points],
            linewidths=2.2,
            zorder=3,
        )
    if filled_points:
        ax.scatter(
            [point[0] for point in filled_points],
            [point[1] for point in filled_points],
            s=CALIBRATION_MARKER_SIZE,
            c=[point[3] for point in filled_points],
            cmap=cmap,
            norm=norm,
            edgecolors="0.25",
            linewidths=0.8,
            zorder=3,
        )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(values)
    return mappable


def _sharpness_color_norm(values: list[float]) -> Normalize:
    """Return a log normalization for multiplicative range factors."""

    if not values:
        return LogNorm(vmin=1, vmax=RANGE_FACTOR_COLOR_CAP, clip=True)
    return LogNorm(vmin=1, vmax=RANGE_FACTOR_COLOR_CAP, clip=True)


def _format_range_factor_tick(value: float, _position: int | None) -> str:
    """Format the fixed range-factor colorbar with a capped top label."""

    if value >= RANGE_FACTOR_COLOR_CAP:
        return f">={RANGE_FACTOR_COLOR_CAP:g}x"
    return f"{value:g}x"


_range_factor_formatter = FuncFormatter(_format_range_factor_tick)


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
        if count < config.CALIBRATION_MIN_EVIDENCE_COUNT
    ]
    filled_points = [
        (x_value, y_value)
        for x_value, y_value, count in zip(x_values, y_values, counts, strict=True)
        if count >= config.CALIBRATION_MIN_EVIDENCE_COUNT
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
            label=f"<{config.CALIBRATION_MIN_EVIDENCE_COUNT} in bucket",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.8,
            linestyle="None",
            label=f">={config.CALIBRATION_MIN_EVIDENCE_COUNT} in bucket",
        ),
    ]


def _binary_stats_text(binary_stats: stats.BinaryStats) -> str:
    """Return text-box content for the binary calibration plot."""

    return (
        f"Number resolved: {binary_stats.resolved_count}\n"
        f"Mean Brier score: {_format_optional_score(binary_stats.mean_brier_score)}\n"
        f"Correct lean rate: {_format_optional_rate(binary_stats.directional_hit_rate)}"
    )


def _range_stats_text(range_stats: stats.RangeStats) -> str:
    """Return text-box content for the range calibration and sharpness plot."""

    return (
        f"Number resolved: {range_stats.resolved_count}\n"
        f"Mean Winkler score: {_format_optional_score(range_stats.mean_winkler_score)}\n"
        f"Inside range rate: {_format_optional_rate(range_stats.containment_rate)}"
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
