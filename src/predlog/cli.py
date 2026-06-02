"""Command line interface for Predlog.

The CLI layer is responsible for parsing terminal commands, validating
user-facing percentage arguments, calling storage, and printing readable output.
It deliberately does not contain SQL or scoring formulas; those stay in
``storage.py`` and ``scoring.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.markup import escape
from rich.table import Table
import typer

from predlog import config, scoring, stats as stats_module, storage
from predlog.models import (
    OPEN_STATUS,
    RESOLVED_STATUS,
    AnyPrediction,
    BinaryPrediction,
    PredictionStatus,
    RangePrediction,
)

app = typer.Typer(
    help="A local-first prediction journal for personal calibration tracking.",
    no_args_is_help=True,
)
plot_app = typer.Typer(help="Generate calibration and diagnostics plots.")
app.add_typer(plot_app, name="plot")
console = Console(highlight=False, width=160)


@app.command()
def binary(
    question: Annotated[
        str,
        typer.Argument(help="The yes/no prediction question to log."),
    ],
    prob: Annotated[
        float,
        typer.Option(
            ...,
            "--prob",
            help="Forecast probability as a percentage greater than 0 and less than 100.",
        ),
    ],
) -> None:
    """Log a binary yes/no prediction."""

    probability = _percentage_to_decimal(
        prob,
        name="probability",
        allow_zero=False,
        allow_hundred=False,
    )
    try:
        prediction = storage.add_binary_prediction(question, probability)
    except ValueError as error:
        _fail(str(error))

    console.print("[green]Logged binary prediction[/green]")
    console.print(f"ID: {prediction.id}")
    console.print(f"Question: {escape(prediction.question)}")
    console.print(f"Forecast: {_format_percent(prediction.probability)}")


@app.command("range")
def range_command(
    question: Annotated[
        str,
        typer.Argument(help="The numerical range prediction question to log."),
    ],
    low: Annotated[
        float,
        typer.Option(..., "--low", help="Lower bound of the forecast interval."),
    ],
    high: Annotated[
        float,
        typer.Option(..., "--high", help="Upper bound of the forecast interval."),
    ],
    conf: Annotated[
        float,
        typer.Option(
            ...,
            "--conf",
            help="Interval confidence as a percentage greater than 0 and less than 100.",
        ),
    ],
) -> None:
    """Log a numerical range prediction."""

    confidence = _percentage_to_decimal(
        conf,
        name="confidence",
        allow_zero=False,
        allow_hundred=False,
    )
    try:
        prediction = storage.add_range_prediction(question, low, high, confidence)
    except ValueError as error:
        _fail(str(error))

    console.print("[green]Logged range prediction[/green]")
    console.print(f"ID: {prediction.id}")
    console.print(f"Question: {escape(prediction.question)}")
    console.print(
        "Interval: "
        f"{_format_interval(prediction.lower, prediction.upper)} "
        f"at {_format_percent(prediction.confidence)} confidence"
    )


@app.command("list")
def list_command(
    status: Annotated[
        str | None,
        typer.Argument(help="Optional filter: open or resolved."),
    ] = None,
) -> None:
    """List logged predictions."""

    normalized_status = _normalize_status_filter(status)
    predictions = storage.list_predictions(normalized_status)
    if not predictions:
        console.print(_empty_list_message(normalized_status))
        return

    table = Table(title=_list_title(normalized_status))
    table.add_column("ID", justify="right")
    table.add_column("Status")
    table.add_column("Type")
    table.add_column("Question", overflow="fold")
    table.add_column("Forecast", overflow="fold")
    table.add_column("Result")
    table.add_column("Created")
    table.add_column("Resolved")

    for prediction in predictions:
        table.add_row(
            str(prediction.id),
            prediction.status,
            prediction.kind,
            escape(prediction.question),
            _format_forecast(prediction),
            _format_result(prediction),
            _format_date(prediction.created_at),
            _format_optional_date(prediction.resolved_at),
        )

    console.print(table)


@app.command()
def resolve(
    prediction_id: Annotated[
        int | None,
        typer.Argument(
            help="Optional database ID to resolve directly. Omit for interactive mode.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Resolve a binary prediction as yes."),
    ] = False,
    no: Annotated[
        bool,
        typer.Option("--no", help="Resolve a binary prediction as no."),
    ] = False,
    actual: Annotated[
        float | None,
        typer.Option("--actual", help="Resolve a range prediction with this value."),
    ] = None,
) -> None:
    """Resolve one open prediction."""

    if prediction_id is not None:
        _resolve_direct(prediction_id, yes=yes, no=no, actual=actual)
        return

    if yes or no or actual is not None:
        _fail("Pass a prediction ID when using --yes, --no, or --actual.")

    _resolve_interactively()


@app.command()
def delete(
    prediction_id: Annotated[
        int,
        typer.Argument(help="Database ID of the prediction to delete."),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Allow deleting a resolved prediction.",
        ),
    ] = False,
) -> None:
    """Delete a prediction by ID."""

    try:
        deleted = storage.delete_prediction(prediction_id, force=force)
    except (storage.StorageError, ValueError) as error:
        _fail(str(error))

    console.print("[green]Deleted prediction[/green]")
    console.print(f"ID: {deleted.id}")
    console.print(f"Status: {deleted.status}")
    console.print(f"Type: {deleted.kind}")
    console.print(f"Question: {escape(deleted.question)}")


def _resolve_interactively() -> None:
    """Run the interactive resolution menu."""

    open_predictions = storage.list_open_predictions()
    if not open_predictions:
        console.print("No open predictions to resolve.")
        return

    _print_open_prediction_menu(open_predictions)
    selected_prediction = _choose_prediction(open_predictions)

    if isinstance(selected_prediction, BinaryPrediction):
        _resolve_binary_interactively(selected_prediction)
        return
    _resolve_range_interactively(selected_prediction)


@app.command()
def stats() -> None:
    """Show scoring and calibration summaries for resolved predictions."""

    prediction_stats = stats_module.summarize_predictions(
        storage.load_resolved_predictions()
    )
    _print_binary_stats(prediction_stats.binary)
    console.print()
    _print_range_stats(prediction_stats.range)


@app.command()
def where() -> None:
    """Show the local paths Predlog uses for data and generated plots."""

    console.print("[bold]Predlog paths[/bold]")
    console.print(f"Home: {_format_path(config.get_predlog_home())}")
    console.print(f"Database: {_format_path(config.get_database_path())}")
    console.print(f"Plots: {_format_path(config.get_plots_dir())}")
    console.print(f"Binary plot: {_format_path(config.get_binary_plot_path())}")
    console.print(f"Range plot: {_format_path(config.get_range_plot_path())}")


@plot_app.command("binary")
def plot_binary_command() -> None:
    """Generate the binary calibration plot."""

    from predlog import plots as plots_module

    try:
        saved_path = plots_module.plot_binary_calibration(
            storage.load_resolved_predictions()
        )
    except ValueError as error:
        _fail(str(error))

    console.print(f"Saved binary calibration plot: {_format_path(saved_path)}")


@plot_app.command("range")
def plot_range_command() -> None:
    """Generate the range diagnostics plot."""

    from predlog import plots as plots_module

    try:
        saved_path = plots_module.plot_range_diagnostics(
            storage.load_resolved_predictions()
        )
    except ValueError as error:
        _fail(str(error))

    console.print(f"Saved range diagnostics plot: {_format_path(saved_path)}")


def _resolve_direct(
    prediction_id: int,
    *,
    yes: bool,
    no: bool,
    actual: float | None,
) -> None:
    """Resolve one prediction directly from command options."""

    if yes and no:
        _fail("Use either --yes or --no, not both.")
    if actual is not None and (yes or no):
        _fail("Use --actual for range predictions or --yes/--no for binary predictions.")
    if yes or no:
        _resolve_binary_direct(prediction_id, 1 if yes else 0)
        return
    if actual is not None:
        _resolve_range_direct(prediction_id, actual)
        return
    _fail("Provide --yes, --no, or --actual when resolving by ID.")


def _resolve_binary_direct(prediction_id: int, outcome: int) -> None:
    """Resolve a binary prediction by permanent database ID."""

    try:
        resolved = storage.resolve_binary_prediction(prediction_id, outcome)
    except (storage.StorageError, ValueError) as error:
        _fail(str(error))
    _print_binary_resolution_feedback(resolved)


def _resolve_range_direct(prediction_id: int, actual: float) -> None:
    """Resolve a range prediction by permanent database ID."""

    try:
        resolved = storage.resolve_range_prediction(prediction_id, actual)
    except (storage.StorageError, ValueError) as error:
        _fail(str(error))
    _print_range_resolution_feedback(resolved)


def _resolve_binary_interactively(prediction: BinaryPrediction) -> None:
    """Prompt for a binary outcome, resolve it, and print scoring feedback."""

    outcome = _prompt_yes_no("Did it happen? yes/no")
    try:
        resolved = storage.resolve_binary_prediction(prediction.id, outcome)
    except (storage.StorageError, ValueError) as error:
        _fail(str(error))

    _print_binary_resolution_feedback(resolved)


def _print_binary_resolution_feedback(resolved: BinaryPrediction) -> None:
    """Print scoring feedback for a resolved binary prediction."""

    assert resolved.outcome is not None
    score = scoring.brier_score(resolved.probability, resolved.outcome)

    console.print(f"Resolved: {escape(resolved.question)}")
    console.print(f"Outcome: {_format_binary_outcome(resolved.outcome)}")
    console.print(f"Brier score: {score:.3f}")


def _resolve_range_interactively(prediction: RangePrediction) -> None:
    """Prompt for a numerical actual value, resolve it, and print feedback."""

    actual = _prompt_finite_float("Actual value")
    try:
        resolved = storage.resolve_range_prediction(prediction.id, actual)
    except (storage.StorageError, ValueError) as error:
        _fail(str(error))

    _print_range_resolution_feedback(resolved)


def _print_range_resolution_feedback(resolved: RangePrediction) -> None:
    """Print scoring feedback for a resolved range prediction."""

    assert resolved.actual is not None
    is_contained = scoring.contained(resolved.lower, resolved.upper, resolved.actual)
    score = scoring.winkler_score(
        resolved.lower,
        resolved.upper,
        resolved.confidence,
        resolved.actual,
    )

    console.print(f"Resolved: {escape(resolved.question)}")
    console.print(f"Actual value: {_format_number(resolved.actual)}")
    console.print(f"Contained in interval: {_format_bool(is_contained)}")
    console.print(f"Winkler score: {score:.3f}")


def _print_binary_stats(binary_stats: stats_module.BinaryStats) -> None:
    """Print terminal summaries for resolved binary predictions."""

    if binary_stats.resolved_count == 0:
        console.print("No resolved binary predictions yet.")
        return

    console.print("[bold]Binary predictions[/bold]")
    console.print(
        f"Number of resolved predictions: {binary_stats.resolved_count}"
    )
    console.print(
        f"Mean Brier score: {_format_optional_score(binary_stats.mean_brier_score)}"
    )
    if binary_stats.directional_hit_rate is None:
        console.print("Correct lean rate: n/a")
    else:
        console.print(
            "Correct lean rate: "
            f"{_format_rate(binary_stats.directional_hit_rate)}"
        )
    _print_binary_calibration_table(binary_stats)


def _print_range_stats(range_stats: stats_module.RangeStats) -> None:
    """Print terminal summaries for resolved range predictions."""

    if range_stats.resolved_count == 0:
        console.print("No resolved range predictions yet.")
        return

    console.print("[bold]Range predictions[/bold]")
    console.print(
        f"Number of resolved predictions: {range_stats.resolved_count}"
    )
    console.print(
        f"Mean Winkler score: {_format_optional_score(range_stats.mean_winkler_score)}"
    )
    console.print(f"Containment rate: {_format_optional_rate(range_stats.containment_rate)}")
    console.print()
    console.print("[bold]Range sharpness[/bold]")
    console.print(
        "Typical uncertainty: "
        f"{_format_optional_margin(range_stats.typical_uncertainty)}"
    )
    _print_range_calibration_table(range_stats)


def _print_binary_calibration_table(binary_stats: stats_module.BinaryStats) -> None:
    """Print bucket-level binary calibration details."""

    if not binary_stats.calibration_buckets:
        return

    table = Table(title="Binary calibration")
    table.add_column("Bucket", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Average predicted chance", justify="right")
    table.add_column("Actual event rate", justify="right")
    table.add_column("Calibration gap", justify="right")
    table.add_column("Evidence")
    table.add_column("Feedback")

    for bucket in binary_stats.calibration_buckets:
        table.add_row(
            _format_bucket_label(bucket.bucket),
            str(bucket.count),
            _format_table_rate(bucket.mean_probability),
            _format_table_rate(bucket.event_rate),
            _format_table_gap(bucket.calibration_gap),
            _format_bucket_evidence(bucket.count),
            bucket.feedback,
        )

    console.print()
    console.print(table)


def _print_range_calibration_table(range_stats: stats_module.RangeStats) -> None:
    """Print bucket-level range confidence calibration details."""

    if not range_stats.confidence_buckets:
        return

    table = Table(title="Range calibration")
    table.add_column("Confidence", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Inside range rate", justify="right")
    table.add_column("Calibration gap", justify="right")
    table.add_column("Evidence")
    table.add_column("Feedback")

    for bucket in range_stats.confidence_buckets:
        table.add_row(
            _format_table_rate(bucket.mean_confidence),
            str(bucket.count),
            _format_table_rate(bucket.containment_rate),
            _format_table_gap(bucket.calibration_gap),
            _format_bucket_evidence(bucket.count),
            bucket.feedback,
        )

    console.print()
    console.print(table)


def _print_open_prediction_menu(predictions: list[AnyPrediction]) -> None:
    """Print the temporary menu labels used by interactive resolution."""

    console.print("[bold]Open predictions[/bold]")
    console.print()
    for index, prediction in enumerate(predictions, start=1):
        console.print(f"[{index}] {escape(prediction.question)}")
        console.print(f"    Type: {prediction.kind}")
        if isinstance(prediction, BinaryPrediction):
            console.print(f"    Forecast: {_format_percent(prediction.probability)}")
        else:
            console.print(f"    Interval: {_format_interval(prediction.lower, prediction.upper)}")
            console.print(f"    Confidence: {_format_percent(prediction.confidence)}")
        console.print(f"    Created: {_format_date(prediction.created_at)}")
        console.print()


def _choose_prediction(predictions: list[AnyPrediction]) -> AnyPrediction:
    """Prompt for and return the selected prediction from an open menu."""

    selection = typer.prompt("Choose a prediction to resolve")
    try:
        index = int(selection)
    except ValueError:
        _fail(f"Choose a number from 1 to {len(predictions)}.")

    if not 1 <= index <= len(predictions):
        _fail(f"Choose a number from 1 to {len(predictions)}.")
    return predictions[index - 1]


def _prompt_yes_no(prompt: str) -> int:
    """Prompt until the user enters yes or no, returning 1 for yes and 0 for no."""

    while True:
        answer = typer.prompt(prompt).strip().lower()
        if answer in {"yes", "y"}:
            return 1
        if answer in {"no", "n"}:
            return 0
        console.print("[red]Please enter yes or no.[/red]")


def _prompt_finite_float(prompt: str) -> float:
    """Prompt until the user enters a finite number."""

    while True:
        value_text = typer.prompt(prompt)
        try:
            value = float(value_text)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")
            continue
        if math.isfinite(value):
            return value
        console.print("[red]Please enter a finite number.[/red]")


def _percentage_to_decimal(
    value: float,
    *,
    name: str,
    allow_zero: bool,
    allow_hundred: bool,
) -> float:
    """Validate a user-facing percentage and return its decimal value."""

    if not math.isfinite(value):
        _fail(f"{name} must be a finite percentage.")
    lower_ok = value >= 0 if allow_zero else value > 0
    upper_ok = value <= 100 if allow_hundred else value < 100
    if not lower_ok or not upper_ok:
        if allow_zero and allow_hundred:
            _fail(f"{name} must be between 0 and 100 percent.")
        _fail(f"{name} must be greater than 0 and less than 100 percent.")
    return value / 100


def _normalize_status_filter(status: str | None) -> PredictionStatus | None:
    """Normalize the optional list status argument."""

    if status is None:
        return None
    normalized = status.lower()
    if normalized == OPEN_STATUS:
        return OPEN_STATUS
    if normalized == RESOLVED_STATUS:
        return RESOLVED_STATUS
    _fail("Status must be 'open' or 'resolved'.")


def _empty_list_message(status: PredictionStatus | None) -> str:
    """Return the empty-state message for a list command."""

    if status == OPEN_STATUS:
        return "No open predictions."
    if status == RESOLVED_STATUS:
        return "No resolved predictions."
    return "No predictions yet."


def _list_title(status: PredictionStatus | None) -> str:
    """Return the table title for a list command."""

    if status == OPEN_STATUS:
        return "Open predictions"
    if status == RESOLVED_STATUS:
        return "Resolved predictions"
    return "Predictions"


def _format_forecast(prediction: AnyPrediction) -> str:
    """Return a compact forecast summary for list output."""

    if isinstance(prediction, BinaryPrediction):
        return _format_percent(prediction.probability)
    return (
        f"{_format_interval(prediction.lower, prediction.upper)} "
        f"@ {_format_percent(prediction.confidence)}"
    )


def _format_result(prediction: AnyPrediction) -> str:
    """Return a compact result summary for list output."""

    if isinstance(prediction, BinaryPrediction):
        if prediction.outcome is None:
            return "-"
        return _format_binary_outcome(prediction.outcome)
    if prediction.actual is None:
        return "-"
    return _format_number(prediction.actual)


def _format_binary_outcome(outcome: int) -> str:
    """Return yes/no display text for a stored binary outcome."""

    return "yes" if outcome == 1 else "no"


def _format_bool(value: bool) -> str:
    """Return yes/no display text for a boolean diagnostic."""

    return "yes" if value else "no"


def _format_percent(decimal_value: float) -> str:
    """Format a decimal probability or confidence as a percentage."""

    return f"{_format_number(decimal_value * 100)} percent"


def _format_interval(lower: float, upper: float) -> str:
    """Format an interval as ``[lower, upper]``."""

    return f"[{_format_number(lower)}, {_format_number(upper)}]"


def _format_number(value: float) -> str:
    """Format a number compactly for terminal output."""

    return f"{value:g}"


def _format_rate(value: float) -> str:
    """Format a decimal rate as a percentage with one decimal place."""

    return f"{value * 100:.1f} percent"


def _format_table_rate(value: float) -> str:
    """Format a decimal rate compactly for stats tables."""

    return f"{value * 100:.1f}%"


def _format_table_gap(value: float) -> str:
    """Format a signed calibration gap for stats tables."""

    return f"{value * 100:+.1f}%"


def _format_bucket_label(bucket: int) -> str:
    """Format a calibration bucket label."""

    return f"{bucket}%"


def _format_bucket_evidence(count: int) -> str:
    """Return the evidence label for a calibration bucket count."""

    if count < config.CALIBRATION_MIN_EVIDENCE_COUNT:
        return "sparse"
    return "enough"


def _format_optional_rate(value: float | None) -> str:
    """Format an optional decimal rate for terminal stats output."""

    if value is None:
        return "n/a"
    return _format_rate(value)


def _format_optional_margin(value: float | None) -> str:
    """Format an optional relative uncertainty margin."""

    if value is None:
        return "n/a"
    return f"+/- {value * 100:.1f}%"


def _format_optional_score(value: float | None) -> str:
    """Format an optional score for terminal stats output."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_date(value) -> str:
    """Format a timestamp as a date for compact terminal output."""

    return value.date().isoformat()


def _format_optional_date(value) -> str:
    """Format an optional timestamp for compact terminal output."""

    if value is None:
        return "-"
    return _format_date(value)


def _format_path(path: Path) -> str:
    """Return escaped path text for Rich table output."""

    return escape(str(path))


def _fail(message: str) -> None:
    """Print a friendly error message and exit the CLI command."""

    console.print(f"[red]Error:[/red] {escape(message)}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
