# Predlog

Predlog is a local-first command-line prediction journal for tracking personal probabilistic forecasts. It helps you log binary yes/no predictions and numerical range predictions, resolve them later, and see whether your confidence is becoming better calibrated over time.

The problem Predlog solves is that prediction skill is hard to improve from memory alone. By keeping forecasts, outcomes, scores, and calibration plots in one small SQLite-backed CLI, Predlog makes it easier to practice forecasting deliberately and notice where you are overconfident, underconfident, or improving.

## Usage

Predlog is managed with `uv`. To install it from GitHub into a `uv` project:

```bash
uv add "git+https://github.com/russluber/predlog.git"
```

If you are working from a local clone of this repository, run commands with:

```bash
uv run predlog --help
```

Predlog stores data locally in `~/.predlog` by default. To try it in a temporary demo directory instead:

```bash
PREDLOG_HOME=/tmp/predlog-demo uv run predlog where
```

### Log Predictions

Log a binary prediction when the outcome will eventually be yes or no:

```bash
uv run predlog binary "Will it rain in San Diego tomorrow?" --prob 30
```

This records a 30 percent forecast. Binary probabilities must be greater than 0 and less than 100.

Log a range prediction when the outcome will be a number:

```bash
uv run predlog range "What will my commute time be tomorrow, in minutes?" --low 20 --high 35 --conf 80
```

This records an 80 percent interval forecast from 20 to 35.

### List Predictions

Show every logged prediction:

```bash
uv run predlog list
```

Show only open or resolved predictions:

```bash
uv run predlog list open
uv run predlog list resolved
```

### Resolve Predictions

Resolve predictions interactively:

```bash
uv run predlog resolve
```

Or resolve directly by ID:

```bash
uv run predlog resolve 1 --yes
uv run predlog resolve 2 --no
uv run predlog resolve 3 --actual 28
```

Binary predictions use `--yes` or `--no`. Range predictions use `--actual` with the observed numerical value.

### View Stats

Show summary statistics for resolved predictions:

```bash
uv run predlog stats
```

For binary predictions, Predlog reports resolved count, mean Brier score, and directional hit rate. For range predictions, it reports resolved count, mean Winkler score, containment rate, and interval width summaries.

Stats also include non-empty calibration bucket tables. These show the bucket count, observed rate, and whether the bucket is still sparse or has enough evidence according to Predlog's 5-forecast threshold.

### Generate Plots

Create a binary calibration plot:

```bash
uv run predlog plot binary
```

Create range diagnostics:

```bash
uv run predlog plot range
```

Generated plots are saved under Predlog's plots directory. Use `where` to see the exact paths:

```bash
uv run predlog where
```

The binary calibration plot compares predicted probability buckets against actual event frequency. Hollow markers mean a bucket has fewer than 5 resolved forecasts, and filled markers mean it has at least 5.
