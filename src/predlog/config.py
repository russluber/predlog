"""Configuration defaults and local filesystem paths for Predlog.

This module is intentionally small. It centralizes the paths and constants
that other Predlog modules need so that storage, plotting, and CLI code do not
each invent their own idea of where local data should live.

Predlog is local-first. By default it stores data below ``~/.predlog``. During
tests, demos, or development, callers can set the ``PREDLOG_HOME`` environment
variable to point Predlog at a different directory.
"""

from __future__ import annotations

import os
from pathlib import Path

PREDLOG_HOME_ENV = "PREDLOG_HOME"
"""Name of the environment variable that overrides Predlog's data directory."""

DEFAULT_HOME_DIRNAME = ".predlog"
"""Directory name used under the user's home directory by default."""

DATABASE_FILENAME = "predlog.db"
"""SQLite database filename used inside the Predlog data directory."""

PLOTS_DIRNAME = "plots"
"""Directory name used for generated plot files."""

BINARY_CALIBRATION_PLOT_FILENAME = "binary_calibration.png"
"""Default filename for the binary calibration plot."""

RANGE_CALIBRATION_SHARPNESS_PLOT_FILENAME = "range_calibration_sharpness.png"
"""Default filename for the range calibration and sharpness plot."""

BINARY_CALIBRATION_BUCKETS = tuple(range(10, 100, 10))
"""Default percentage buckets for binary calibration summaries and plots."""

RANGE_CONFIDENCE_BUCKETS = tuple(range(10, 100, 10))
"""Default percentage buckets for range confidence calibration summaries."""

CALIBRATION_MIN_EVIDENCE_COUNT = 5
"""Minimum resolved predictions for a calibration bucket to count as enough evidence."""


def get_predlog_home() -> Path:
    """Return the directory where Predlog should store local user data.

    If the ``PREDLOG_HOME`` environment variable is set, its value is used.
    Otherwise Predlog falls back to ``~/.predlog``.

    The returned path is expanded with :meth:`pathlib.Path.expanduser` so that
    values such as ``~/predlog-demo`` work consistently across operating
    systems. This function does not create the directory; callers that need the
    directory to exist should call :func:`ensure_directories`.
    """

    configured_home = os.environ.get(PREDLOG_HOME_ENV)
    if configured_home:
        return Path(configured_home).expanduser()
    return Path.home() / DEFAULT_HOME_DIRNAME


def get_database_path() -> Path:
    """Return the path to Predlog's SQLite database file.

    The database path is always the active Predlog home directory joined with
    ``predlog.db``. This function only computes the path; it does not create a
    database file or initialize a schema.
    """

    return get_predlog_home() / DATABASE_FILENAME


def get_plots_dir() -> Path:
    """Return the directory where Predlog should write generated plots.

    Plot files live in a dedicated ``plots`` directory below the active Predlog
    home directory. This function only computes the path; it does not create the
    directory.
    """

    return get_predlog_home() / PLOTS_DIRNAME


def get_binary_plot_path() -> Path:
    """Return the default output path for the binary calibration plot.

    The v0.1 plotting command is expected to overwrite this file whenever the
    user runs ``predlog plot binary``.
    """

    return get_plots_dir() / BINARY_CALIBRATION_PLOT_FILENAME


def get_range_plot_path() -> Path:
    """Return the default output path for the range calibration and sharpness plot.

    The v0.1 plotting command is expected to overwrite this file whenever the
    user runs ``predlog plot range``.
    """

    return get_plots_dir() / RANGE_CALIBRATION_SHARPNESS_PLOT_FILENAME


def ensure_directories() -> None:
    """Create Predlog's local data and plot directories if they do not exist.

    This helper is the small boundary where configuration code intentionally
    touches the filesystem. Storage and plotting code can call it before opening
    the database or saving image files.

    The function creates:

    * the active Predlog home directory
    * the active plots directory

    It does not create the SQLite database file. Database creation belongs to
    the storage module, which owns the schema.
    """

    get_predlog_home().mkdir(parents=True, exist_ok=True)
    get_plots_dir().mkdir(parents=True, exist_ok=True)
