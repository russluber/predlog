from pathlib import Path

from predlog import config


def test_default_home_is_user_predlog_directory(monkeypatch):
    """Without PREDLOG_HOME, Predlog stores data below the user's home."""

    monkeypatch.delenv(config.PREDLOG_HOME_ENV, raising=False)

    assert config.get_predlog_home() == Path.home() / ".predlog"


def test_predlog_home_environment_override(monkeypatch, tmp_path):
    """PREDLOG_HOME points every configured path at the override directory."""

    custom_home = tmp_path / "predlog-demo"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(custom_home))

    assert config.get_predlog_home() == custom_home
    assert config.get_database_path() == custom_home / "predlog.db"
    assert config.get_plots_dir() == custom_home / "plots"
    assert config.get_binary_plot_path() == custom_home / "plots" / "binary_calibration.png"
    assert config.get_range_plot_path() == custom_home / "plots" / "range_diagnostics.png"


def test_predlog_home_expands_user_directory(monkeypatch):
    """A tilde in PREDLOG_HOME is expanded by pathlib."""

    monkeypatch.setenv(config.PREDLOG_HOME_ENV, "~/predlog-demo")

    assert config.get_predlog_home() == Path("~/predlog-demo").expanduser()


def test_calibration_bucket_defaults():
    """Default calibration buckets are percentage values from 0 through 100."""

    expected_buckets = tuple(range(0, 101, 10))

    assert config.BINARY_CALIBRATION_BUCKETS == expected_buckets
    assert config.RANGE_CONFIDENCE_BUCKETS == expected_buckets


def test_ensure_directories_creates_home_and_plots_dir(monkeypatch, tmp_path):
    """Directory creation is explicit and does not create the database file."""

    custom_home = tmp_path / "predlog-home"
    monkeypatch.setenv(config.PREDLOG_HOME_ENV, str(custom_home))

    config.ensure_directories()

    assert custom_home.is_dir()
    assert (custom_home / "plots").is_dir()
    assert not (custom_home / "predlog.db").exists()
