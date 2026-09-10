"""Smoke tests for the initial project scaffold."""

from market_resonance import __version__


def test_package_version_is_defined() -> None:
    """The package exposes a version before research code is added."""
    assert __version__ == "0.1.0"
