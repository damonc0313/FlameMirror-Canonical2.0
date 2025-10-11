"""Pytest configuration helpers for FlameMirror tests."""
from __future__ import annotations

import warnings

try:  # pragma: no cover - runtime guard
    import pytest_cov.plugin  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - fallback path
    def pytest_addoption(parser):
        """Register noop coverage options when pytest-cov is unavailable."""

        parser.addoption(
            "--cov",
            action="append",
            default=[],
            metavar="MODULE",
            help=(
                "Coverage flag requested but pytest-cov is not installed; "
                "the option will be accepted and ignored."
            ),
        )
        parser.addoption(
            "--cov-report",
            action="append",
            default=[],
            metavar="TYPE",
            help=(
                "Coverage report flag requested but pytest-cov is not installed; "
                "the option will be accepted and ignored."
            ),
        )

    def pytest_configure(config):
        """Warn once when coverage options are ignored."""

        if config.getoption("--cov") or config.getoption("--cov-report"):
            warnings.warn(
                "pytest-cov is not installed; coverage options were ignored.",
                RuntimeWarning,
                stacklevel=1,
            )
