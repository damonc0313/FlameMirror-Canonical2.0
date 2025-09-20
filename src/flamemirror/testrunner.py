"""Pytest integration used by the autonomous loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple


@dataclass(slots=True)
class TestOutcome:
    passed: bool
    exit_code: int
    output: str
    coverage: float = 0.0


class PytestRunner:
    """Thin wrapper around ``pytest.main`` with dependency injection for tests."""

    def __init__(
        self,
        workspace: Path,
        *,
        pytest_args: Optional[Sequence[str]] = None,
        executor: Optional[Callable[[Tuple[str, ...]], Tuple[int, str, Optional[float]]]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.pytest_args = tuple(pytest_args or ())
        self.executor = executor or self._run_pytest

    def run(self) -> TestOutcome:
        exit_code, output, coverage = self.executor(self.pytest_args)
        return TestOutcome(
            passed=exit_code == 0,
            exit_code=exit_code,
            output=output,
            coverage=coverage or 0.0,
        )

    def _run_pytest(self, args: Tuple[str, ...]) -> Tuple[int, str, Optional[float]]:
        import pytest

        exit_code = pytest.main(list(args))
        return exit_code, "", None
