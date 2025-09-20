"""Git management helpers for FlameMirror."""

from __future__ import annotations

import subprocess  # nosec B404

# Controlled usage restricted to git CLI invocations.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class CommitResult:
    committed: bool
    message: str
    dry_run: bool
    staged_files: List[Path] = field(default_factory=list)


class GitManager:
    """Coordinates git operations with optional dry-run mode."""

    def __init__(
        self,
        workspace: Path,
        *,
        dry_run: bool = True,
        runner: Optional[Callable[[Sequence[str]], subprocess.CompletedProcess[str]]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.runner = runner or self._run

    def stage_and_commit(self, files: Iterable[Path], message: str) -> CommitResult:
        staged = [Path(file) for file in files]
        if self.dry_run:
            return CommitResult(committed=True, message=message, dry_run=True, staged_files=staged)

        for file in staged:
            self.runner(("git", "add", str(file)))
        self.runner(("git", "commit", "-m", message))
        return CommitResult(committed=True, message=message, dry_run=False, staged_files=staged)

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if not command or command[0] != "git":
            raise ValueError("GitManager only permits git commands")
        return subprocess.run(  # nosec B603
            command,
            cwd=self.workspace,
            check=True,
            text=True,
            capture_output=True,
        )
