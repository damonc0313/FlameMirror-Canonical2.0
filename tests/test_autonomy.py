"""Tests for the autonomous orchestrator scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from codex.auto_orchestrator import (  # noqa: E402
    AutonomousDirective,
    AutonomousOrchestrator,
    AutonomousTask,
)
from codex_reflective_kernel import Evaluation, Genome, Hypothesis  # noqa: E402


class StubKernel:
    """Deterministic kernel stub used for orchestrator tests."""

    def __init__(self) -> None:
        self.calls = 0

    def run_cycle(self):  # type: ignore[override]
        self.calls += 1
        evaluation = Evaluation(rational=0.9, creative=0.91, ethical=0.95)
        hypothesis = Hypothesis(
            text=f"stub-synthesis-{self.calls}",
            evaluation=evaluation,
            constraint_summary="constraint stub",
            techniques=["paradox drift"],
        )
        genome = Genome(
            timestamp=float(self.calls),
            conceptual_kernel="stub",
            method_heuristics={"bias": 0.1},
            integrity_vector={"rational": 0.9, "creative": 0.91, "ethical": 0.95},
            text_digest=f"digest-{self.calls}",
            active_constraint="constraint stub",
            seed=42,
        )
        final = f"Final stub {self.calls}"
        meta = f"Meta stub {self.calls}"
        return hypothesis, genome, final, meta


@pytest.fixture()
def conversation_file(tmp_path: Path) -> Path:
    path = tmp_path / "conversation.txt"
    path.write_text(
        "User: Please evolve this kernel autonomously and document the lineage.\n"
        "Assistant: Acknowledged.\n"
        "User: Implement a new safety audit to avoid harm.\n"
        "User: Provide analysis of prior runs.\n",
        encoding="utf-8",
    )
    return path


def test_scan_directives(conversation_file: Path, tmp_path: Path) -> None:
    orchestrator = AutonomousOrchestrator(
        lambda: StubKernel(), conversation_file, tmp_path
    )
    directives = orchestrator.scan_directives()
    assert len(directives) == 4
    assert directives[0].speaker == "user"
    assert "autonomously" in directives[0].content


def test_task_derivation_prioritises_autonomy(conversation_file: Path, tmp_path: Path) -> None:
    orchestrator = AutonomousOrchestrator(
        lambda: StubKernel(), conversation_file, tmp_path
    )
    directives = orchestrator.scan_directives()
    tasks = orchestrator.derive_tasks(directives)
    assert tasks
    # Highest priority directive should be the autonomy request
    assert tasks[0].priority == 3
    assert "autonomy" in tasks[0].tags


def test_orchestrator_manifest(tmp_path: Path, conversation_file: Path) -> None:
    orchestrator = AutonomousOrchestrator(
        lambda: StubKernel(), conversation_file, tmp_path, run_solver=True, max_tasks=2
    )
    manifest = orchestrator.orchestrate()

    assert orchestrator.manifest_path.exists()
    data = json.loads(orchestrator.manifest_path.read_text(encoding="utf-8"))
    assert data["tasks"][0]["result"]["integrity"]["harmonic"] > 0.9
    # Manifest returned by orchestrate should match the file content
    assert data == manifest
    assert len(data["tasks"]) == 2
    assert data["tasks"][0]["result"]["genome"]["seed"] == 42



