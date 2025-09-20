"""Crucible training harness for continuous learning."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model import GraphformicCoder
from .sandbox import Sandbox


@dataclass(slots=True)
class TrainingProblem:
    name: str
    prompt: str
    ast: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingResult:
    problem: TrainingProblem
    success: bool
    reward: float
    notes: str = ""


class CrucibleTrainer:
    """Coordinates reinforcement style self-improvement for FlameMirror."""

    def __init__(
        self,
        training_dir: Path,
        checkpoint_dir: Path,
        model: GraphformicCoder,
        sandbox: Sandbox,
    ) -> None:
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.sandbox = sandbox
        self.failed_tasks_path = self.training_dir / "failed_tasks.jsonl"
        self._problems: List[TrainingProblem] = self._load_problems()
        self._stop_event = threading.Event()
        self._background_thread: Optional[threading.Thread] = None

    def _load_problems(self) -> List[TrainingProblem]:
        problems: List[TrainingProblem] = []
        for path in sorted(self.training_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            problems.append(
                TrainingProblem(
                    name=data["name"],
                    prompt=data["prompt"],
                    ast=data.get("ast"),
                    metadata=data.get("metadata", {}),
                )
            )
        if not problems:
            problems.append(
                TrainingProblem(name="bootstrap", prompt="def solution():\n    return 0\n")
            )
        return problems

    def start_background_loop(self, interval_seconds: float = 60.0) -> None:
        if self._background_thread and self._background_thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop_event.is_set():
                self.run_episode()
                self._stop_event.wait(interval_seconds)

        self._stop_event.clear()
        self._background_thread = threading.Thread(
            target=_loop,
            name="crucible-trainer",
            daemon=True,
        )
        self._background_thread.start()

    def stop_background_loop(self) -> None:
        if not self._background_thread:
            return
        self._stop_event.set()
        self._background_thread.join(timeout=5)
        self._background_thread = None

    def run_episode(self, problem: Optional[TrainingProblem] = None) -> TrainingResult:
        problem = problem or self._problems[0]
        generated = self.model.generate(problem.prompt, problem.ast)
        sandbox_result = self.sandbox.execute(generated)
        success = sandbox_result.success
        reward = 1.0 if success else 0.0
        notes = sandbox_result.stderr
        if success:
            self.save_checkpoint(
                {"problem": problem.name, "timestamp": time.time(), "reward": reward}
            )
        else:
            self.log_failure({
                "problem": problem.name,
                "prompt": problem.prompt,
                "stderr": sandbox_result.stderr,
            })
        return TrainingResult(problem=problem, success=success, reward=reward, notes=notes)

    def log_failure(self, payload: Dict[str, Any]) -> None:
        record = {**payload, "timestamp": time.time()}
        with self.failed_tasks_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def save_checkpoint(self, payload: Dict[str, Any]) -> Path:
        timestamp = int(payload.get("timestamp", time.time()))
        checkpoint = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint
