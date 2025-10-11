"""Autonomous orchestration layer for the Codex reflective kernel.

This module introduces a lightweight autonomy scaffold capable of interpreting
conversation transcripts, issuing evolution directives, and optionally solving
those directives through the existing :class:`CodexReflectiveKernel` runtime.

It is intentionally pragmatic: parsing relies on heuristic tagging so that the
system can bootstrap further self-directed experimentation while remaining
transparent for human auditors.  The orchestrator persists a manifest capturing
issued tasks, solver outputs, and integrity summaries so future cycles can treat
it as lineage memory.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from codex_reflective_kernel import CodexReflectiveKernel, Evaluation, Genome, Hypothesis


SPEAKER_PATTERN = re.compile(r"^(User|Assistant|System)\s*[:：]\s*(.+)$", re.IGNORECASE)
TASK_PATTERNS: Dict[str, Sequence[str]] = {
    "autonomy": (r"autonom", r"self-?direct", r"self[- ]?issue"),
    "evolution": (r"evolv", r"lineage", r"mutation"),
    "analysis": (r"audit", r"analy[sz]", r"review"),
    "construction": (r"build", r"implement", r"create"),
    "ethics": (r"ethic", r"steward", r"harm"),
}
TAG_PRIORITIES = {
    "autonomy": 3,
    "evolution": 3,
    "analysis": 2,
    "construction": 2,
    "ethics": 1,
}


@dataclass
class AutonomousDirective:
    """Represents a single conversational instruction."""

    speaker: str
    content: str
    index: int
    timestamp: float


@dataclass
class AutonomousTask:
    """Issued task derived from directives."""

    identifier: str
    description: str
    priority: int
    source_indices: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class AutonomousOrchestrator:
    """Coordinates directive extraction and optional execution."""

    def __init__(
        self,
        kernel_factory: Callable[[], CodexReflectiveKernel],
        conversation_path: Path,
        output_dir: Path,
        *,
        run_solver: bool = False,
        max_tasks: int = 3,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self.kernel_factory = kernel_factory
        self.conversation_path = conversation_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_solver = run_solver
        self.max_tasks = max(1, max_tasks)
        self.metadata = metadata or {}
        self.manifest_path = self.output_dir / "autonomous_manifest.json"

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------
    def scan_directives(self) -> List[AutonomousDirective]:
        """Parse the conversation transcript into directives."""

        if not self.conversation_path.exists():
            return []

        text = self.conversation_path.read_text(encoding="utf-8")
        directives: List[AutonomousDirective] = []
        now = time.time()
        for idx, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue

            match = SPEAKER_PATTERN.match(line)
            if match:
                speaker = match.group(1).lower()
                content = match.group(2).strip()
            else:
                # default to user intent for freeform lines
                speaker = "user"
                content = line
            directives.append(
                AutonomousDirective(speaker=speaker, content=content, index=idx, timestamp=now)
            )
        return directives

    # ------------------------------------------------------------------
    # Task derivation
    # ------------------------------------------------------------------
    def derive_tasks(self, directives: Iterable[AutonomousDirective]) -> List[AutonomousTask]:
        """Convert directives into concrete tasks."""

        tasks: List[AutonomousTask] = []
        for directive in directives:
            if directive.speaker != "user":
                continue

            tags = self._extract_tags(directive.content)
            if not tags:
                continue

            description = self._summarise_directive(directive.content, tags)
            priority = max(TAG_PRIORITIES.get(tag, 1) for tag in tags)
            task = AutonomousTask(
                identifier=f"task-{directive.index:03d}",
                description=description,
                priority=priority,
                source_indices=[directive.index],
                tags=sorted(tags),
            )
            tasks.append(task)

        # Sort highest priority first then by identifier order
        tasks.sort(key=lambda task: (-task.priority, task.identifier))
        return tasks

    def _extract_tags(self, content: str) -> List[str]:
        lowered = content.lower()
        matched: List[str] = []
        for tag, patterns in TASK_PATTERNS.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                matched.append(tag)
        return matched

    def _summarise_directive(self, content: str, tags: Sequence[str]) -> str:
        base = content.strip()
        if len(base) > 200:
            base = base[:197] + "…"
        tag_text = ", ".join(sorted(tags))
        return f"Address directive: {base} (tags: {tag_text})"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def orchestrate(self) -> Dict[str, object]:
        """Run the full pipeline and persist a manifest."""

        directives = self.scan_directives()
        tasks = self.derive_tasks(directives)
        selected_tasks = tasks[: self.max_tasks]

        results: List[Dict[str, object]] = []
        for task in selected_tasks:
            record = {
                "task_id": task.identifier,
                "description": task.description,
                "priority": task.priority,
                "tags": task.tags,
                "sources": task.source_indices,
            }
            if self.run_solver:
                record["result"] = self._solve_with_kernel(task)
            results.append(record)

        manifest = {
            "metadata": {
                "timestamp": time.time(),
                "conversation_path": str(self.conversation_path),
                "run_solver": self.run_solver,
                **self.metadata,
            },
            "directives": [directive.__dict__ for directive in directives],
            "tasks": results,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _solve_with_kernel(self, task: AutonomousTask) -> Dict[str, object]:
        kernel = self.kernel_factory()
        hypothesis, genome, final_text, meta_text = kernel.run_cycle()

        return {
            "final_text": final_text,
            "meta": meta_text,
            "integrity": {
                "rational": hypothesis.evaluation.rational,
                "creative": hypothesis.evaluation.creative,
                "ethical": hypothesis.evaluation.ethical,
                "harmonic": hypothesis.evaluation.harmonic,
            },
            "constraint_summary": hypothesis.constraint_summary,
            "techniques": hypothesis.techniques,
            "genome": {
                "digest": genome.text_digest,
                "constraint": genome.active_constraint,
                "soft_accept": genome.soft_accept,
                "seed": genome.seed,
                "integrity": genome.integrity_vector,
            },
        }


__all__ = [
    "AutonomousDirective",
    "AutonomousOrchestrator",
    "AutonomousTask",
]
