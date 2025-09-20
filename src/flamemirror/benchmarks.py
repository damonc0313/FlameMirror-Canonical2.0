"""Benchmark tracking utilities for FlameMirror."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class BenchmarkMetric:
    """Represents an individual measurable quality target."""

    name: str
    achieved: float
    target: float
    unit: str = "score"
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def meets_target(self) -> bool:
        """Return ``True`` when the achieved value meets or exceeds the target."""

        return self.achieved >= self.target

    def gap(self) -> float:
        """Return the remaining delta to reach the configured target."""

        remaining = self.target - self.achieved
        return remaining if remaining > 0 else 0.0

    def to_dict(self) -> JsonDict:
        """Serialise the metric for persistence."""

        return {
            "name": self.name,
            "achieved": self.achieved,
            "target": self.target,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkMetric":
        """Rehydrate a metric from serialised JSON data."""

        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            metadata_mapping: MutableMapping[str, Any] = dict(metadata)
        else:  # pragma: no cover - defensive guard for corrupt payloads
            metadata_mapping = {}
        return cls(
            name=str(payload["name"]),
            achieved=float(payload["achieved"]),
            target=float(payload["target"]),
            unit=str(payload.get("unit", "score")),
            metadata=metadata_mapping,
        )


@dataclass(slots=True)
class BenchmarkReport:
    """Collection of metrics for a single benchmark execution."""

    benchmark: str
    metrics: List[BenchmarkMetric]
    notes: str = ""
    run_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def success_rate(self) -> float:
        """Return the proportion of metrics that hit their targets."""

        if not self.metrics:
            return 1.0
        successes = sum(1 for metric in self.metrics if metric.meets_target())
        return successes / len(self.metrics)

    def meets_all_targets(self) -> bool:
        """Return ``True`` if every metric hit its target."""

        return all(metric.meets_target() for metric in self.metrics)

    def failing_metrics(self) -> List[BenchmarkMetric]:
        """Return metrics that did not reach their target."""

        return [metric for metric in self.metrics if not metric.meets_target()]

    def to_dict(self) -> JsonDict:
        """Serialise the report for JSON persistence."""

        return {
            "benchmark": self.benchmark,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "notes": self.notes,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkReport":
        """Recreate a report from persisted data."""

        metrics_payload = payload.get("metrics", [])
        metrics: List[BenchmarkMetric] = []
        if isinstance(metrics_payload, Sequence):
            for item in metrics_payload:
                if isinstance(item, Mapping):
                    metrics.append(BenchmarkMetric.from_dict(item))
        timestamp_raw = payload.get("timestamp")
        if isinstance(timestamp_raw, str):
            timestamp = datetime.fromisoformat(timestamp_raw)
        else:
            timestamp = datetime.now(timezone.utc)
        return cls(
            benchmark=str(payload["benchmark"]),
            metrics=metrics,
            notes=str(payload.get("notes", "")),
            run_id=str(payload.get("run_id")) if payload.get("run_id") is not None else None,
            timestamp=timestamp,
        )


class BenchmarkRegistry:
    """Persistent registry storing benchmark reports for longitudinal tracking."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._reports: List[BenchmarkReport] = []
        self._load()

    def record(self, report: BenchmarkReport) -> None:
        """Append a new report and persist it to disk."""

        self._reports.append(report)
        self._persist()

    def history(self, benchmark: Optional[str] = None) -> List[BenchmarkReport]:
        """Return stored reports, optionally filtered by benchmark name."""

        if benchmark is None:
            return list(self._reports)
        return [report for report in self._reports if report.benchmark == benchmark]

    def best_report(self, benchmark: str) -> Optional[BenchmarkReport]:
        """Return the best run for the given benchmark based on success rate."""

        candidates = self.history(benchmark)
        if not candidates:
            return None
        return max(candidates, key=lambda report: (report.success_rate(), report.timestamp))

    def summary(self) -> str:
        """Return a human-readable overview of stored benchmarks."""

        if not self._reports:
            return "No benchmark runs recorded."
        lines = ["Benchmark Summary:"]
        for benchmark in sorted({report.benchmark for report in self._reports}):
            best = self.best_report(benchmark)
            if best is None:  # pragma: no cover - defensive guard
                continue
            status = "green" if best.meets_all_targets() else "yellow"
            rate = f"{best.success_rate():.0%}"
            lines.append(f"- {benchmark}: {rate} success ({status})")
        return "\n".join(lines)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        raw_text = self.storage_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return
        payload = json.loads(raw_text)
        if not isinstance(payload, Iterable):  # pragma: no cover - corrupt payload guard
            return
        reports: List[BenchmarkReport] = []
        for item in payload:
            if isinstance(item, Mapping):
                reports.append(BenchmarkReport.from_dict(item))
        self._reports = reports

    def _persist(self) -> None:
        serialised = [report.to_dict() for report in self._reports]
        self.storage_path.write_text(
            json.dumps(serialised, indent=2, sort_keys=True),
            encoding="utf-8",
        )

