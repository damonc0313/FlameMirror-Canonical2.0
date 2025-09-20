"""Tests for the benchmarking utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from flamemirror.benchmarks import BenchmarkMetric, BenchmarkRegistry, BenchmarkReport


def test_metric_gap_and_target_evaluation() -> None:
    metric = BenchmarkMetric(name="SWE-bench Verified", achieved=0.62, target=0.60, unit="resolved")
    assert metric.meets_target()
    assert metric.gap() == 0.0

    trailing_metric = BenchmarkMetric(name="BigCodeBench", achieved=0.44, target=0.70)
    assert not trailing_metric.meets_target()
    assert trailing_metric.gap() == pytest.approx(0.26)


def test_report_success_rate_and_serialisation_round_trip() -> None:
    metrics = [
        BenchmarkMetric(name="SWE-bench Verified", achieved=0.62, target=0.60),
        BenchmarkMetric(name="BigCodeBench Complete", achieved=0.50, target=0.65),
        BenchmarkMetric(name="HumanEval", achieved=0.97, target=0.96),
    ]
    report = BenchmarkReport(
        benchmark="capability",
        metrics=metrics,
        notes="Nightly evaluation",
        run_id="run-001",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert report.success_rate() == 2 / 3
    assert not report.meets_all_targets()
    assert [metric.name for metric in report.failing_metrics()] == ["BigCodeBench Complete"]

    payload = report.to_dict()
    restored = BenchmarkReport.from_dict(payload)
    assert restored.run_id == "run-001"
    assert restored.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert restored.success_rate() == report.success_rate()


def test_registry_persistence_and_best_report(tmp_path: Path) -> None:
    storage = tmp_path / "benchmarks.json"
    registry = BenchmarkRegistry(storage)

    report_one = BenchmarkReport(
        benchmark="capability",
        metrics=[
            BenchmarkMetric(name="SWE-bench Verified", achieved=0.50, target=0.70),
            BenchmarkMetric(name="BigCodeBench Complete", achieved=0.40, target=0.60),
        ],
        run_id="baseline",
    )
    report_two = BenchmarkReport(
        benchmark="capability",
        metrics=[
            BenchmarkMetric(name="SWE-bench Verified", achieved=0.61, target=0.70),
            BenchmarkMetric(name="BigCodeBench Complete", achieved=0.63, target=0.60),
        ],
        run_id="improved",
    )
    report_three = BenchmarkReport(
        benchmark="reliability",
        metrics=[BenchmarkMetric(name="Autonomy Uptime", achieved=12.0, target=24.0, unit="hours")],
        run_id="uptime-sprint",
    )

    registry.record(report_one)
    registry.record(report_two)
    registry.record(report_three)

    summary = registry.summary()
    assert "capability" in summary
    assert "reliability" in summary

    reloaded = BenchmarkRegistry(storage)
    capability_history = reloaded.history("capability")
    assert len(capability_history) == 2

    best = reloaded.best_report("capability")
    assert best is not None
    assert best.run_id == "improved"
    assert best.success_rate() == 0.5

