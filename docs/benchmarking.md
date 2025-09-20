# Benchmarking & Quality Gates

FlameMirror targets strict external benchmarks across capability, reliability,
and scale. This document explains how to record progress using the new
`flamemirror.benchmarks` module and how to plug the results into CI.

## Tracking Benchmark Runs

```python
from pathlib import Path

from flamemirror.benchmarks import BenchmarkMetric, BenchmarkRegistry, BenchmarkReport

registry = BenchmarkRegistry(Path(".flamemirror/benchmarks.json"))

report = BenchmarkReport(
    benchmark="capability",
    metrics=[
        BenchmarkMetric(name="SWE-bench Verified", achieved=0.51, target=0.70, unit="resolved"),
        BenchmarkMetric(name="BigCodeBench Complete", achieved=0.45, target=0.65, unit="pass@1"),
        BenchmarkMetric(name="BigCodeBench Instruct", achieved=0.48, target=0.65, unit="pass@1"),
        BenchmarkMetric(name="HumanEval", achieved=0.92, target=0.96, unit="pass@1"),
    ],
    notes="Weekly offline evaluation",
    run_id="2024-04-05",
)

registry.record(report)
print(registry.summary())
```

Persisted reports can be committed to the repository to provide an auditable
history. Nightly CI can append additional runs with higher targets as the agent
improves.

## Recommended Benchmarks

| Category | Benchmark | Target | Notes |
| --- | --- | --- | --- |
| Capability | SWE-bench Verified | ≥0.70 resolved | Use latest dataset release and GraphformicCoder backend |
| Capability | BigCodeBench Complete | ≥0.65 pass@1 | Evaluate both Complete and Instruct splits |
| Capability | HumanEval (or successor) | ≥0.96 pass@1 | Validate zero template leakage |
| Reliability | Ruff/Mypy/Bandit/Pytest Coverage | 100% pass | Enforced per-PR via CI |
| Reliability | Autonomy Uptime | ≥24h | Track median uninterrupted run |
| Security | SARIF reports | 0 high-severity findings | Exported from bandit and dependency scans |
| Scale | Docker build | 100% success | Build from lockfile to ensure hermeticity |

The registry allows capturing intermediate progress even when targets are not
yet hit, making regressions or plateaus visible.

## CI Integration Sketch

1. Run benchmark harnesses in scheduled GitHub Actions workflows.
2. Convert results into `BenchmarkMetric` instances.
3. Append to `.flamemirror/benchmarks.json` and upload as an artifact.
4. Fail the workflow if any mandatory metric regresses versus the stored best
   report.

See `.github/workflows/ci.yml` for the main validation pipeline; scheduled jobs
can import `flamemirror.benchmarks` to enforce long-term goals.
