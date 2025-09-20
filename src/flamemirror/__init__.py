"""FlameMirror autonomous agent package."""

from .autonomous_agent import AgentCycleReport, AutonomousAgent, PlanStep
from .benchmarks import BenchmarkMetric, BenchmarkRegistry, BenchmarkReport
from .generator import CodeGenerator, GenerationResult
from .gitmanager import CommitResult, GitManager
from .testrunner import PytestRunner, TestOutcome
from .validator import CodeValidator, ValidationResult

__all__ = [
    "AgentCycleReport",
    "AutonomousAgent",
    "BenchmarkMetric",
    "BenchmarkRegistry",
    "BenchmarkReport",
    "CodeGenerator",
    "CodeValidator",
    "CommitResult",
    "GenerationResult",
    "GitManager",
    "PlanStep",
    "PytestRunner",
    "TestOutcome",
    "ValidationResult",
]
