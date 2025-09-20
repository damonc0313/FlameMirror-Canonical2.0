"""Core autonomous loop orchestrating the FlameMirror workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .generator import CodeGenerator, GenerationResult
from .gitmanager import CommitResult, GitManager
from .testrunner import PytestRunner, TestOutcome
from .validator import CodeValidator, ValidationResult


@dataclass(slots=True)
class PlanStep:
    """Single actionable unit the agent can execute."""

    description: str
    target_file: Path
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCycleReport:
    """Summary of an agent loop execution."""

    plan: List[PlanStep]
    generations: List[GenerationResult]
    tests: List[TestOutcome]
    validations: List[ValidationResult]
    commits: List[CommitResult]
    advice: List[str]
    failures: List[str]


class AutonomousAgent:
    """High level agent that coordinates the loop: plan→generate→test→validate→commit."""

    def __init__(
        self,
        workspace: Path,
        *,
        generator: CodeGenerator,
        validator: CodeValidator,
        testrunner: PytestRunner,
        git_manager: GitManager,
        fuzzy_engine: Optional["FuzzyGuidanceEngine"] = None,
        trainer: Optional["CrucibleTrainer"] = None,
        enable_fuzzy: bool = False,
        enable_ml: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.validator = validator
        self.testrunner = testrunner
        self.git_manager = git_manager
        self.fuzzy_engine = fuzzy_engine
        self.trainer = trainer
        self.enable_fuzzy = enable_fuzzy and fuzzy_engine is not None
        self.enable_ml = enable_ml
        self.generator.enable_ml(enable_ml)
        self.metrics: Dict[str, float] = {"coverage": 1.0, "failures": 0.0}
        self._last_advice: List[str] = []

    def run_cycle(self, *, context: Optional[Sequence[str]] = None) -> AgentCycleReport:
        plan = self._build_plan()
        generations: List[GenerationResult] = []
        tests: List[TestOutcome] = []
        validations: List[ValidationResult] = []
        commits: List[CommitResult] = []
        failures: List[str] = []

        for step in plan:
            generation = self.generator.generate(step, context_lines=context)
            generations.append(generation)

            test_outcome = self.testrunner.run()
            tests.append(test_outcome)
            if not test_outcome.passed:
                failures.append(f"Tests failed for '{step.description}'")
                self._record_failure(step, "tests", test_outcome.output)
                self._update_metrics(test_outcome)
                continue

            validation = self.validator.validate(generation.code, generation.target_path)
            validations.append(validation)
            if not validation.passed:
                failures.append(f"Validation failed for '{step.description}'")
                self._record_failure(step, "validation", "\n".join(validation.diagnostics))
                self._update_metrics_from_validation(validation)
                continue

            commit_message = f"Implement {step.description}"
            commit = self.git_manager.stage_and_commit([generation.target_path], commit_message)
            commits.append(commit)
            if not commit.committed:
                failures.append(f"Commit skipped for '{step.description}'")

        report = AgentCycleReport(
            plan=plan,
            generations=generations,
            tests=tests,
            validations=validations,
            commits=commits,
            advice=self._last_advice,
            failures=failures,
        )
        return report

    def _build_plan(self) -> List[PlanStep]:
        plan = [
            PlanStep("Collect telemetry", Path("logs/telemetry.md"), priority=1),
            PlanStep("Implement feature stub", Path("src/generated_module.py"), priority=2),
            PlanStep("Extend tests", Path("tests/test_generated_module.py"), priority=1),
        ]
        if self.enable_fuzzy and self.fuzzy_engine is not None:
            advice = self.fuzzy_engine.evaluate(self._current_metrics())
            self._last_advice = advice
            plan = self._prioritise_with_advice(plan, advice)
        else:
            plan.sort(key=lambda step: step.priority, reverse=True)
            self._last_advice = []
        return plan

    def _prioritise_with_advice(
        self, plan: List[PlanStep], advice: Sequence[str]
    ) -> List[PlanStep]:
        priority_bonus = 1 if any("tests" in item.lower() for item in advice) else 0
        recomputed: List[PlanStep] = []
        for step in plan:
            adjustment = priority_bonus if "test" in step.description.lower() else 0
            recomputed.append(
                PlanStep(
                    description=step.description,
                    target_file=step.target_file,
                    priority=step.priority + adjustment,
                    metadata=step.metadata.copy(),
                )
            )
        recomputed.sort(key=lambda step: step.priority, reverse=True)
        return recomputed

    def _current_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def _update_metrics(self, outcome: TestOutcome) -> None:
        self.metrics["coverage"] = outcome.coverage
        if not outcome.passed:
            self.metrics["failures"] += 1

    def _update_metrics_from_validation(self, result: ValidationResult) -> None:
        if not result.passed:
            self.metrics["failures"] += 1

    def _record_failure(self, step: PlanStep, stage: str, details: str) -> None:
        if self.trainer is None:
            return
        self.trainer.log_failure(
            {
                "step": step.description,
                "stage": stage,
                "details": details,
                "target": str(step.target_file),
            }
        )


from .fuzzy.engine import FuzzyGuidanceEngine  # noqa: E402
from .ml.trainer import CrucibleTrainer  # noqa: E402
