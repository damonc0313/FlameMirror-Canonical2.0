from pathlib import Path

from flamemirror.autonomous_agent import AutonomousAgent
from flamemirror.fuzzy.engine import FuzzyGuidanceEngine
from flamemirror.generator import CodeGenerator
from flamemirror.gitmanager import CommitResult, GitManager
from flamemirror.ml.model import GraphformicCoder
from flamemirror.ml.sandbox import Sandbox
from flamemirror.ml.trainer import CrucibleTrainer
from flamemirror.testrunner import PytestRunner
from flamemirror.validator import CodeValidator


def test_agent_cycle_success(tmp_path):
    workspace = tmp_path / "workspace"
    generator = CodeGenerator(workspace, ml_model=GraphformicCoder())
    validator = CodeValidator()
    runner = PytestRunner(workspace, executor=lambda args: (0, "ok", 0.92))
    git_manager = GitManager(workspace, dry_run=True)
    trainer = CrucibleTrainer(
        training_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / "checkpoints",
        model=GraphformicCoder(),
        sandbox=Sandbox(),
    )

    agent = AutonomousAgent(
        workspace=workspace,
        generator=generator,
        validator=validator,
        testrunner=runner,
        git_manager=git_manager,
        trainer=trainer,
    )

    report = agent.run_cycle(context=["Ensure docstring coverage"])
    assert report.failures == []
    assert len(report.generations) == len(report.plan)
    assert all(commit.committed for commit in report.commits)


def test_agent_cycle_logs_failures(tmp_path):
    workspace = tmp_path / "workspace"
    generator = CodeGenerator(workspace, ml_model=GraphformicCoder())
    validator = CodeValidator()
    runner = PytestRunner(workspace, executor=lambda args: (1, "boom", 0.4))
    git_manager = GitManager(workspace, dry_run=True)
    trainer = CrucibleTrainer(
        training_dir=tmp_path / "train_fail",
        checkpoint_dir=tmp_path / "checkpoints_fail",
        model=GraphformicCoder(),
        sandbox=Sandbox(),
    )

    agent = AutonomousAgent(
        workspace=workspace,
        generator=generator,
        validator=validator,
        testrunner=runner,
        git_manager=git_manager,
        trainer=trainer,
    )

    report = agent.run_cycle()
    assert report.failures
    log_path = trainer.failed_tasks_path
    assert log_path.exists()
    assert log_path.read_text().strip()


def test_agent_cycle_with_fuzzy_guidance(tmp_path):
    workspace = tmp_path / "workspace_fuzzy"
    generator = CodeGenerator(workspace, ml_model=GraphformicCoder())
    validator = CodeValidator()
    runner = PytestRunner(workspace, executor=lambda args: (0, "ok", 0.9))
    git_manager = GitManager(workspace, dry_run=True)
    rules_path = Path(__file__).resolve().parents[1] / "src/flamemirror/fuzzy/rules.yaml"
    fuzzy_engine = FuzzyGuidanceEngine(rules_path)
    trainer = CrucibleTrainer(
        tmp_path / "train_fuzzy",
        tmp_path / "checkpoints_fuzzy",
        GraphformicCoder(),
        Sandbox(),
    )

    agent = AutonomousAgent(
        workspace=workspace,
        generator=generator,
        validator=validator,
        testrunner=runner,
        git_manager=git_manager,
        fuzzy_engine=fuzzy_engine,
        trainer=trainer,
        enable_fuzzy=True,
    )

    report = agent.run_cycle()
    assert report.advice
    assert report.plan[0].description


def test_agent_cycle_commit_skipped(tmp_path):
    workspace = tmp_path / "workspace_commit"
    generator = CodeGenerator(workspace, ml_model=GraphformicCoder())
    validator = CodeValidator()
    runner = PytestRunner(workspace, executor=lambda args: (0, "ok", 0.95))

    class StubGitManager(GitManager):
        def stage_and_commit(self, files, message):  # type: ignore[override]
            return CommitResult(
                committed=False,
                message=message,
                dry_run=True,
                staged_files=list(files),
            )

    git_manager = StubGitManager(workspace)
    trainer = CrucibleTrainer(
        tmp_path / "train_commit",
        tmp_path / "checkpoints_commit",
        GraphformicCoder(),
        Sandbox(),
    )

    agent = AutonomousAgent(
        workspace=workspace,
        generator=generator,
        validator=validator,
        testrunner=runner,
        git_manager=git_manager,
        trainer=trainer,
    )

    report = agent.run_cycle()
    assert any("Commit skipped" in failure for failure in report.failures)
