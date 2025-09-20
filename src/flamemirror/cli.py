"""Command line entrypoint for running the FlameMirror agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .autonomous_agent import AutonomousAgent
from .fuzzy import FuzzyGuidanceEngine
from .generator import CodeGenerator
from .gitmanager import GitManager
from .ml import CodexBackend, CrucibleTrainer, GraphformicCoder, Sandbox
from .testrunner import PytestRunner
from .validator import CodeValidator


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FlameMirror autonomous agent")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd() / "workspace",
        help="Workspace directory for generated code",
    )
    parser.add_argument(
        "--enable-ml",
        action="store_true",
        help="Enable GraphformicCoder Codex backend",
    )
    parser.add_argument(
        "--enable-fuzzy",
        action="store_true",
        help="Enable fuzzy guidance engine",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Allow git commits (default: dry run)",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=("-q",),
        help="Arguments forwarded to pytest",
    )
    return parser


def create_agent(args: argparse.Namespace) -> AutonomousAgent:
    workspace: Path = args.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    model = GraphformicCoder()
    if args.enable_ml:
        model.attach_backend(CodexBackend())

    generator = CodeGenerator(workspace=workspace, ml_model=model)
    validator = CodeValidator()
    testrunner = PytestRunner(
        workspace=workspace, pytest_args=_format_pytest_args(args.pytest_args)
    )
    git_manager = GitManager(workspace=workspace, dry_run=not args.no_dry_run)

    fuzzy_engine = None
    if args.enable_fuzzy:
        rules_path = Path(__file__).resolve().parent / "fuzzy" / "rules.yaml"
        fuzzy_engine = FuzzyGuidanceEngine(rules_path)

    trainer = CrucibleTrainer(
        training_dir=Path("training_problems"),
        checkpoint_dir=Path("checkpoints"),
        model=model,
        sandbox=Sandbox(),
    )

    return AutonomousAgent(
        workspace=workspace,
        generator=generator,
        validator=validator,
        testrunner=testrunner,
        git_manager=git_manager,
        fuzzy_engine=fuzzy_engine,
        trainer=trainer,
        enable_fuzzy=args.enable_fuzzy,
        enable_ml=args.enable_ml,
    )


def _format_pytest_args(args: Sequence[str]) -> Sequence[str]:
    return tuple(str(arg) for arg in args) or ("-q",)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    agent = create_agent(args)
    report = agent.run_cycle()
    print("Executed plan:")
    for step in report.plan:
        print(f"  - {step.description} (priority {step.priority})")
    if report.advice:
        print("Fuzzy guidance:")
        for item in report.advice:
            print(f"  * {item}")
    if report.failures:
        print("Failures detected:")
        for failure in report.failures:
            print(f"  ! {failure}")
    else:
        print("Cycle completed without failures.")


if __name__ == "__main__":
    main()
