import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from flamemirror import cli
from flamemirror.generator import CodeGenerator


def test_build_argument_parser_defaults():
    parser = cli.build_argument_parser()
    args = parser.parse_args([])
    assert isinstance(args.workspace, Path)
    assert args.pytest_args == ("-q",)


def test_create_agent_with_stubbed_trainer(monkeypatch, tmp_path):
    records = {}

    class StubTrainer:
        def __init__(self, training_dir, checkpoint_dir, model, sandbox):
            self.training_dir = Path(training_dir)
            self.checkpoint_dir = Path(checkpoint_dir)
            self.failed_tasks_path = self.training_dir / "failures.jsonl"
            records["model"] = model

        def log_failure(self, payload):  # pragma: no cover - integration invokes this
            records.setdefault("failures", []).append(payload)

    monkeypatch.setattr(cli, "CrucibleTrainer", StubTrainer)
    args = Namespace(
        workspace=tmp_path / "workspace",
        enable_ml=False,
        enable_fuzzy=False,
        no_dry_run=False,
        pytest_args=(),
    )
    agent = cli.create_agent(args)
    assert agent.workspace == args.workspace
    assert isinstance(agent.generator, CodeGenerator)
    assert records["model"] is agent.generator.ml_model


def test_cli_main(monkeypatch, tmp_path, capsys):
    class StubAgent:
        def run_cycle(self):
            return SimpleNamespace(plan=[], advice=[], failures=[])

    monkeypatch.setattr(cli, "create_agent", lambda args: StubAgent())
    monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
    cli.main()
    out = capsys.readouterr().out
    assert "Cycle completed without failures." in out


def test_create_agent_with_flags(monkeypatch, tmp_path):
    class StubTrainer:
        def __init__(self, training_dir, checkpoint_dir, model, sandbox):
            self.training_dir = Path(training_dir)
            self.checkpoint_dir = Path(checkpoint_dir)
            self.failed_tasks_path = self.training_dir / "failures.jsonl"

        def log_failure(self, payload):  # pragma: no cover
            pass

    monkeypatch.setattr(cli, "CrucibleTrainer", StubTrainer)
    args = Namespace(
        workspace=tmp_path / "workspace",
        enable_ml=True,
        enable_fuzzy=True,
        no_dry_run=True,
        pytest_args=(),
    )
    agent = cli.create_agent(args)
    assert agent.generator.ml_model is not None
    assert agent.fuzzy_engine is not None
    assert not agent.git_manager.dry_run
