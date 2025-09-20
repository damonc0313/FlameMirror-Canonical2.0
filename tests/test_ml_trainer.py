import json

from flamemirror.ml.model import GraphformicCoder
from flamemirror.ml.sandbox import Sandbox
from flamemirror.ml.trainer import CrucibleTrainer


def test_crucible_trainer_runs_episode(tmp_path):
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    problem = {"name": "demo", "prompt": "value = 1"}
    (training_dir / "demo.json").write_text(json.dumps(problem), encoding="utf-8")
    trainer = CrucibleTrainer(training_dir, tmp_path / "checkpoints", GraphformicCoder(), Sandbox())
    result = trainer.run_episode()
    assert result.success
    checkpoint_dir = tmp_path / "checkpoints"
    assert any(checkpoint_dir.glob("*.json")) or any(training_dir.glob("failed_tasks.jsonl"))


def test_crucible_trainer_logs_failure(tmp_path):
    trainer = CrucibleTrainer(
        tmp_path / "train",
        tmp_path / "checkpoints",
        GraphformicCoder(),
        Sandbox(),
    )
    trainer.log_failure({"problem": "demo"})
    log = trainer.failed_tasks_path
    assert log.exists()
    entries = [json.loads(line) for line in log.read_text().strip().splitlines()]
    assert entries and entries[0]["problem"] == "demo"


def test_crucible_trainer_background_loop(monkeypatch, tmp_path):
    trainer = CrucibleTrainer(
        tmp_path / "train_bg",
        tmp_path / "checkpoints_bg",
        GraphformicCoder(),
        Sandbox(),
    )
    calls: list[object | None] = []
    original_run_episode = trainer.run_episode

    def fake_episode(problem=None):
        calls.append(problem)
        trainer._stop_event.set()
        return original_run_episode(problem)

    monkeypatch.setattr(trainer, "run_episode", fake_episode)
    trainer.start_background_loop(interval_seconds=0.01)
    trainer.stop_background_loop()
    assert calls


def test_crucible_trainer_records_episode_failure(tmp_path):
    class BrokenModel(GraphformicCoder):
        def generate(self, prompt: str, ast: str | None = None) -> str:  # type: ignore[override]
            return "raise RuntimeError('fail')\n"

    trainer = CrucibleTrainer(
        tmp_path / "train_fail",
        tmp_path / "checkpoints_fail",
        BrokenModel(),
        Sandbox(),
    )
    result = trainer.run_episode()
    assert not result.success
    assert trainer.failed_tasks_path.exists()
