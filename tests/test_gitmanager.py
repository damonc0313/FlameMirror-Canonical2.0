import subprocess
from pathlib import Path

from flamemirror.gitmanager import GitManager


def test_gitmanager_dry_run(tmp_path):
    manager = GitManager(tmp_path, dry_run=True)
    result = manager.stage_and_commit([Path("file.py")], "Test commit")
    assert result.dry_run
    assert result.committed
    assert result.staged_files == [Path("file.py")]


def test_gitmanager_executes_commands(tmp_path):
    calls = []

    def fake_runner(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    manager = GitManager(tmp_path, dry_run=False, runner=fake_runner)
    result = manager.stage_and_commit([Path("file.py")], "Real commit")
    assert not result.dry_run
    assert calls[0][0] == "git"
    assert any("commit" in part for part in calls[1])
