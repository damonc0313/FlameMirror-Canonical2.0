from flamemirror.testrunner import PytestRunner


def test_testrunner_success(tmp_path):
    runner = PytestRunner(tmp_path, executor=lambda args: (0, "ok", 0.95))
    result = runner.run()
    assert result.passed
    assert result.coverage == 0.95


def test_testrunner_failure(tmp_path):
    runner = PytestRunner(tmp_path, executor=lambda args: (1, "boom", 0.5))
    result = runner.run()
    assert not result.passed
    assert result.output == "boom"


def test_testrunner_default_executor(monkeypatch, tmp_path):
    runner = PytestRunner(tmp_path)

    class FakeResult:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, cwd, text, capture_output):  # pragma: no cover - subprocess path
        assert command[0].endswith("python")
        assert cwd == tmp_path
        assert text and capture_output
        return FakeResult(5, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    exit_code, output, coverage = runner._run_pytest(("tests",))
    assert exit_code == 0
    assert "No tests collected" in output
    assert coverage is None
