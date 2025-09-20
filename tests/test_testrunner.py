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

    def fake_pytest(args):  # pragma: no cover - exercised for branch coverage
        return 0

    monkeypatch.setattr("pytest.main", fake_pytest)
    exit_code, output, coverage = runner._run_pytest(("tests",))
    assert exit_code == 0
    assert output == ""
    assert coverage is None
