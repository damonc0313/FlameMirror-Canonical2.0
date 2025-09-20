from argparse import Namespace

from flamemirror import cli


def test_full_agent_cycle_creates_artifacts(tmp_path):
    workspace = tmp_path / "workspace"
    args = Namespace(
        workspace=workspace,
        enable_ml=False,
        enable_fuzzy=False,
        no_dry_run=False,
        pytest_args=("-q",),
    )
    agent = cli.create_agent(args)
    agent.testrunner.executor = lambda pytest_args: (0, "simulated", 0.91)

    report = agent.run_cycle()

    telemetry_path = workspace / "logs" / "telemetry.md"
    generated_module = workspace / "src" / "generated_module.py"
    generated_test = workspace / "tests" / "test_generated_module.py"

    assert telemetry_path.exists()
    assert generated_module.exists()
    assert generated_test.exists()
    assert not report.failures
    assert all(outcome.passed for outcome in report.tests)
    assert all(validation.passed for validation in report.validations)
    assert all(commit.dry_run for commit in report.commits)
    assert agent.metrics["coverage"] == 0.91


