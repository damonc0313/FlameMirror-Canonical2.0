"""Tests for the platform-compatible KF-UVE adapter."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.kfuve_adapter import AdaptedKFUVE


@pytest.fixture
def adapter() -> AdaptedKFUVE:
    return AdaptedKFUVE(consensus_threshold=0.75)


def test_ledger_chain(adapter: AdaptedKFUVE) -> None:
    payload_one = {"status": "passed"}
    entry_one = adapter.ledger_seal("static-analysis", payload_one)

    payload_two = {"status": "passed", "coverage": 1.0}
    entry_two = adapter.ledger_seal("tests", payload_two)

    serialised_one = json.dumps(payload_one, sort_keys=True)
    expected_first = AdaptedKFUVE._hash_data(f"genesis:static-analysis:{serialised_one}")
    assert entry_one.digest == expected_first
    serialised_two = json.dumps(payload_two, sort_keys=True)
    expected_second = AdaptedKFUVE._hash_data(
        f"{entry_one.digest}:tests:{serialised_two}"
    )
    assert entry_two.digest == expected_second
    assert entry_two.previous == entry_one.digest


def test_static_analysis_reports_metrics(adapter: AdaptedKFUVE) -> None:
    code = textwrap.dedent(
        '''
        """Example module."""

        def helper(value):
            return value * 2
        '''
    )
    report = adapter.static_analysis(code)
    assert report["status"] == "passed"
    assert report["metrics"]["function_count"] == 1
    assert report["metrics"]["has_docstring"] is True

    bad_code = "def broken(:\n    pass"
    failure = adapter.static_analysis(bad_code)
    assert failure["status"] == "failed"
    assert "SyntaxError" in failure["error"]


def test_security_audit_detects_banned_calls(adapter: AdaptedKFUVE) -> None:
    code = textwrap.dedent(
        """
        import os

        def launch(cmd):
            return os.system(cmd)
        """
    )
    report = adapter.security_audit(code)
    assert report["status"] == "failed"
    assert any(issue["call"] == "os.system" for issue in report["issues"])


def test_run_tests_success_and_failure(adapter: AdaptedKFUVE) -> None:
    good_code = textwrap.dedent(
        """
        def add(a, b):
            return a + b
        """
    )
    good_tests = "assert add(2, 3) == 5\nassert add(0, 0) == 0"
    success = adapter.run_tests(good_code, good_tests)
    assert success["status"] == "passed"
    assert success["coverage"] >= 0.5

    bad_code = textwrap.dedent(
        """
        def explode():
            return 1 / 0
        """
    )
    bad_tests = "explode()"
    failure = adapter.run_tests(bad_code, bad_tests)
    assert failure["status"] == "failed"
    assert "division" in failure["error"].lower()


def test_profile_performance_and_consensus(adapter: AdaptedKFUVE) -> None:
    counter = {"value": 0}

    def increment() -> None:
        counter["value"] += 1

    profile = adapter.profile_performance(increment, runs=5)
    assert profile["status"] == "profiled"
    assert profile["runs"] == 5

    consensus = adapter.simulate_consensus([0.9, 0.8, 0.7])
    assert consensus["status"] in {"passed", "pending"}
    assert consensus["average"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)

    invalid = adapter.simulate_consensus([1.2])
    assert invalid["status"] == "failed"


def test_generate_documentation_and_pipeline(adapter: AdaptedKFUVE) -> None:
    code = textwrap.dedent(
        '''
        """Docstring."""

        def greet(name):
            """Return a friendly greeting."""
            return f"Hello {name}!"
        '''
    )
    tests = "assert greet('Ada') == 'Hello Ada!'"

    documentation = adapter.generate_documentation(code)
    assert documentation["status"] == "generated"
    assert documentation["documentation"]["functions"][0]["name"] == "greet"

    namespace: dict = {}
    exec(code, namespace)

    reports = adapter.run_validation_pipeline(
        code=code,
        tests=tests,
        performance_callable=lambda: namespace["greet"]("World"),
        consensus_scores=(0.8, 0.9, 0.85),
    )

    assert reports["static_analysis"]["status"] == "passed"
    assert reports["tests"]["status"] == "passed"
    assert reports["consensus"]["approved"] is True
    assert reports["ledger_tail"]
    assert len(adapter.ledger) >= 6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
