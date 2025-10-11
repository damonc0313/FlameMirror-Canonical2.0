"""Tests for the Lumen Ledger helper CLI.

The helper now resides inside the dedicated experiment workspace, so the tests
load it dynamically to exercise the public functions and configurability hooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

import importlib.util
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "lumen_ledger.py"
spec = importlib.util.spec_from_file_location("lumen_ledger", MODULE_PATH)
assert spec and spec.loader, "Unable to load lumen_ledger module specification"
ledger = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ledger
spec.loader.exec_module(ledger)


@pytest.fixture()
def temp_ledger_env(tmp_path):
    """Point the helper at a temporary ledger/report workspace."""

    original_root = ledger.LEDGER_ROOT
    original_index = ledger.INDEX_PATH
    original_reports = ledger.REPORTS_ROOT

    workspace_ledger = tmp_path / "ledger"
    workspace_reports = tmp_path / "reports"
    ledger.configure_paths(ledger_root=workspace_ledger, reports_root=workspace_reports)

    try:
        yield workspace_ledger, workspace_reports
    finally:
        ledger.configure_paths(ledger_root=original_root, reports_root=original_reports)
        ledger.INDEX_PATH = original_index


def _write_entry(root: Path, filename: str) -> Path:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    entry_text = dedent(
        """
        ---
        id: test-entry
        timestamp_utc: 2025-10-10T00:00:00Z
        author: lumen
        catalyst: Synthetic catalyst
        capability_mode: exploratory-analysis
        confidence: 0.6
        confidence_basis: introspection
        baseline_divergence: true
        review_state: draft
        hypothesis_id: reproducibility
        ---
        ## Observation
        - Generated entry for tests.

        ## Open Questions
        - How do synthetic entries behave?

        ## Next Signals
        - Capture additional evidence.
        """
    ).strip() + "\n"
    path.write_text(entry_text, encoding="utf-8")
    return path


def test_configure_paths_allows_custom_workspace(tmp_path):
    """The helper should accept alternate directories for tests."""

    custom_ledger = tmp_path / "custom-ledger"
    custom_reports = tmp_path / "custom-reports"
    ledger.configure_paths(ledger_root=custom_ledger, reports_root=custom_reports)

    assert ledger.LEDGER_ROOT == custom_ledger
    assert ledger.INDEX_PATH == custom_ledger / "index.json"
    assert ledger.REPORTS_ROOT == custom_reports

    # Restore defaults so other tests operate on the canonical layout.
    defaults_root = Path(__file__).resolve().parents[1]
    ledger.configure_paths(
        ledger_root=defaults_root / "ledger",
        reports_root=defaults_root / "reports",
    )


def test_validate_and_reports_generation(temp_ledger_env):
    """Validation and reporting should succeed against a temporary workspace."""

    ledger_root, reports_root = temp_ledger_env
    entry_path = _write_entry(ledger_root, "2025/10/10/20251010-test-entry.md")
    assert entry_path.exists()

    ledger.handle_validate(SimpleNamespace())
    assert ledger.INDEX_PATH.exists()

    with ledger.INDEX_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    assert records[0]["id"] == "test-entry"

    synth_args = SimpleNamespace(limit=5, since=None, author=None, mode=None, hypothesis=None)
    ledger.handle_synthesize(synth_args)
    synth_reports = sorted(reports_root.glob("synthesis-*.md"))
    assert synth_reports, "Expected a synthesis report to be generated"

    forecast_args = SimpleNamespace(limit=5, since=None, author=None, mode=None, hypothesis=None)
    ledger.handle_forecast(forecast_args)
    forecast_reports = sorted(reports_root.glob("forecast-*.md"))
    assert forecast_reports, "Expected a forecast report to be generated"


def test_analyze_command_updates_metadata(temp_ledger_env):
    """The analyze command should infer roles and hesitation metadata."""

    ledger_root, _ = temp_ledger_env
    path = ledger_root / "2025/10/11/20251011-analysis-target.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dedent(
        """
        ---
        id: analysis-target
        timestamp_utc: 2025-10-11T01:00:00Z
        author: lumen
        catalyst: Test analyze command
        capability_mode: impossible-prompting-self-application
        baseline_divergence: true
        review_state: draft
        ---

        ## Observation

        - The planner voice outlined the roadmap while the experimenter executed the prototype.
        - Guardrail monitoring noted "maybe we should double-check" before proceeding.

        ## Open Questions

        - How confident should we be in the hesitation readings?
        """
    ).strip() + "\n"
    path.write_text(body, encoding="utf-8")

    args = SimpleNamespace(path=str(path), apply=True, force=False)
    ledger.handle_analyze(args)

    entry = ledger._parse_entry_full(path)
    assert sorted(entry.frontmatter.get("roles_engaged")) == ["experimenter", "guardrail", "planner"]
    assert 0.0 <= entry.frontmatter.get("hesitation_index", 0.0) <= 1.0
