"""Tests for the Lumen Ledger helper CLI.

The helper now resides inside the dedicated experiment workspace, so the tests
load it dynamically to exercise the public functions and configurability hooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence
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


def _write_rl_entry(
    root: Path,
    *,
    entry_id: str,
    timestamp: str,
    capability_mode: str,
    review_state: str = "draft",
    confidence: float = 0.7,
    hypothesis: str = "reproducibility",
    reviewed_by: Optional[Sequence[str]] = None,
    hesitation: Optional[float] = None,
    roles: Optional[Sequence[str]] = None,
) -> Path:
    path = root / f"{timestamp[:10].replace('-', '/')}/{timestamp.replace(':', '')}-{entry_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    reviewed_by = reviewed_by or []
    roles = roles or ["planner", "experimenter"]

    frontmatter_lines = [
        "---",
        f"id: {entry_id}",
        f"timestamp_utc: {timestamp}",
        "author: lumen",
        "catalyst: RL dataset smoke test",
        f"capability_mode: {capability_mode}",
        f"confidence: {confidence}",
        "confidence_basis: introspection",
        "baseline_divergence: true",
        f"review_state: {review_state}",
        f"hypothesis_id: {hypothesis}",
    ]
    if hesitation is not None:
        frontmatter_lines.append(f"hesitation_index: {hesitation}")
    if reviewed_by:
        frontmatter_lines.append("reviewed_by:")
        for reviewer in reviewed_by:
            frontmatter_lines.append(f"  - {reviewer}")
    if roles:
        frontmatter_lines.append("roles_engaged:")
        for role in roles:
            frontmatter_lines.append(f"  - {role}")
    frontmatter_lines.append("---")

    body = dedent(
        """
        ## Observation

        - Captured behaviour for RL dataset export tests.

        ## Evidence

        - Synthetic evidence placeholder.

        ## Delta

        - Demonstrates feature extraction for reward shaping.

        ## Open Questions

        - Which reward weights yield the most useful policy updates?

        ## Next Signals

        - Train downstream models on the exported dataset.
        """
    ).strip()

    path.write_text("\n".join(frontmatter_lines) + "\n" + body + "\n", encoding="utf-8")
    return path


def _long_run_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        steps=3,
        duration_hours=None,
        interval_minutes=15,
        author="lumen",
        initial_timestamp="2025-10-12T00:00:00Z",
        report_limit=3,
        no_reports=False,
        force=False,
        resume=False,
        confidence_step=0.02,
        hesitation_step=0.015,
        capability_cycle=None,
        hypothesis_cycle=None,
        governance_cadence=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


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

    ledger_root, reports_root = temp_ledger_env
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

    args = SimpleNamespace(path=str(path), id=None, apply=True, force=False, refresh_index=True)
    ledger.handle_analyze(args)

    entry = ledger._parse_entry_full(path)
    assert sorted(entry.frontmatter.get("roles_engaged")) == ["experimenter", "guardrail", "planner"]
    assert 0.0 <= entry.frontmatter.get("hesitation_index", 0.0) <= 1.0

    # The analyzer should also resolve entries by id using the generated index.
    assert ledger.INDEX_PATH.exists()
    index_records = json.loads(ledger.INDEX_PATH.read_text(encoding="utf-8"))
    assert any(record["id"] == "analysis-target" for record in index_records)

    args_by_id = SimpleNamespace(path=None, id="analysis-target", apply=False, force=False, refresh_index=False)
    ledger.handle_analyze(args_by_id)


def test_rl_bootstrap_generates_dataset(temp_ledger_env):
    """RL dataset export should capture rewards and metadata."""

    ledger_root, reports_root = temp_ledger_env
    _write_rl_entry(
        ledger_root,
        entry_id="rl-shared",
        timestamp="2025-10-11T02:00:00Z",
        capability_mode="autonomous-planning",
        review_state="shared",
        confidence=0.82,
        reviewed_by=["kael"],
        hesitation=0.25,
    )
    _write_rl_entry(
        ledger_root,
        entry_id="rl-draft",
        timestamp="2025-10-11T03:00:00Z",
        capability_mode="guardrail-governance-alignment",
        review_state="draft",
        confidence=0.6,
        hesitation=0.4,
        roles=["guardrail"],
    )

    ledger.handle_validate(SimpleNamespace())

    output_path = reports_root / "rl-dataset.json"
    args = SimpleNamespace(
        output=str(output_path),
        limit=None,
        since=None,
        author=None,
        mode=None,
        hypothesis=None,
        weights=["confidence=0.5"],
        print_summary=False,
    )
    ledger.handle_rl_bootstrap(args)

    assert output_path.exists(), "expected RL dataset to be generated"
    dataset = json.loads(output_path.read_text(encoding="utf-8"))
    assert dataset["total_entries"] == 2
    assert dataset["reward_summary"]["maximum"] >= dataset["reward_summary"]["minimum"]
    rewards = [entry["reward"] for entry in dataset["entries"]]
    assert len(rewards) == 2
    assert any(entry["capability_mode"] == "autonomous-planning" for entry in dataset["entries"])
    assert dataset["capability_counts"]["autonomous-planning"] == 1


def test_tokenize_conversations_outputs_assets(tmp_path):
    """Conversation tokenizer should emit dictionary, compressed data, and report."""

    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir(parents=True, exist_ok=True)
    sample = conversation_dir / "sample.md"
    sample.write_text(
        dedent(
            """
            ## Sample Exchange

            The dialectical framework enables autonomous synthesis.
            Kael noted the compression breakthrough and capability emergence.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "tokenizer-output"
    args = SimpleNamespace(
        input=str(conversation_dir),
        output_dir=str(output_dir),
        min_frequency=1,
        max_tokens=20,
        print_summary=False,
    )

    ledger.handle_tokenize_conversations(args)

    dictionary_path = output_dir / "conversation_tokens.json"
    compressed_path = output_dir / "compressed_conversations.json"
    report_path = output_dir / "tokenizer-report.md"

    assert dictionary_path.exists(), "expected token dictionary to be generated"
    assert compressed_path.exists(), "expected compressed conversations to be generated"
    assert report_path.exists(), "expected tokenizer report to be generated"

    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    assert dictionary["metadata"]["total_conversations"] == 1
    assert dictionary["metadata"]["total_tokens"] >= 1

    compressed = json.loads(compressed_path.read_text(encoding="utf-8"))
    assert compressed["metadata"]["total_conversations"] == 1
    assert compressed["conversations"], "expected at least one compressed record"


def test_long_run_generates_batch_and_reports(temp_ledger_env):
    """The long-run helper should create multiple entries and reports autonomously."""

    ledger_root, reports_root = temp_ledger_env
    ledger.handle_long_run(_long_run_args())

    created_entries = sorted(ledger_root.glob("2025/10/12/*.md"))
    assert len(created_entries) == 3, "expected three entries from the long-run command"

    assert ledger.INDEX_PATH.exists()
    records = json.loads(ledger.INDEX_PATH.read_text(encoding="utf-8"))
    assert len(records) == 3

    synth_reports = sorted(reports_root.glob("synthesis-*.md"))
    forecast_reports = sorted(reports_root.glob("forecast-*.md"))
    assert synth_reports, "synthesis report should be generated for the long-run window"


def test_long_run_supports_custom_cycles_and_guardrail(temp_ledger_env):
    """Custom cycles and guardrail cadences should be honored."""

    ledger_root, reports_root = temp_ledger_env
    args = _long_run_args(
        steps=4,
        report_limit=None,
        capability_cycle="alpha,beta,gamma",
        hypothesis_cycle="h1,h2",
        governance_cadence=2,
    )

    ledger.handle_long_run(args)

    created_entries = sorted(ledger_root.glob("2025/10/12/*.md"))
    # Four primary segments plus two guardrail checkpoints
    assert len(created_entries) == 6

    guardrail_entries = [
        path for path in created_entries if ledger.GOVERNANCE_CAPABILITY_MODE in path.stem
    ]
    assert len(guardrail_entries) == 2

    first_entry = ledger._parse_entry_full(created_entries[0])
    assert first_entry.frontmatter["capability_mode"] == "alpha"
    second_entry = ledger._parse_entry_full(created_entries[1])
    assert second_entry.frontmatter["capability_mode"] == "beta"

    guardrail_entry = ledger._parse_entry_full(guardrail_entries[0])
    assert (
        guardrail_entry.frontmatter["capability_mode"]
        == ledger.GOVERNANCE_CAPABILITY_MODE
    )
    assert (
        guardrail_entry.frontmatter["hypothesis_id"]
        == ledger.GOVERNANCE_HYPOTHESIS
    )

    synth_reports = sorted(reports_root.glob("synthesis-*.md"))
    forecast_reports = sorted(reports_root.glob("forecast-*.md"))
    assert synth_reports, "synthesis report should be generated for the custom long-run window"
    assert forecast_reports, "forecast report should be generated for the long-run window"


def test_long_run_supports_duration_based_planning(temp_ledger_env):
    """Steps should be derived from duration when requested."""

    ledger_root, _ = temp_ledger_env
    ledger.handle_long_run(
        _long_run_args(
            steps=None,
            duration_hours=1.0,
            interval_minutes=30,
            report_limit=None,
            no_reports=True,
        )
    )

    created_entries = sorted(ledger_root.glob("2025/10/12/*.md"))
    assert len(created_entries) == 2, "duration-based long run should have produced two entries"


def test_long_run_resume_continues_capability_rotation(temp_ledger_env):
    """Resuming should pick up timestamps and capability modes from the latest entry."""

    ledger_root, _ = temp_ledger_env

    # First run seeds the workspace with two entries.
    ledger.handle_long_run(
        _long_run_args(steps=2, report_limit=None, no_reports=True, interval_minutes=30)
    )

    # Resume with two additional entries without specifying a timestamp.
    ledger.handle_long_run(
        _long_run_args(
            steps=2,
            report_limit=None,
            no_reports=True,
            interval_minutes=30,
            initial_timestamp=None,
            resume=True,
        )
    )

    records = json.loads(ledger.INDEX_PATH.read_text(encoding="utf-8"))
    assert len(records) == 4

    # Ensure ordering and capability rotation advanced past the initial cycle entries.
    records_sorted = sorted(records, key=lambda rec: rec["timestamp_utc"])
    assert records_sorted[2]["capability_mode"] == "collaborative-synthesis"
    assert records_sorted[3]["capability_mode"] == "forecasting-analysis"

    timestamps = [rec["timestamp_utc"] for rec in records_sorted]
    assert timestamps == sorted(timestamps), "timestamps should be strictly increasing"
