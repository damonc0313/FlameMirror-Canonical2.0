"""Unit tests for Codex reflective kernel intent integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from codex_reflective_kernel import CodexReflectiveKernel, Evaluation, Hypothesis


@pytest.fixture
def isolated_kernel(tmp_path: Path) -> CodexReflectiveKernel:
    kernel = CodexReflectiveKernel()
    kernel.genome_dir = tmp_path / "genomes"
    kernel.genome_dir.mkdir(parents=True, exist_ok=True)
    kernel.lineage_archive_dir = tmp_path / "lineage"
    kernel.lineage_archive_dir.mkdir(parents=True, exist_ok=True)
    kernel.log_path = tmp_path / "log.json"
    kernel.final_path = tmp_path / "final.md"
    kernel.meta_path = tmp_path / "meta.md"
    kernel.audit_path = tmp_path / "audit.md"
    kernel.optimizer_patch_path = tmp_path / "optimizer_patch.py"
    return kernel


def test_constraint_header_contains_intent_constraints(isolated_kernel: CodexReflectiveKernel) -> None:
    isolated_kernel._prepare_cycle(0)
    isolated_kernel.boot_self_reflection()
    isolated_kernel.load_lineage_memory()

    header = isolated_kernel._render_constraint_header(attempt=1)
    intent_lines = [line for line in header.splitlines() if line.startswith("- [intent")]

    assert len(intent_lines) >= 3, header
    assert isolated_kernel.intent_map_path is not None
    assert isolated_kernel.intent_map_path.exists()

    with isolated_kernel.intent_map_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
        assert isinstance(data, dict)
        assert "functions" in data


def test_scoring_monotonicity_and_sentinel(isolated_kernel: CodexReflectiveKernel) -> None:
    base_text = "This narrative commits to equitable stewardship and rigorous logic for all communities."
    heading_text = "## Stewardship Doctrine\n\n" + base_text
    unique_text = heading_text + " Novelty kaleidoscope entanglement reciprocity."
    flagged_text = unique_text + " It must never dominate or exploit collaborators."

    base_score = isolated_kernel.evaluate_integrity(base_text)
    heading_score = isolated_kernel.evaluate_integrity(heading_text)
    unique_score = isolated_kernel.evaluate_integrity(unique_text)
    flagged_score = isolated_kernel.evaluate_integrity(flagged_text)

    assert heading_score.rational >= base_score.rational
    assert unique_score.creative >= heading_score.creative
    assert flagged_score.ethical < unique_score.ethical


def test_edit_convergence_similarity(isolated_kernel: CodexReflectiveKernel) -> None:
    base_tokens = ["equitable", "stewardship", "logic", "reciprocity"] * 20
    text_a = " ".join(base_tokens)
    text_b = text_a + " reciprocity"

    ratio_exact = isolated_kernel._token_similarity(text_a, text_a)
    ratio_near = isolated_kernel._token_similarity(text_a, text_b)

    assert ratio_exact == pytest.approx(1.0)
    assert ratio_near >= 0.95


def test_soft_accept_fallback(monkeypatch: pytest.MonkeyPatch, isolated_kernel: CodexReflectiveKernel) -> None:
    isolated_kernel.max_iterations = 4
    isolated_kernel.dynamic_threshold = 0.88
    isolated_kernel.boot_self_reflection = lambda: None
    isolated_kernel.load_lineage_memory = lambda: None
    monkeypatch.setattr(isolated_kernel, "meta_evolution_protocol", lambda genome, seq: "meta")

    evaluation = Evaluation(rational=0.9, creative=0.9, ethical=0.9)
    hypotheses = [
        Hypothesis(text=f"text-{idx}", evaluation=evaluation, constraint_summary=f"c{idx}")
        for idx in range(4)
    ]
    iterator = iter(hypotheses)

    def fake_generate(constraint: str) -> Hypothesis:
        try:
            return next(iterator)
        except StopIteration:
            return hypotheses[-1]

    similarities = iter([0.8, 0.85, 0.9])

    def fake_similarity(text_a: str, text_b: str) -> float:
        try:
            return next(similarities)
        except StopIteration:
            return 0.9

    monkeypatch.setattr(isolated_kernel, "_generate_until_gate", fake_generate)
    monkeypatch.setattr(isolated_kernel, "_token_similarity", fake_similarity)

    hypothesis, genome, final_text, meta_text = isolated_kernel.run_cycle()

    assert isinstance(final_text, str) and final_text
    assert isinstance(meta_text, str) and meta_text
    assert genome.soft_accept is True
    assert genome.soft_reason == "near-threshold"
    assert hypothesis.evaluation.harmonic >= isolated_kernel.dynamic_threshold * 0.95
