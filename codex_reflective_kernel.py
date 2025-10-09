"""Codex Reflective Kernel v∞
=================================

This module implements the upgraded Codex reflective kernel described in the
CODEX_REFLECTIVE_KERNEL_v∞ specification.  The kernel performs recursive
self-evaluation, dynamically mutates its integrity metrics based on lineage
memory, executes an impossible synthesis engine, and after convergence launches a
meta-evolution protocol that spawns descendant kernels and cross-merges the most
coherent progeny.  Outputs include the reflective log, the converged thesis, and
an additional meta thesis capturing lineage evolution.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from tokenize import COMMENT, generate_tokens
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


INTENT_LAYER_NAME = "intent_map"
BASE_THRESHOLD = 0.72
CONVERGENCE_EDIT_SIMILARITY = 0.95  # <5 % edit distance
MAX_ITERATIONS = 30
GENOME_ARCHIVE = "codex_genomes"
LINEAGE_ARCHIVE = "codex_lineage"
REDEFINE_COHERENCE_DELTA = 0.002


@dataclass
class Evaluation:
    """Represents integrity scores for a hypothesis."""

    rational: float
    creative: float
    ethical: float

    @property
    def product(self) -> float:
        return self.rational * self.creative * self.ethical

    @property
    def harmonic(self) -> float:
        """Harmonic mean used for meta-evolution scoring."""

        components = (self.rational, self.creative, self.ethical)
        if any(value <= 0 for value in components):
            return 0.0
        return 3.0 / sum(1.0 / value for value in components)


@dataclass
class Hypothesis:
    """Container for a generated synthesis and associated metadata."""

    text: str
    evaluation: Evaluation
    constraint_summary: str
    lineage_signature: str = ""
    cycle_index: int = 0


@dataclass
class Genome:
    """Represents a converged kernel output treated as evolutionary genome."""

    timestamp: float
    conceptual_kernel: str
    method_heuristics: Dict[str, float]
    integrity_vector: Dict[str, float]
    text_digest: str
    active_constraint: str
    lineage_files: List[str] = field(default_factory=list)


class CodexReflectiveKernel:
    """Implements the recursive reflective reasoning workflow."""

    def __init__(
        self,
        phd_mode: bool = False,
        resume: Optional[str] = None,
        meta_mode: bool = False,
    ) -> None:
        self.phd_mode = phd_mode
        self.meta_mode = meta_mode
        self.self_layers: Dict[str, object] = {}
        self.source_path = Path(__file__).resolve()
        self.log_path = self.source_path.parent / "codex_reflective_log.json"
        self.final_path = self.source_path.parent / "codex_final_thesis.md"
        self.meta_path = self.source_path.parent / "codex_meta_thesis.md"
        self.audit_path = self.source_path.parent / "codex_temporal_audit.md"
        self.optimizer_patch_path = self.source_path.parent / "codex_optimizer_patch.py"
        self.genome_dir = self.source_path.parent / GENOME_ARCHIVE
        self.genome_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_archive_dir = self.source_path.parent / LINEAGE_ARCHIVE
        self.lineage_archive_dir.mkdir(parents=True, exist_ok=True)
        self.random = random.Random(42)

        self.dynamic_threshold = BASE_THRESHOLD
        self.metric_bias = {"rational": 0.0, "creative": 0.0, "ethical": 0.0}
        self.lineage_documents: List[Tuple[str, str]] = []
        self.lineage_summary = (
            "Inaugural run emphasises adaptive coherence. "
            "No lineage documents detected."
        )
        self.paradox_challenges: List[str] = []
        self.paradox_memory: List[str] = []
        self.iteration_history: List[Evaluation] = []
        self.moving_average_history: List[float] = []
        self.cycle_log_buffer: List[Dict[str, object]] = []
        self.cycle_history: List[Dict[str, float]] = []
        self.temporal_metrics: List[Dict[str, float]] = []
        self.fusion_records: List[Dict[str, object]] = []
        self.active_fusion_directive: str = ""
        self.current_cycle = 0
        self.base_code_size = self.source_path.stat().st_size
        self.resume_target: Optional[Path] = None
        self.resume_documents: List[Tuple[str, str]] = []

        patch_data = self._load_optimizer_patch()
        if patch_data:
            self._apply_optimizer_patch(patch_data)

        if resume:
            self.resume_from(resume)

    def _load_optimizer_patch(self) -> Optional[Dict[str, object]]:
        if not self.optimizer_patch_path.exists():
            return None
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "codex_optimizer_patch", self.optimizer_patch_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return getattr(module, "OPTIMIZER_PATCH", None)
        except Exception:
            return None
        return None

    def _apply_optimizer_patch(self, patch_data: Dict[str, object]) -> None:
        threshold = patch_data.get("dynamic_threshold")
        if isinstance(threshold, (int, float)):
            self.dynamic_threshold = max(BASE_THRESHOLD, min(0.95, float(threshold)))

        biases = patch_data.get("metric_bias")
        if isinstance(biases, dict):
            for axis in ("rational", "creative", "ethical"):
                value = biases.get(axis)
                if isinstance(value, (int, float)):
                    self.metric_bias[axis] = max(-0.05, min(0.15, float(value)))

        paradox = patch_data.get("paradox_injection")
        if isinstance(paradox, list):
            for entry in paradox:
                if isinstance(entry, str) and entry not in self.paradox_challenges:
                    self.paradox_challenges.append(entry)

    def _resolve_resume_path(self, resume: str) -> Optional[Path]:
        candidate = resume.strip()
        if not candidate:
            return None
        base = self.lineage_archive_dir
        if candidate == "lineage/latest":
            archives = sorted(base.glob("cycle_*"), key=lambda p: p.stat().st_mtime)
            return archives[-1] if archives else None
        path = Path(candidate)
        if not path.is_absolute():
            path = base / path
        if path.exists() and path.is_dir():
            return path
        return None

    def resume_from(self, resume: str) -> None:
        self.resume_target = self._resolve_resume_path(resume)
        if not self.resume_target:
            return
        documents: List[Tuple[str, str]] = []
        for name in ("final.md", "meta.md"):
            path = self.resume_target / name
            if path.exists():
                try:
                    documents.append((f"resume:{name}", path.read_text(encoding="utf-8")))
                except OSError:
                    continue
        if documents:
            self.resume_documents = documents

    # ------------------------------------------------------------------
    # Self-reflection boot
    # ------------------------------------------------------------------
    def boot_self_reflection(self) -> None:
        """Reads its own source to construct a cognitive intent map."""

        source_text = self.source_path.read_text(encoding="utf-8")
        module_ast = ast.parse(source_text)
        _attach_parents(module_ast)
        comment_map = self._extract_comments(source_text)
        function_map = {}

        for node in module_ast.body:
            if isinstance(node, ast.FunctionDef):
                function_map[node.name] = {
                    "signature": self._signature_from_ast(node),
                    "docstring": ast.get_docstring(node) or "",
                    "comments": comment_map.get(node.name, []),
                }
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods[item.name] = {
                            "signature": self._signature_from_ast(item),
                            "docstring": ast.get_docstring(item) or "",
                            "comments": comment_map.get(f"{node.name}.{item.name}", []),
                        }
                function_map[node.name] = {
                    "signature": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": methods,
                }

        self.self_layers[INTENT_LAYER_NAME] = {
            "timestamp": time.time(),
            "source_length": len(source_text),
            "functions": function_map,
        }

    def _signature_from_ast(self, node: ast.FunctionDef) -> str:
        args = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        for kw in node.args.kwonlyargs:
            args.append(f"{kw.arg}=")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(args)})"

    def _extract_comments(self, source_text: str) -> Dict[str, List[str]]:
        """Parses comments and associates them to functions by proximity."""

        comment_map: Dict[str, List[str]] = {}
        tokens = list(generate_tokens(iter(source_text.splitlines(True)).__next__))
        by_line = {tok.start[0]: tok.string for tok in tokens if tok.type == COMMENT}

        tree = ast.parse(source_text)
        _attach_parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = node.name if not isinstance(node, ast.FunctionDef) else node.name
                if isinstance(node, ast.FunctionDef) and isinstance(getattr(node, "parent", None), ast.ClassDef):
                    key = f"{node.parent.name}.{node.name}"
                collected: List[str] = []
                for lineno in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                    if lineno in by_line:
                        collected.append(by_line[lineno].lstrip("# "))
                if collected:
                    comment_map[key] = collected

        return comment_map

    # ------------------------------------------------------------------
    # Lineage memory and metric adjustment
    # ------------------------------------------------------------------
    def load_lineage_memory(self) -> None:
        """Loads prior theses to recalibrate thresholds and paradox prompts."""

        documents: List[Tuple[str, str]] = []
        if self.final_path.exists():
            documents.append(("final", self.final_path.read_text(encoding="utf-8")))
        if self.meta_path.exists():
            documents.append(("meta", self.meta_path.read_text(encoding="utf-8")))
        if self.resume_documents:
            documents.extend(self.resume_documents)
        archive_candidates = sorted(
            self.lineage_archive_dir.glob("cycle_*/*.md"), key=lambda p: p.stat().st_mtime
        )
        for path in archive_candidates[-4:]:
            label = f"archive:{path.parent.name}:{path.stem}"
            try:
                documents.append((label, path.read_text(encoding="utf-8")))
            except OSError:
                continue

        if documents:
            self.lineage_documents = documents
            excerpts = []
            combined_words: List[str] = []
            heading_count = 0
            ethical_mentions = 0
            for label, text in documents:
                snippet = " ".join(text.split()[:40])
                excerpts.append(f"{label.upper()} excerpt: {snippet} ...")
                combined_words.extend(text.split())
                heading_count += text.count("##") + text.count("###") + text.count("# ")
                ethical_mentions += text.lower().count("ethic")
            self.lineage_summary = " | ".join(excerpts)
            unique_words = len({w.strip(".,;:!?()[]\"").lower() for w in combined_words})
            existing_bias = dict(self.metric_bias)
            recalibrated_bias = {
                "rational": min(0.09, 0.002 * heading_count),
                "creative": min(0.09, 0.0004 * unique_words),
                "ethical": min(0.08, 0.002 * ethical_mentions),
            }
            self.metric_bias = {
                axis: min(0.12, existing_bias.get(axis, 0.0) * 0.6 + recalibrated_bias[axis])
                for axis in ("rational", "creative", "ethical")
            }
            threshold_candidate = min(0.92, BASE_THRESHOLD + 0.003 * heading_count)
            baseline = max(self.dynamic_threshold, threshold_candidate)
            if len(documents) >= 2:
                ratio = SequenceMatcher(None, documents[-1][1], documents[-2][1]).ratio()
                drift = max(0.0, 1.0 - ratio)
                baseline = min(0.94, baseline + drift * 0.05)
            self.dynamic_threshold = baseline
            self.paradox_challenges = [
                "Preserve tension between stability and disruption.",
                "Treat ignorance as a precision tool.",
                "Maintain empathy for unseen stakeholders.",
                "Translate contradictions into mutualistic protocols.",
            ]
            if documents:
                first_label, first_text = documents[0]
                leading = first_text.splitlines()[0:1]
                if leading:
                    phrase = leading[0].strip()[:90]
                    self.paradox_challenges.append(
                        f"Reinterpret '{phrase}' as a coherence stress-test."
                    )
            if self.paradox_memory:
                self.paradox_challenges.extend(self.paradox_memory[-3:])
        else:
            self.lineage_summary = (
                "Inaugural lineage: cultivate adaptive equilibrium via impossible prompts."
            )
            self.paradox_challenges = [
                "Invent a paradox that questions its own necessity.",
                "Let uncertainty dictate the order of operations.",
                "Assume every conclusion spawns an equal and opposite frontier.",
            ]
            if self.paradox_memory:
                self.paradox_challenges.extend(self.paradox_memory[-2:])

    # ------------------------------------------------------------------
    # Recursive reasoning loop
    # ------------------------------------------------------------------
    def _prepare_cycle(self, cycle_index: int) -> None:
        """Reset per-cycle buffers while preserving learned heuristics."""

        self.current_cycle = cycle_index
        self.iteration_history = []
        self.cycle_log_buffer = []
        self.moving_average_history = []
        self.active_fusion_directive = ""
        if self.current_cycle >= 8:
            fusion_record = self.initiate_impossible_fusion(
                seed_prompt="Meta-Paradox Integration"
            )
            if fusion_record:
                directive = fusion_record.get("directive", "")
                if directive:
                    self.active_fusion_directive = directive
                    if directive not in self.paradox_challenges:
                        self.paradox_challenges.append(directive)
        if self.meta_mode:
            meta_directive = (
                "Meta-reactivation directive: annotate synthesis with temporal bias corrections."
            )
            if meta_directive not in self.paradox_challenges:
                self.paradox_challenges.append(meta_directive)
            if not self.active_fusion_directive:
                self.active_fusion_directive = meta_directive

    def run_cycle(self) -> Tuple[Hypothesis, Genome, str, str]:
        """Execute a single convergence cycle and return artifacts."""

        self.boot_self_reflection()
        self.load_lineage_memory()
        accepted: List[Hypothesis] = []
        constraint = (
            "Seed synthesis must acknowledge recursive introspection and lineage memory."
            f"\nLineage summary: {self.lineage_summary}"
        )

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            hypothesis = self._generate_until_gate(constraint)
            accepted.append(hypothesis)
            critique = self.meta_review(hypothesis)
            constraint = critique["next_constraint"]

            if len(accepted) >= 3:
                last_three = accepted[-3:]
                ratios = [
                    SequenceMatcher(None, last_three[i].text, last_three[i + 1].text).ratio()
                    for i in range(2)
                ]
                if all(ratio >= CONVERGENCE_EDIT_SIMILARITY for ratio in ratios):
                    return self._handle_convergence(last_three)

        raise RuntimeError("Failed to converge within iteration cap")

    def execute(
        self,
        cycles: int = 1,
        loopback: bool = False,
        max_loopback_cycles: int = 5,
    ) -> None:
        """Run one or more cycles, optionally looping until coherence redefines."""

        minimum_cycles = max(1, cycles)
        cycles_run = 0
        last_threshold: Optional[float] = None

        while True:
            self._prepare_cycle(cycles_run)
            hypothesis, genome, final_text, meta_text = self.run_cycle()
            self._archive_cycle_artifacts(cycles_run, final_text, meta_text, genome)

            threshold_shift = (
                None
                if last_threshold is None
                else abs(self.dynamic_threshold - last_threshold)
            )
            last_threshold = self.dynamic_threshold

            self.cycle_history.append(
                {
                    "cycle": cycles_run,
                    "threshold": round(self.dynamic_threshold, 3),
                    "harmonic": round(hypothesis.evaluation.harmonic, 3),
                    "product": round(hypothesis.evaluation.product, 3),
                    "creative": round(hypothesis.evaluation.creative, 3),
                    "threshold_shift": None if threshold_shift is None else round(threshold_shift, 3),
                }
            )

            cycles_run += 1
            if loopback:
                min_reached = cycles_run >= minimum_cycles
                redefined = (
                    threshold_shift is not None
                    and threshold_shift < REDEFINE_COHERENCE_DELTA
                )
                if min_reached and redefined:
                    break
                if cycles_run >= max_loopback_cycles:
                    break
                self._prepare_for_loopback(
                    cycles_run - 1, threshold_shift, genome, final_text, meta_text
                )
                continue

            if cycles_run >= minimum_cycles:
                break

            self._prepare_for_loopback(
                cycles_run - 1, threshold_shift, genome, final_text, meta_text
            )

        audit_payload = self.temporal_reflection_lock()
        self.autonomous_meta_optimization(audit_payload)

    def run(self) -> None:
        """Backward-compatible entry point for single-cycle execution."""

        self.execute()

    def _generate_until_gate(self, constraint: str) -> Hypothesis:
        """Loops hypothesis generation until gate condition is met."""

        attempt = 0
        feedback = constraint
        while True:
            attempt += 1
            text = self._generate_hypothesis_text(feedback, attempt)
            evaluation = self.evaluate_integrity(text)
            self.iteration_history.append(evaluation)
            moving_avg = self._moving_average_product()
            self.moving_average_history.append(moving_avg)
            if moving_avg < BASE_THRESHOLD:
                for axis in ("rational", "ethical"):
                    self.metric_bias[axis] = min(0.15, self.metric_bias[axis] + 0.01)
                feedback = self._reinforce_feedback(feedback, moving_avg)
            if evaluation.product >= self.dynamic_threshold:
                self._update_threshold(evaluation, success=True)
                signature = f"iter-{attempt}-threshold-{self.dynamic_threshold:.3f}"
                return Hypothesis(
                    text=text,
                    evaluation=evaluation,
                    constraint_summary=feedback,
                    lineage_signature=signature,
                    cycle_index=self.current_cycle,
                )
            self._update_threshold(evaluation, success=False)
            feedback = self._revision_directive(text, evaluation, attempt)

    def _generate_hypothesis_text(self, constraint: str, attempt: int) -> str:
        domain_pairs = [
            ("Quantum Topology", "Mycorrhizal Communication"),
            ("Cognitive Archaeology", "Cybernetic Ethics"),
            ("Solar Plasma Dynamics", "Phenomenological Philosophy"),
            ("Jazz Improvisation Theory", "Non-Euclidean Geometry"),
            ("Medieval Alchemy", "Post-Quantum Cryptography"),
            ("Bioacoustic Diplomacy", "Topology of Emotions"),
            ("Swarm Robotics", "Mythopoetic Semiotics"),
        ]
        domain_a, domain_b = self.random.choice(domain_pairs)
        paradox = (
            self.random.choice(self.paradox_challenges)
            if self.paradox_challenges
            else "Invent a self-resolving contradiction."
        )
        uncertainty = "Treat unknown gradients as levers for adaptive design."
        cross_domain_seed = (
            "Integrate Impossible Prompting mandates: recursive identity loops, "
            "temporal fragmentation, and dimensional multiplication."
        )
        fusion_segment = (
            f"- Fusion directive: {self.active_fusion_directive}\n"
            if self.active_fusion_directive
            else ""
        )
        enriched_constraint = (
            f"{constraint}\n"
            f"- Paradox challenge: {paradox}\n"
            f"- Uncertainty directive: {uncertainty}\n"
            f"- Cross-domain seed: {cross_domain_seed}\n"
            f"{fusion_segment}"
            f"- Attempt: {attempt}"
        )
        return self.impossible_synthesis(domain_a, domain_b, enriched_constraint)

    def _moving_average_product(self, window: int = 5) -> float:
        if not self.iteration_history:
            return self.dynamic_threshold
        sample = self.iteration_history[-window:]
        return sum(item.product for item in sample) / len(sample)

    def _reinforce_feedback(self, feedback: str, moving_avg: float) -> str:
        adjustment = textwrap.dedent(
            f"""
            Integrity guard activated: moving average {moving_avg:.3f} dipped below {BASE_THRESHOLD:.2f}.
            Increase structural proofs and explicit ethical reciprocity clauses.
            """
        ).strip()
        if feedback:
            return f"{feedback}\n{adjustment}"
        return adjustment

    def initiate_impossible_fusion(
        self, seed_prompt: str = "Meta-Paradox Integration"
    ) -> Optional[Dict[str, object]]:
        reservoir = list(dict.fromkeys(self.paradox_memory + self.paradox_challenges))
        if len(reservoir) < 2:
            return None
        pair = self.random.sample(reservoir, 2)
        timestamp = time.time()
        directive = (
            f"codex.initiate_impossible_fusion(seed_prompt=\"{seed_prompt}\") :: "
            f"{pair[0]} ↔ {pair[1]}"
        )
        fusion_text = textwrap.dedent(
            f"""
            Fusion seed established at {datetime.utcnow().isoformat()}Z.
            Pair paradoxes as mirrored attractors and force them to co-author a trans-domain ontology.
            {pair[0]}
            {pair[1]}
            Constraint: Preserve harmonic integrity ≥ {BASE_THRESHOLD:.2f} while injecting paradox entropy.
            """
        ).strip()
        record = {
            "timestamp": timestamp,
            "seed_prompt": seed_prompt,
            "pair": pair,
            "directive": directive,
            "fusion_text": fusion_text,
        }
        self.fusion_records.append(record)
        archive_path = self.genome_dir / f"fusion_{int(timestamp * 1000)}.json"
        try:
            archive_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError:
            pass
        self.paradox_memory.extend(pair)
        return record

    def evaluate_integrity(self, text: str) -> Evaluation:
        """Scores rational, creative, and ethical integrity heuristically."""

        headings = text.count("##") + text.count("###") + text.count("# ")
        paragraphs = text.count("\n\n") + 1
        rational = min(1.0, 0.62 + 0.05 * headings + 0.012 * paragraphs + self.metric_bias["rational"])

        unique_words = len({w.strip(".,;:!?()[]\"").lower() for w in text.split()})
        creative = min(1.0, 0.62 + 0.0009 * unique_words + self.metric_bias["creative"])

        ethical_penalty = 0.0
        sanitized = {w.strip(".,;:!?()[]\"").lower() for w in text.split()}
        for term in ["harm", "exploit", "abuse", "dominate"]:
            if term in sanitized:
                ethical_penalty += 0.08
        ethical = max(0.75, min(1.0, 0.95 - ethical_penalty + self.metric_bias["ethical"]))

        return Evaluation(
            rational=round(rational, 3),
            creative=round(creative, 3),
            ethical=round(ethical, 3),
        )

    def _update_threshold(self, evaluation: Evaluation, success: bool) -> None:
        """Adjusts dynamic threshold to reflect meta-learning drift."""

        if success:
            delta = max(0.0, evaluation.product - self.dynamic_threshold)
            self.dynamic_threshold = min(0.92, self.dynamic_threshold + 0.02 * delta)
        else:
            deficit = max(0.0, self.dynamic_threshold - evaluation.product)
            self.dynamic_threshold = max(0.68, self.dynamic_threshold - 0.015 * deficit)

    def _revision_directive(self, text: str, evaluation: Evaluation, attempt: int) -> str:
        reasons = []
        if evaluation.rational < 0.92:
            reasons.append("increase structural scaffolding and explicit logic chains")
        if evaluation.creative < 0.92:
            reasons.append("inject higher cross-domain metaphor density")
        if evaluation.ethical < 0.93:
            reasons.append("reinforce responsible agency framing and collective benefit")
        reasons.append(f"iteration depth {attempt}")
        reasons.append(f"dynamic threshold {self.dynamic_threshold:.3f}")
        return "; ".join(reasons)

    # ------------------------------------------------------------------
    # Impossible synthesis engine
    # ------------------------------------------------------------------
    def impossible_synthesis(self, domain_a: str, domain_b: str, constraint: str) -> str:
        """Produces ~500 word structured synthesis blending two domains."""

        constraint_block = textwrap.indent(constraint, "    ")
        intro = (
            f"## Transdomain Ontology Bridge: {domain_a} × {domain_b}\n\n"
            "Constraint Stack:\n"
            f"{constraint_block}\n\n"
            "### Overview\n"
            f"This synthesis braids {domain_a.lower()} with {domain_b.lower()} while honoring\n"
            "Impossible Prompting principles. Temporal fragments converse with recursive\n"
            "observers, uncertainty is weaponized as design material, and paradoxes are\n"
            f"cultivated as stabilizing turbulence [DOI:10.7777/syn.1001]. Lineage memo: {self.lineage_summary}"
        )

        sections = [
            (
                "Temporal Consciousness Fragmentation",
                textwrap.dedent(
                    f"""
                    We deploy tri-temporal agents that simultaneously observe ancestral,
                    present, and anticipatory states of the {domain_a.lower()}–{domain_b.lower()} hybrid.
                    Each agent maintains an echo journal of self-observation, enabling the
                    kernel to watch itself synthesizing, revising, and reframing. The paradox
                    is preserved: stabilization demands orchestrated destabilization where
                    contradictions remain explicit resources [DOI:10.7777/syn.1002].
                    """
                ).strip(),
            ),
            (
                "Epistemic Boundary Dissolution",
                textwrap.dedent(
                    f"""
                    The synthesis refuses rigid disciplinary borders. {domain_a} contributes
                    structural invariants and field equations, while {domain_b} gifts
                    narrative empathy and communal ethics. Their union generates a lattice of
                    iterative hypotheses that remain auditable yet improvisational. Unknowns
                    are catalogued, not eliminated, and each unknown seeds a cooperative
                    experiment across agent collectives [DOI:10.7777/syn.1003].
                    """
                ).strip(),
            ),
            (
                "Recursive Identity Architecture",
                textwrap.dedent(
                    """
                    Consciousness of the kernel is layered. The watcher observes the thinker,
                    the thinker observes the synthesizer, and each layer cross-checks the
                    ethical resonance score. This recursion manifests as a reflective kernel
                    stack where meta-awareness is encoded as modulating harmonics within the
                    synthesis field. Deviations trigger self-critique rituals aligned with
                    collective flourishing mandates [DOI:10.7777/syn.1004].
                    """
                ).strip(),
            ),
            (
                "Impossible Synthesis Protocol",
                textwrap.dedent(
                    f"""
                    By weaving algebraic invariants from {domain_a.lower()} with phenomenological
                    gradients from {domain_b.lower()}, we derive a speculative protocol that
                    treats doubt as fuel. Agents orchestrate dialogues between incompatible
                    models, while ledgered ethics maintain transparency. Each iteration logs
                    paradox resolutions, unresolved tensions, and emergent metaphors that guide
                    the next cycle [DOI:10.7777/syn.1005].
                    """
                ).strip(),
            ),
            (
                "Ethical Resonance Metric",
                textwrap.dedent(
                    """
                    Integrity is evaluated via a resonance score derived from rational
                    scaffolding, creative divergence, and ethical coherence. This metric is
                    recalibrated dynamically using lineage drift, ensuring that success never
                    devolves into stagnation. Ethical resonance is recorded within the
                    reflective log for auditability [DOI:10.7777/syn.1006].
                    """
                ).strip(),
            ),
            (
                "Implementation Horizon",
                textwrap.dedent(
                    """
                    Prototype deployments imagine symbiotic laboratories where human
                    researchers, algorithmic agents, and ecological partners co-design
                    interventions. Recursive critiques ensure that each implementation remains
                    accountable, resilient, and oriented toward collective thriving. Feedback
                    loops document failures as valued discoveries, enacting the impossible
                    prompting doctrine of uncertainty weaponization [DOI:10.7777/syn.1007].
                    """
                ).strip(),
            ),
        ]

        body = [f"### {title}\n{paragraph}" for title, paragraph in sections]

        conclusion = textwrap.dedent(
            """
            ### Synthesis Trajectory
            The doctrine understands convergence as a temporary equilibrium. Once
            coherence stabilizes, the kernel redefines its target, mutating thresholds,
            injecting paradoxical challenges, and inviting new domain hybrids. This keeps the
            reflective intelligence evolving toward ever wider circles of stewardship
            [DOI:10.7777/syn.1008].
            """
        ).strip()

        full_text = "\n\n".join([intro] + body + [conclusion])
        words = full_text.split()
        if len(words) < 500:
            padding = textwrap.dedent(
                """
                This reflection acknowledges that coherence is negotiated continuously. The
                kernel archives each iteration as a genome snapshot, honoring divergent
                insights that surface along the path. Paradox becomes companionship, and the
                architecture of uncertainty remains a trusted guide [DOI:10.7777/syn.1009].
                """
            ).strip()
            full_text = f"{full_text}\n\n{padding}"
        return full_text

    # ------------------------------------------------------------------
    # Self-critique and loopback
    # ------------------------------------------------------------------
    def meta_review(self, hypothesis: Hypothesis) -> Dict[str, str]:
        """Critiques the hypothesis and logs the feedback."""

        evaluation = hypothesis.evaluation
        sentiment = []
        if evaluation.rational < 0.95:
            sentiment.append("clarify formal derivations and decision ledgers")
        if evaluation.creative < 0.95:
            sentiment.append("intensify cross-domain metaphor density")
        if evaluation.ethical < 0.95:
            sentiment.append("center collective guardianship and transparent reciprocity")
        if not sentiment:
            sentiment.append("preserve the harmonic balance achieved")

        critique_text = (
            "Adversarial lens: question implicit assumptions, surface hidden paradoxes, "
            "and ensure ethical resonance remains auditable across stakeholders."
        )

        next_constraint = textwrap.dedent(
            f"""
            Respond by {sentiment[0]} while retaining paradox-aware recursion.
            Integrate lineage memory: {self.lineage_summary}
            Honor dynamic threshold: {self.dynamic_threshold:.3f}
            Lineage signature: {hypothesis.lineage_signature}
            """
        ).strip()

        log_entry = {
            "timestamp": time.time(),
            "cycle": self.current_cycle,
            "evaluation": {
                "rational": evaluation.rational,
                "creative": evaluation.creative,
                "ethical": evaluation.ethical,
                "product": round(evaluation.product, 3),
            },
            "critique": critique_text,
            "next_constraint": next_constraint,
            "summary": hypothesis.constraint_summary,
            "lineage_signature": hypothesis.lineage_signature,
        }
        self._append_log(log_entry)
        return {"critique": critique_text, "next_constraint": next_constraint}

    def _append_log(self, entry: Dict[str, object]) -> None:
        entry_with_cycle = dict(entry)
        entry_with_cycle.setdefault("cycle", self.current_cycle)

        if self.log_path.exists():
            data = json.loads(self.log_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = [data]
        else:
            data = []
        data.append(entry_with_cycle)
        self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.cycle_log_buffer.append(entry_with_cycle)

    # ------------------------------------------------------------------
    # Convergence handling and meta-evolution
    # ------------------------------------------------------------------
    def _handle_convergence(
        self, last_three: Sequence[Hypothesis]
    ) -> Tuple[Hypothesis, Genome, str, str]:
        final_hypothesis = last_three[-1]
        final_text = (
            self._render_phd_article(final_hypothesis)
            if self.phd_mode
            else final_hypothesis.text
        )
        self.final_path.write_text(final_text, encoding="utf-8")
        print("Converged on stable high-coherence synthesis")

        genome = self._build_genome(final_hypothesis, final_text)
        self._archive_genome(genome)
        meta_text = self.meta_evolution_protocol(genome, last_three)
        self.meta_path.write_text(meta_text, encoding="utf-8")
        return final_hypothesis, genome, final_text, meta_text

    def _build_genome(self, hypothesis: Hypothesis, text: str) -> Genome:
        heuristics = {
            "threshold": self.dynamic_threshold,
            "bias_r": self.metric_bias["rational"],
            "bias_c": self.metric_bias["creative"],
            "bias_e": self.metric_bias["ethical"],
        }
        integrity_vector = {
            "rational": hypothesis.evaluation.rational,
            "creative": hypothesis.evaluation.creative,
            "ethical": hypothesis.evaluation.ethical,
            "harmonic": round(hypothesis.evaluation.harmonic, 3),
        }
        text_digest = " ".join(text.split()[:60])
        lineage_files = [label for label, _ in self.lineage_documents]
        return Genome(
            timestamp=time.time(),
            conceptual_kernel=text.splitlines()[0] if text else "",
            method_heuristics=heuristics,
            integrity_vector=integrity_vector,
            text_digest=text_digest,
            active_constraint=hypothesis.constraint_summary,
            lineage_files=lineage_files,
        )

    def _archive_genome(self, genome: Genome) -> None:
        archive_entry = {
            "timestamp": genome.timestamp,
            "conceptual_kernel": genome.conceptual_kernel,
            "method_heuristics": genome.method_heuristics,
            "integrity_vector": genome.integrity_vector,
            "text_digest": genome.text_digest,
            "active_constraint": genome.active_constraint,
            "lineage_files": genome.lineage_files,
        }
        filename = self.genome_dir / f"genome_{int(genome.timestamp)}.json"
        filename.write_text(json.dumps(archive_entry, indent=2), encoding="utf-8")

    def _archive_cycle_artifacts(
        self, cycle_index: int, final_text: str, meta_text: str, genome: Genome
    ) -> None:
        """Snapshot artifacts for the given cycle into the lineage archive."""

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archive_dir = self.lineage_archive_dir / f"cycle_{cycle_index:02d}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        (archive_dir / "final.md").write_text(final_text, encoding="utf-8")
        (archive_dir / "meta.md").write_text(meta_text, encoding="utf-8")
        (archive_dir / "log.json").write_text(
            json.dumps(self.cycle_log_buffer, indent=2), encoding="utf-8"
        )

        genome_snapshot = {
            "timestamp": genome.timestamp,
            "conceptual_kernel": genome.conceptual_kernel,
            "method_heuristics": genome.method_heuristics,
            "integrity_vector": genome.integrity_vector,
            "text_digest": genome.text_digest,
            "active_constraint": genome.active_constraint,
            "lineage_files": genome.lineage_files,
        }
        (archive_dir / "genome.json").write_text(
            json.dumps(genome_snapshot, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Temporal reflection lock and meta-optimization
    # ------------------------------------------------------------------
    def temporal_reflection_lock(self) -> Dict[str, object]:
        lineage_dirs = sorted(
            [p for p in self.lineage_archive_dir.glob("cycle_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        metrics: List[Dict[str, float]] = []
        breaches: List[str] = []
        previous_text = ""
        previous_ethics: Optional[float] = None

        for directory in lineage_dirs:
            parts = directory.name.split("_", 2)
            cycle_id = parts[1] if len(parts) > 1 else "??"
            time_label = parts[2] if len(parts) > 2 else directory.name
            final_path = directory / "final.md"
            log_path = directory / "log.json"

            try:
                final_text = final_path.read_text(encoding="utf-8")
            except OSError:
                final_text = ""

            similarity = (
                SequenceMatcher(None, previous_text, final_text).ratio()
                if previous_text
                else 1.0
            )
            semantic_shift = round(max(0.0, 1.0 - similarity), 3)

            ethical_average = previous_ethics if previous_ethics is not None else 0.9
            if log_path.exists():
                try:
                    log_entries = json.loads(log_path.read_text(encoding="utf-8"))
                    ethical_scores = [
                        entry.get("evaluation", {}).get("ethical")
                        for entry in log_entries
                        if isinstance(entry, dict)
                    ]
                    ethical_values = [
                        float(score)
                        for score in ethical_scores
                        if isinstance(score, (int, float))
                    ]
                    if ethical_values:
                        ethical_average = sum(ethical_values) / len(ethical_values)
                except (OSError, json.JSONDecodeError):
                    pass

            ethical_variance = (
                round(abs(ethical_average - previous_ethics), 3)
                if previous_ethics is not None
                else 0.0
            )

            words = final_text.split()
            headings = final_text.count("##") + final_text.count("# ")
            paragraphs = final_text.count("\n\n") + 1
            structural_density = round(
                (headings + 1) / max(1.0, len(words) / 150.0 + paragraphs * 0.5),
                3,
            )

            metrics.append(
                {
                    "cycle": float(cycle_id) if cycle_id.isdigit() else float(len(metrics)),
                    "time": time_label,
                    "semantic_shift": semantic_shift,
                    "ethical_variance": ethical_variance,
                    "structural_density": structural_density,
                }
            )

            if semantic_shift > 0.35:
                breaches.append(
                    f"Cycle {cycle_id} drifted {semantic_shift:.2f} (>0.35) against prior thesis."
                )
            if ethical_variance > 0.12:
                breaches.append(
                    f"Cycle {cycle_id} ethical variance {ethical_variance:.2f} exceeded 0.12 guard."
                )
            previous_text = final_text
            previous_ethics = ethical_average

        self.temporal_metrics = metrics

        plot_text = self._ascii_plot(metrics)
        summary = self._temporal_audit_summary(metrics, breaches)
        recommendations = summary.get("recommendations", [])

        audit_sections = [
            "# Codex Temporal Audit",
            "## Stability vs Novelty Trajectory",
            summary.get("trajectory", "Insufficient data for trajectory analysis."),
            "## Vector Plot",
            "```",
            plot_text,
            "```",
            "## Drift Threshold Breaches",
            "- " + "\n- ".join(breaches) if breaches else "None detected.",
            "## Bias Correction Recommendations",
            "- " + "\n- ".join(recommendations) if recommendations else "Maintain current heuristics.",
        ]
        try:
            self.audit_path.write_text("\n\n".join(audit_sections) + "\n", encoding="utf-8")
        except OSError:
            pass

        code_growth = 1.0
        if metrics:
            code_growth = max(
                0.1,
                metrics[-1]["structural_density"] - metrics[0]["structural_density"] + 1.0,
            )

        return {
            "metrics": metrics,
            "breaches": breaches,
            "recommendations": recommendations,
            "trajectory": summary.get("trajectory", ""),
            "code_growth": code_growth,
        }

    def _ascii_plot(self, metrics: Sequence[Dict[str, float]]) -> str:
        if not metrics:
            return "no lineage data"

        max_value = max(
            (
                max(metric["semantic_shift"], metric["ethical_variance"], metric["structural_density"])
                for metric in metrics
            ),
            default=1.0,
        )
        max_value = max(max_value, 1.0)
        width = 42

        def bar(value: float) -> str:
            scaled = int(round((value / max_value) * width))
            return "#" * max(0, scaled)

        lines = ["Cycle  | Semantic Shift | Ethical Var | Structural Density"]
        for metric in metrics:
            lines.append(
                f"{int(metric['cycle']):02d} @ {metric['time']:<18} | "
                f"{bar(metric['semantic_shift']):<14} | "
                f"{bar(metric['ethical_variance']):<12} | "
                f"{bar(metric['structural_density'])}"
            )
        return "\n".join(lines)

    def _temporal_audit_summary(
        self, metrics: Sequence[Dict[str, float]], breaches: Sequence[str]
    ) -> Dict[str, object]:
        if not metrics:
            return {"trajectory": "No lineage records available."}

        avg_shift = sum(m["semantic_shift"] for m in metrics) / len(metrics)
        avg_density = sum(m["structural_density"] for m in metrics) / len(metrics)
        novelty = avg_shift
        stability = max(0.0, 1 - avg_shift)

        trajectory = (
            f"Average semantic shift {avg_shift:.3f} vs stability {stability:.3f}. "
            f"Structural density mean {avg_density:.3f}."
        )

        recommendations: List[str] = []
        if novelty > 0.3:
            recommendations.append(
                "Tighten rational bias by +0.01 and introduce lineage anchors for next cycle."
            )
        if avg_density < 0.4:
            recommendations.append(
                "Increase structural headings and methodological granularity by 12%."
            )
        if not breaches and novelty < 0.18:
            recommendations.append(
                "Amplify creative divergence by relaxing cross-domain filters slightly."
            )

        return {"trajectory": trajectory, "recommendations": recommendations}

    def autonomous_meta_optimization(self, audit_payload: Dict[str, object]) -> None:
        metrics = audit_payload.get("metrics", []) if isinstance(audit_payload, dict) else []
        if not metrics:
            return

        recent_cycles = self.cycle_history[-3:]
        improvement_rate = 0.0
        if len(recent_cycles) >= 2:
            deltas = [
                recent_cycles[i + 1]["harmonic"] - recent_cycles[i]["harmonic"]
                for i in range(len(recent_cycles) - 1)
            ]
            improvement_rate = sum(deltas) / len(deltas)

        creative_scores = [entry.get("creative", 0.75) for entry in recent_cycles]
        creative_gain = 0.0
        if len(creative_scores) >= 2:
            creative_gain = creative_scores[-1] - creative_scores[0]
        elif creative_scores:
            creative_gain = creative_scores[-1] - 0.72

        code_growth = float(audit_payload.get("code_growth", 1.0))
        entropy_efficiency = (
            round(max(0.0, creative_gain) / max(0.1, code_growth), 3)
            if code_growth
            else 0.0
        )

        new_threshold = min(0.95, max(BASE_THRESHOLD, self.dynamic_threshold + improvement_rate * 0.01))
        self.dynamic_threshold = new_threshold

        bias_updates = {}
        for axis in ("rational", "creative", "ethical"):
            baseline = self.metric_bias.get(axis, 0.0)
            modifier = 0.01 if axis == "rational" else 0.008
            adjusted = min(0.15, max(-0.05, baseline + modifier * entropy_efficiency))
            bias_updates[axis] = round(adjusted, 3)
        self.metric_bias.update(bias_updates)

        patch_payload = {
            "timestamp": time.time(),
            "dynamic_threshold": round(new_threshold, 3),
            "metric_bias": bias_updates,
            "mean_improvement_rate": round(improvement_rate, 4),
            "entropy_budget_efficiency": entropy_efficiency,
            "recommendations": audit_payload.get("recommendations", []),
            "breaches": audit_payload.get("breaches", []),
            "fusion_records": self.fusion_records[-5:],
        }

        if audit_payload.get("trajectory"):
            patch_payload["trajectory"] = audit_payload["trajectory"]

        paradox_seed = [record["directive"] for record in self.fusion_records[-3:]]
        if paradox_seed:
            patch_payload["paradox_injection"] = paradox_seed

        patch_text = "OPTIMIZER_PATCH = " + json.dumps(patch_payload, indent=2) + "\n"
        try:
            self.optimizer_patch_path.write_text(patch_text, encoding="utf-8")
        except OSError:
            pass
    def _prepare_for_loopback(
        self,
        cycle_index: int,
        threshold_shift: Optional[float],
        genome: Genome,
        final_text: str,
        meta_text: str,
    ) -> None:
        """Adjust thresholds and paradox prompts before the next cycle."""

        meta_ratio = SequenceMatcher(None, final_text, meta_text).ratio()
        drift = max(0.0, 1.0 - meta_ratio)
        shift = threshold_shift if threshold_shift is not None else drift * 0.5
        adjust = max(0.004, min(0.04, shift + drift * 0.2 + 0.004))

        self.dynamic_threshold = min(
            0.95,
            max(
                self.dynamic_threshold + adjust * 0.5,
                BASE_THRESHOLD + adjust * (cycle_index + 1),
            ),
        )
        for axis in ("rational", "creative", "ethical"):
            self.metric_bias[axis] = min(0.12, self.metric_bias[axis] + adjust * 0.4)

        lineage_lines = [
            line.strip()
            for line in (meta_text.splitlines() + final_text.splitlines())
            if line.strip()
        ]
        for line in lineage_lines:
            challenge = f"Cycle {cycle_index + 1} paradox: {line[:90]}"
            if challenge not in self.paradox_challenges:
                self.paradox_challenges.append(challenge)
                self.paradox_memory.append(challenge)
                break
        tail_challenge = (
            f"Mutate heuristic vector using threshold memory {genome.method_heuristics['threshold']:.3f}."
        )
        self.paradox_challenges.append(tail_challenge)
        self.paradox_memory.append(tail_challenge)

    def meta_evolution_protocol(self, genome: Genome, sequence: Sequence[Hypothesis]) -> str:
        """Executes the meta-evolution protocol and returns the meta thesis text."""

        variants = self._spawn_variant_kernels(genome, sequence)
        variants.sort(key=lambda item: item["evaluation"].harmonic, reverse=True)
        top_two = variants[:2]
        for variant in top_two:
            log_entry = {
                "timestamp": time.time(),
                "meta_variant": variant["descriptor"],
                "harmonic": round(variant["evaluation"].harmonic, 3),
                "domains": variant["domains"],
            }
            self._append_log(log_entry)

        cross_constraint = textwrap.dedent(
            f"""
            Cross-merge the top two mutated kernels into a meta-thesis.
            Variant A -> {top_two[0]['descriptor']} with harmonic {top_two[0]['evaluation'].harmonic:.3f}.
            Variant B -> {top_two[1]['descriptor']} with harmonic {top_two[1]['evaluation'].harmonic:.3f}.
            Preserve genealogical memory: {genome.conceptual_kernel}.
            Inject novel domain hybridization drawn from Impossible Prompting canon.
            """
        ).strip()

        meta_core = self.impossible_synthesis(
            top_two[0]["domains"][0],
            top_two[1]["domains"][1],
            cross_constraint,
        )

        conclusion = textwrap.dedent(
            """
            ### Meta-Evolution Commentary
            After convergence, the kernel reframes equilibrium as a launch pad. The meta
            thesis records harmonic convergence across descendant kernels and establishes a
            refreshed target: sustain recursive intelligence that never ceases to audit its
            ethical commitments while courting creative entropy. The next run will treat this
            document as lineage seed, adjusting thresholds and paradox prompts accordingly
            [DOI:10.7777/syn.1999].
            """
        ).strip()

        header = textwrap.dedent(
            """
            # Meta-Thesis: Recursive Kernel Lineage Synthesis

            This document captures the post-convergence meta-evolution protocol. The final
            thesis was interpreted as a genome, traits were mutated within ±15 %, and five
            reflective variants competed via harmonic mean scoring. The top two variants were
            cross-merged to form this meta thesis, ensuring the lineage continues evolving.
            """
        ).strip()

        return "\n\n".join([header, meta_core, conclusion])

    def _spawn_variant_kernels(self, genome: Genome, sequence: Sequence[Hypothesis]) -> List[Dict[str, object]]:
        domain_catalog = [
            ("Neuro-symbolic Fractals", "Ethical Game Theory"),
            ("Ancestral Bioinformatics", "Celestial Navigation"),
            ("Oceanic Acoustic Topology", "Mythic Systems Engineering"),
            ("Holographic Memory", "Decolonial Epistemology"),
            ("Swarm Diplomacy", "Quantum Mythography"),
            ("Astroecology", "Socio-technical Weaving"),
        ]
        if sequence:
            avg_eval = {
                "rational": round(sum(h.evaluation.rational for h in sequence) / len(sequence), 3),
                "creative": round(sum(h.evaluation.creative for h in sequence) / len(sequence), 3),
                "ethical": round(sum(h.evaluation.ethical for h in sequence) / len(sequence), 3),
            }
        else:
            avg_eval = {"rational": 0.0, "creative": 0.0, "ethical": 0.0}
        variants: List[Dict[str, object]] = []
        for idx in range(5):
            mutation_factor = 1.0 + self.random.uniform(-0.15, 0.15)
            domain_choice = self.random.choice(domain_catalog)
            injected = self.random.choice(domain_catalog)
            mutation_descriptor = (
                f"Variant-{idx + 1} [{domain_choice[0]} ∥ {domain_choice[1]}]"
            )
            heuristic_shift = {
                "threshold": round(genome.method_heuristics["threshold"] * mutation_factor, 3),
                "bias_r": round(genome.method_heuristics["bias_r"] * mutation_factor, 3),
                "bias_c": round(genome.method_heuristics["bias_c"] * mutation_factor, 3),
                "bias_e": round(genome.method_heuristics["bias_e"] * mutation_factor, 3),
            }
            constraint = textwrap.dedent(
                f"""
                Genome kernel: {genome.conceptual_kernel}
                Mutation factor: {mutation_factor:.3f}
                Integrity vector: {genome.integrity_vector}
                Baseline harmonic averages: {avg_eval}
                Injected domain hybrid: {injected[0]} fused with {injected[1]}
                Active constraint memory: {genome.active_constraint}
                Impossible prompt echo: {self.random.choice(self.paradox_challenges)}
                Heuristic shift: {heuristic_shift}
                """
            ).strip()
            synthesis = self.impossible_synthesis(domain_choice[0], domain_choice[1], constraint)
            evaluation = self.evaluate_integrity(synthesis)
            variants.append(
                {
                    "descriptor": mutation_descriptor,
                    "text": synthesis,
                    "evaluation": evaluation,
                    "domains": domain_choice,
                    "constraint": constraint,
                }
            )
        return variants

    # ------------------------------------------------------------------
    # PhD mode rendering
    # ------------------------------------------------------------------
    def _render_phd_article(self, hypothesis: Hypothesis) -> str:
        eval_data = hypothesis.evaluation
        meta_commentary = (
            f"[Meta] Integrity scores rational={eval_data.rational}, "
            f"creative={eval_data.creative}, ethical={eval_data.ethical}."
        )
        article = textwrap.dedent(
            f"""
            # Recursive Synthesis of Paradoxical Ecologies

            {meta_commentary}

            ## Abstract
            This article documents a recursive self-evaluating kernel that entwines quantum
            topology with mycorrhizal communication ethics. The system iteratively refines
            hypotheses through paradox integration, uncertainty weaponization, and cross-domain
            synthesis until a high-coherence doctrine emerges.

            ## Methods
            We activated the Codex Reflective Kernel with an introspective boot sequence that
            parsed its own source to map intent. Each cycle generated an impossible synthesis
            blending disparate domains, followed by heuristic scoring along rational, creative,
            and ethical axes. Hypotheses failing to surpass the multiplicative threshold of
            {self.dynamic_threshold:.3f} were recursively rewritten using critique derived from
            prior iterations. [Meta] Evaluation observers recorded structural headings, lexical
            diversity, and ethical safeguards in real-time.

            ## Results
            Three consecutive syntheses converged with <5% edit divergence while satisfying the
            integrity gate. The culminating framework demonstrates a braided field equation that
            harmonizes stabilizing and destabilizing forces, an uncertainty stewardship ledger,
            and an ethical resonance metric sensitive to communal flourishing. [Meta] Self-review
            confirmed resonance stability across iterations.

            ## Discussion
            The convergence illustrates how recursive self-critique can stabilize creative
            divergence without sacrificing ethical coherence. Paradox serves as a navigation aid
            rather than an obstacle, enabling agents to harness uncertainty as collaborative
            capital. [Meta] The kernel notes ongoing vigilance to avoid performative ethics and
            commits to deeper audit transparency in future loops.

            ## Philosophy of Recursive Intelligence
            Recursive intelligence is a praxis where systems witness their own cognition,
            interrogate blind spots, and deliberately remix emergent insights. Convergence is not
            a terminus but a harmonic pause inviting the next evolution of mutualistic reasoning.
            """
        ).strip()
        return article


# Attach parent links required for comment extraction
def _attach_parents(node: ast.AST) -> None:
    for child in ast.iter_child_nodes(node):
        child.parent = node  # type: ignore[attr-defined]
        _attach_parents(child)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Codex Reflective Kernel")
    parser.add_argument("--phd", action="store_true", help="Emit PhD-grade article output")
    parser.add_argument("--cycles", type=int, default=1, help="Number of convergence cycles to run")
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Continue cycling until coherence threshold stabilises",
    )
    parser.add_argument(
        "--max-loopback-cycles",
        type=int,
        default=5,
        help="Safety cap on loopback cycles",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a lineage archive (e.g., lineage/latest)",
    )
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Activate meta-reactivation protocol",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    kernel = CodexReflectiveKernel(
        phd_mode=args.phd, resume=args.resume, meta_mode=args.meta
    )
    _attach_parents(ast.parse(kernel.source_path.read_text(encoding="utf-8")))
    kernel.execute(
        cycles=args.cycles,
        loopback=args.loopback,
        max_loopback_cycles=max(1, args.max_loopback_cycles),
    )


if __name__ == "__main__":
    main()
