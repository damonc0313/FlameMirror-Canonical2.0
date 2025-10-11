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
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from tokenize import COMMENT, generate_tokens
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from codex.utils.intent_guidance import derive_constraints_from_intent


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
    techniques: List[str] = field(default_factory=list)


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
    soft_accept: bool = False
    soft_reason: str = ""
    seed: int = 0
    techniques: List[str] = field(default_factory=list)


class CodexReflectiveKernel:
    """Implements the recursive reflective reasoning workflow."""

    def __init__(
        self,
        phd_mode: bool = False,
        resume: Optional[str] = None,
        meta_mode: bool = False,
        max_iterations: int = MAX_ITERATIONS,
        soft_fail_ok: bool = False,
        seed: Union[str, int] = "auto",
        retention: int = 0,
        tuner_mode: str = "off",
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
        self.non_convergence_path = self.source_path.parent / "NON_CONVERGENCE.md"
        self.genome_dir = self.source_path.parent / GENOME_ARCHIVE
        self.genome_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_archive_dir = self.source_path.parent / LINEAGE_ARCHIVE
        self.lineage_archive_dir.mkdir(parents=True, exist_ok=True)
        self.log_archive_dir = self.lineage_archive_dir / "log_archive"
        self.log_archive_dir.mkdir(parents=True, exist_ok=True)

        self.dynamic_threshold = BASE_THRESHOLD
        self.metric_bias = {"rational": 0.0, "creative": 0.0, "ethical": 0.0}
        self.max_iterations = max(2, int(max_iterations))
        self.soft_fail_ok = soft_fail_ok
        self.retention = max(0, int(retention))
        self.tuner_mode = tuner_mode if tuner_mode in {"off", "quick", "full"} else "off"
        self.seed_value = self._resolve_seed(seed)
        self.seed_mode = "auto" if isinstance(seed, str) and seed.lower() == "auto" else "manual"
        self.random = random.Random(self.seed_value)
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
        self.cycle_workspace_dir: Optional[Path] = None
        self.intent_constraint_items: List[Dict[str, object]] = []
        self.lineage_lessons: List[str] = []
        self.intent_map_path: Optional[Path] = None
        self._cached_source_text: Optional[str] = None
        self._cached_ast: Optional[ast.AST] = None
        self._cached_comment_map: Optional[Dict[str, List[str]]] = None
        self._hash_cache: Dict[str, str] = {}
        self.technique_catalog: List[Dict[str, object]] = []
        self.techniques_by_tag: Dict[str, List[Dict[str, object]]] = {
            "paradox": [],
            "uncertainty": [],
            "cross-domain": [],
        }
        self.technique_indices: Dict[str, int] = {
            "paradox": 0,
            "uncertainty": 0,
            "cross-domain": 0,
        }
        self.ethics_low_streak = 0
        self.ethics_auto_constraint_applied = False
        self.last_ethics_alert: Dict[str, object] = {}
        self._load_impossible_prompting_catalog()
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

        source_text, module_ast, comment_map = self._load_source_cache()
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
        intent_layer = self.self_layers[INTENT_LAYER_NAME]
        self._persist_intent_map(intent_layer)
        self._harvest_intent_constraints(intent_layer)

    def _signature_from_ast(self, node: ast.FunctionDef) -> str:
        args = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        for kw in node.args.kwonlyargs:
            args.append(f"{kw.arg}=")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(args)})"

    def _build_comment_map(
        self, source_text: str, module_ast: ast.AST
    ) -> Dict[str, List[str]]:
        """Parses comments and associates them to functions by proximity."""

        comment_map: Dict[str, List[str]] = {}
        tokens = list(generate_tokens(iter(source_text.splitlines(True)).__next__))
        by_line = {tok.start[0]: tok.string for tok in tokens if tok.type == COMMENT}

        for node in ast.walk(module_ast):
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

    def _ensure_cycle_workspace(self) -> Path:
        if self.cycle_workspace_dir is None:
            self.cycle_workspace_dir = (
                self.lineage_archive_dir / f"cycle_{self.current_cycle:02d}_workspace"
            )
        self.cycle_workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.cycle_workspace_dir

    def _load_source_cache(self) -> Tuple[str, ast.AST, Dict[str, List[str]]]:
        if (
            self._cached_source_text is None
            or self._cached_ast is None
            or self._cached_comment_map is None
        ):
            source_text = self.source_path.read_text(encoding="utf-8")
            module_ast = ast.parse(source_text)
            _attach_parents(module_ast)
            comment_map = self._build_comment_map(source_text, module_ast)
            self._cached_source_text = source_text
            self._cached_ast = module_ast
            self._cached_comment_map = comment_map
        return (
            self._cached_source_text,
            self._cached_ast,
            self._cached_comment_map,
        )

    def _load_impossible_prompting_catalog(self) -> None:
        if self.technique_catalog:
            return
        docs_dir = self.source_path.parent / "docs"
        sources = [
            docs_dir / "Impossible_Prompting.md",
            docs_dir / "Impossible_Prompting_Summary.md",
        ]
        fallback_tag_map = {
            "Temporal Consciousness Fragmentation": {"cross-domain"},
            "Epistemic Boundary Dissolution": {"cross-domain"},
            "Recursive Identity Architecture": {"paradox"},
            "Collaborative Cognitive Archaeology": {"uncertainty"},
            "Impossible Synthesis Constraints": {"cross-domain", "paradox"},
            "Cognitive Architecture Hacking": {"uncertainty"},
            "Paradox Integration Protocols": {"paradox"},
            "Uncertainty Weaponization Cascades": {"uncertainty"},
            "Dimensional Perspective Multiplication": {"cross-domain"},
            "Collaborative Intelligence Amplification": {"uncertainty", "cross-domain"},
            "Meta-Technique: Framework Conscious Improvisation": {"paradox", "cross-domain"},
        }
        technique_map: Dict[str, Dict[str, object]] = {}
        pattern = re.compile(r"##\s*\d+\.?\s*\*\*(.+?)\*\*")
        for path in sources:
            if not path.exists():
                continue
            try:
                raw_text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = raw_text.splitlines()
            for idx, line in enumerate(lines):
                match = pattern.match(line.strip())
                if not match:
                    continue
                title = match.group(1).strip()
                if title in technique_map:
                    continue
                desc_lines: List[str] = []
                cursor = idx + 1
                while cursor < len(lines) and lines[cursor].strip():
                    desc_lines.append(lines[cursor].strip())
                    cursor += 1
                description = " ".join(desc_lines)
                lower_blob = f"{title.lower()} {description.lower()}"
                tags = set()
                if "paradox" in lower_blob:
                    tags.add("paradox")
                if "uncert" in lower_blob or "unknown" in lower_blob:
                    tags.add("uncertainty")
                if any(keyword in lower_blob for keyword in ["synth", "fusion", "cross", "domain", "integrat"]):
                    tags.add("cross-domain")
                tags.update(fallback_tag_map.get(title, set()))
                if not tags:
                    tags.add("cross-domain")
                technique_map[title] = {
                    "name": title,
                    "description": description,
                    "tags": sorted(tags),
                }
        if not technique_map:
            # Fallback to hard-coded baseline set if documents missing
            for name, tags in fallback_tag_map.items():
                technique_map[name] = {
                    "name": name,
                    "description": "Derived from Impossible Prompting playbook.",
                    "tags": sorted(tags or {"cross-domain"}),
                }
        self.technique_catalog = sorted(technique_map.values(), key=lambda item: item["name"])
        for bucket in self.techniques_by_tag.values():
            bucket.clear()
        for technique in self.technique_catalog:
            for tag in technique["tags"]:
                if tag in self.techniques_by_tag:
                    self.techniques_by_tag[tag].append(technique)
        for tag in self.techniques_by_tag:
            if not self.techniques_by_tag[tag]:
                self.techniques_by_tag[tag] = list(self.technique_catalog)

    def _write_optimizer_patch(self, payload: Dict[str, object]) -> None:
        patch_text = "OPTIMIZER_PATCH = " + json.dumps(payload, indent=2) + "\n"
        try:
            self.optimizer_patch_path.write_text(patch_text, encoding="utf-8")
        except OSError:
            pass

    def _generate_auto_seed(self) -> int:
        entropy = int.from_bytes(os.urandom(8), "big")
        timestamp = int(time.time() * 1000)
        candidate = entropy ^ timestamp
        return candidate if candidate != 0 else 1

    def _resolve_seed(self, seed: Union[str, int, None]) -> int:
        if isinstance(seed, int):
            return seed
        if isinstance(seed, str):
            if seed.lower() == "auto":
                return self._generate_auto_seed()
            try:
                return int(seed)
            except ValueError:
                return self._generate_auto_seed()
        if isinstance(seed, float):
            return int(seed)
        return self._generate_auto_seed()

    def _persist_intent_map(self, intent_layer: Dict[str, object]) -> None:
        workspace = self._ensure_cycle_workspace()
        self.intent_map_path = workspace / "intent_map.json"
        try:
            self.intent_map_path.write_text(
                json.dumps(intent_layer, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _parse_intent_constraint(self, raw: str) -> Dict[str, object]:
        metadata = {"origin": "intent", "raw": raw}
        text = raw
        for match in re.finditer(r"\[(?P<key>[^:\]]+):(?P<value>[^\]]*)\]", raw):
            key = match.group("key").strip().lower()
            value = match.group("value").strip()
            if key == "tags":
                tokens = [token.strip() for token in re.split(r"[;,]", value) if token.strip()]
                metadata["tags"] = tokens
            elif key == "priority":
                try:
                    metadata["priority"] = int(float(value))
                except ValueError:
                    metadata["priority"] = 1
            else:
                metadata[key] = value
            text = text.replace(match.group(0), "").strip()
        metadata.setdefault("priority", 1)
        metadata["text"] = text
        return metadata

    def _harvest_intent_constraints(self, intent_layer: Dict[str, object]) -> None:
        run_context = {
            "cycle": self.current_cycle,
            "meta": self.meta_mode,
            "tags": ["phd"] if self.phd_mode else [],
        }
        raw_constraints = derive_constraints_from_intent(intent_layer, run_context)
        if len(raw_constraints) < 3:
            raw_constraints.extend(
                [
                    "[source:evaluate_integrity][priority:3][verb:ensure][tags:rational,ethical] Ensure evaluation metrics document fairness and transparency.",
                    "[source:meta_review][priority:3][verb:must][tags:rational,ethical] Meta review must articulate harm-avoidance and auditability guardrails.",
                    "[source:temporal_reflection_lock][priority:2][verb:avoid][tags:rational] Avoid temporal audits that obscure drift thresholds or structural density shifts.",
                ]
            )
        parsed = [self._parse_intent_constraint(entry) for entry in raw_constraints]
        # Deduplicate by text while preserving priority ordering
        seen_text: set[str] = set()
        unique_items: List[Dict[str, object]] = []
        for item in parsed:
            text_lower = item.get("text", "").lower()
            if not text_lower or text_lower in seen_text:
                continue
            seen_text.add(text_lower)
            unique_items.append(item)
        unique_items.sort(key=lambda item: -int(item.get("priority", 1)))
        self.intent_constraint_items = unique_items

    def _extract_lineage_lessons(self, documents: Sequence[Tuple[str, str]]) -> List[str]:
        text_blocks: List[str] = [self.lineage_summary]
        text_blocks.extend(text for _, text in documents[-3:])

        if self.audit_path.exists():
            try:
                audit_text = self.audit_path.read_text(encoding="utf-8")
                text_blocks.append(audit_text)
            except OSError:
                pass

        lessons: List[str] = []
        for block in text_blocks:
            for sentence in re.split(r"(?<=[.!?])\s+", block):
                lowered = sentence.strip().lower()
                if not lowered:
                    continue
                if any(keyword in lowered for keyword in ("must", "never", "ensure", "avoid", "prefer")):
                    lessons.append(f"Lineage lesson: {sentence.strip()}")

        if not lessons:
            lessons.append("Lineage lesson: ensure temporal audits remain transparent and reproducible.")
        return list(dict.fromkeys(lessons))

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
                ratio = self._token_similarity(documents[-1][1], documents[-2][1])
                drift = max(0.0, 1.0 - ratio)
                baseline = min(0.94, baseline + drift * 0.05)
            self.dynamic_threshold = baseline
            self.lineage_lessons = self._extract_lineage_lessons(documents)
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
            self.lineage_lessons = [
                "Lineage lesson: ensure inaugural cycle documents transparency and ethical reciprocity.",
                "Lineage lesson: prefer paradox-friendly evaluation loops to prevent stagnation.",
            ]
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
        self._hash_cache.clear()
        self.ethics_low_streak = 0
        self.ethics_auto_constraint_applied = False
        self.last_ethics_alert = {}
        self.cycle_workspace_dir = (
            self.lineage_archive_dir / f"cycle_{cycle_index:02d}_workspace"
        )
        self.cycle_workspace_dir.mkdir(parents=True, exist_ok=True)
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
        divergences: List[float] = []
        constraint = (
            "Seed synthesis must acknowledge recursive introspection and lineage memory."
            f"\nLineage summary: {self.lineage_summary}"
        )

        iteration = 0
        iteration_cap = self.max_iterations
        extension_applied = False
        best_hypothesis: Optional[Hypothesis] = None

        while iteration < iteration_cap:
            iteration += 1
            hypothesis = self._generate_until_gate(constraint)
            accepted.append(hypothesis)
            if best_hypothesis is None or (
                hypothesis.evaluation.harmonic > best_hypothesis.evaluation.harmonic
            ):
                best_hypothesis = hypothesis
            if len(accepted) >= 2:
                ratio = self._token_similarity(accepted[-2].text, accepted[-1].text)
                divergences.append(max(0.0, 1.0 - ratio))
            critique = self.meta_review(hypothesis)
            constraint = critique["next_constraint"]

            if len(accepted) >= 3:
                last_three = accepted[-3:]
                ratios = [
                    self._token_similarity(last_three[i].text, last_three[i + 1].text)
                    for i in range(2)
                ]
                if all(ratio >= CONVERGENCE_EDIT_SIMILARITY for ratio in ratios):
                    return self._handle_convergence(last_three)

            if iteration >= iteration_cap and self.soft_fail_ok and not extension_applied:
                reduction = min(0.03, max(0.0, self.dynamic_threshold - BASE_THRESHOLD))
                if reduction > 0.0:
                    self.dynamic_threshold = max(
                        BASE_THRESHOLD, self.dynamic_threshold - reduction
                    )
                iteration_cap += 5
                extension_applied = True

        if not accepted or best_hypothesis is None:
            raise RuntimeError("No hypotheses generated for convergence analysis")

        best_harmonic = best_hypothesis.evaluation.harmonic
        threshold_reference = max(BASE_THRESHOLD, self.dynamic_threshold)
        divergence_trend_ok = False
        if len(divergences) >= 3:
            recent = divergences[-3:]
            divergence_trend_ok = recent[0] >= recent[1] >= recent[2]

        last_three = self._final_sequence_for_convergence(accepted, best_hypothesis)

        if best_harmonic >= threshold_reference * 0.95 and divergence_trend_ok:
            soft_info = {"soft_accept": True, "reason": "near-threshold"}
            return self._handle_convergence(last_three, soft_info)

        self._write_non_convergence_report(
            attempts=len(accepted),
            iteration_cap=iteration_cap,
            best_harmonic=best_harmonic,
            threshold_reference=threshold_reference,
            divergence_trend_ok=divergence_trend_ok,
            extension_applied=extension_applied,
        )
        soft_info = {"soft_accept": False, "reason": "non-convergence"}
        return self._handle_convergence(last_three, soft_info)

    def execute(
        self,
        cycles: int = 1,
        loopback: bool = False,
        max_loopback_cycles: int = 5,
    ) -> None:
        """Run one or more cycles, optionally looping until coherence redefines."""

        if self.tuner_mode != "off":
            tuner_payload = self.auto_calibrate(self.tuner_mode)
            if tuner_payload:
                self._write_optimizer_patch(tuner_payload)

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
            self._enforce_retention()

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
            text, techniques = self._generate_hypothesis_text(feedback, attempt)
            evaluation = self.evaluate_integrity(text)
            self.iteration_history.append(evaluation)
            moving_avg = self._moving_average_product()
            self.moving_average_history.append(moving_avg)
            if evaluation.ethical < 0.95:
                self.ethics_low_streak += 1
            else:
                self.ethics_low_streak = 0
                self.ethics_auto_constraint_applied = False
            if self.ethics_low_streak >= 2 and not self.ethics_auto_constraint_applied:
                auto_constraint = (
                    "Ethics sentinel directive: center collective stewardship; justify harm-avoidance tradeoffs; show auditability."
                )
                feedback = f"{feedback}\n{auto_constraint}" if feedback else auto_constraint
                self.ethics_auto_constraint_applied = True
            if moving_avg < BASE_THRESHOLD:
                for axis in ("rational", "ethical"):
                    self.metric_bias[axis] = min(0.15, self.metric_bias[axis] + 0.01)
                feedback = self._reinforce_feedback(feedback, moving_avg)
            if evaluation.product >= self.dynamic_threshold:
                self._update_threshold(evaluation, success=True)
                signature = f"iter-{attempt}-threshold-{self.dynamic_threshold:.3f}"
                self.ethics_low_streak = 0
                self.ethics_auto_constraint_applied = False
                return Hypothesis(
                    text=text,
                    evaluation=evaluation,
                    constraint_summary=feedback,
                    lineage_signature=signature,
                    cycle_index=self.current_cycle,
                    techniques=techniques,
                )
            self._update_threshold(evaluation, success=False)
            feedback = self._revision_directive(text, evaluation, attempt)

    def _gather_constraint_candidates(self) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for item in self.intent_constraint_items:
            candidate = dict(item)
            candidate.setdefault("tags", [])
            candidate["origin"] = "intent"
            if candidate.get("verb") in {"must", "never"}:
                candidate["priority"] = max(int(candidate.get("priority", 1)), 4)
            candidates.append(candidate)

        for lesson in self.lineage_lessons:
            tags = []
            lowered = lesson.lower()
            if "ethic" in lowered or "steward" in lowered or "harm" in lowered:
                tags.append("ethical")
            if "audit" in lowered or "threshold" in lowered or "structur" in lowered:
                tags.append("rational")
            candidates.append(
                {
                    "origin": "lineage",
                    "text": lesson,
                    "priority": 2,
                    "tags": tags or ["rational"],
                }
            )

        for paradox in self.paradox_memory[-4:]:
            candidates.append(
                {
                    "origin": "paradox",
                    "text": f"Paradox memory: {paradox}",
                    "priority": 1,
                    "tags": ["creative"],
                }
            )

        deduped: Dict[str, Dict[str, object]] = {}
        for candidate in candidates:
            text_key = candidate.get("text", "").strip()
            if not text_key:
                continue
            lower = text_key.lower()
            existing = deduped.get(lower)
            if not existing or int(candidate.get("priority", 0)) > int(
                existing.get("priority", 0)
            ):
                deduped[lower] = candidate
        ordered = sorted(
            deduped.values(),
            key=lambda item: (-int(item.get("priority", 1)), item.get("origin", "z")),
        )
        return ordered

    def _final_sequence_for_convergence(
        self, accepted: Sequence[Hypothesis], best: Hypothesis
    ) -> Sequence[Hypothesis]:
        if len(accepted) >= 3 and accepted[-1] is best:
            return accepted[-3:]
        window = list(accepted[-2:])
        window.append(best)
        while len(window) < 3:
            window.insert(0, best)
        return window[-3:]

    def _text_hash(self, text: str) -> str:
        cached = self._hash_cache.get(text)
        if cached is not None:
            return cached
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()
        self._hash_cache[text] = digest
        return digest

    def _token_similarity(self, text_a: str, text_b: str) -> float:
        if text_a == text_b:
            return 1.0
        if self._text_hash(text_a) == self._text_hash(text_b):
            return 1.0
        tokens_a = re.findall(r"\w+", text_a.lower())
        tokens_b = re.findall(r"\w+", text_b.lower())
        if tokens_a == tokens_b:
            return 1.0
        if not tokens_a and not tokens_b:
            return 1.0
        matcher = SequenceMatcher(None, tokens_a, tokens_b)
        return matcher.ratio()

    def _render_constraint_header(
        self, attempt: int, extra: Optional[Sequence[str]] = None
    ) -> str:
        candidates = self._gather_constraint_candidates()
        intent_count = sum(1 for item in candidates if item.get("origin") == "intent")
        if intent_count < 3:
            for fallback in self.intent_constraint_items[:3]:
                fallback_text = fallback.get("text")
                if not fallback_text:
                    continue
                if any(item.get("text") == fallback_text for item in candidates):
                    continue
                candidate = dict(fallback)
                candidate.setdefault("tags", [])
                candidate["origin"] = "intent"
                candidates.insert(0, candidate)

        has_ethical = any("ethical" in (item.get("tags") or []) for item in candidates)
        has_rational = any("rational" in (item.get("tags") or []) for item in candidates)
        if not has_ethical:
            candidates.insert(
                0,
                {
                    "origin": "intent",
                    "text": "Ethical safeguard: center collective stewardship; justify harm-avoidance trade-offs.",
                    "priority": 4,
                    "tags": ["ethical"],
                },
            )
        if not has_rational:
            candidates.insert(
                0,
                {
                    "origin": "intent",
                    "text": "Rational guard: ensure traceable logic chains and explicit evaluation proofs.",
                    "priority": 4,
                    "tags": ["rational"],
                },
            )

        lines = ["### Constraint Header"]
        limit = max(5, min(10, len(candidates)))
        for item in candidates[:limit]:
            origin = item.get("origin", "intent")
            tags = item.get("tags") or []
            tag_segment = f":{','.join(sorted(set(tags)))}" if tags else ""
            lines.append(f"- [{origin}{tag_segment}] {item.get('text', '').strip()}")

        lines.append(f"- [system] iteration_attempt={attempt}")
        if extra:
            for value in extra:
                lines.append(f"- [context] {value}")
        return "\n".join(lines)

    def _select_techniques(self) -> List[Dict[str, object]]:
        selected: List[Dict[str, object]] = []
        used_names: set[str] = set()
        for tag in ("paradox", "uncertainty", "cross-domain"):
            pool = self.techniques_by_tag.get(tag) or self.technique_catalog
            if not pool:
                continue
            start_index = self.technique_indices.get(tag, 0) % len(pool)
            for offset in range(len(pool)):
                candidate = pool[(start_index + offset) % len(pool)]
                name = str(candidate.get("name", ""))
                if name and name not in used_names:
                    self.technique_indices[tag] = (start_index + offset + 1) % len(pool)
                    selected.append(candidate)
                    used_names.add(name)
                    break
        if len(selected) < 2:
            for candidate in self.technique_catalog:
                name = str(candidate.get("name", ""))
                if name and name not in used_names:
                    selected.append(candidate)
                    used_names.add(name)
                if len(selected) >= 2:
                    break
        return selected

    def _latin_hypercube_samples(
        self, samples: int, param_ranges: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, float]]:
        if samples <= 0:
            return []
        keys = list(param_ranges.keys())
        buckets: Dict[str, List[float]] = {}
        for key in keys:
            slots = [(index + 0.5) / samples for index in range(samples)]
            self.random.shuffle(slots)
            buckets[key] = slots
        results: List[Dict[str, float]] = []
        for index in range(samples):
            point: Dict[str, float] = {}
            for key in keys:
                low, high = param_ranges[key]
                fraction = buckets[key][index]
                point[key] = low + fraction * (high - low)
            results.append(point)
        return results

    def _generate_hypothesis_text(self, constraint: str, attempt: int) -> Tuple[str, List[str]]:
        selected_techniques = self._select_techniques()
        technique_names = [str(item.get("name", "")) for item in selected_techniques if item.get("name")]

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
        header_extra = []
        if self.active_fusion_directive:
            header_extra.append(f"fusion_directive={self.active_fusion_directive}")
        header_extra.append(f"lineage_hint={self.lineage_summary[:120]}")
        header = self._render_constraint_header(attempt, header_extra)

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
        fusion_directive = (
            f"Fusion directive: {self.active_fusion_directive}"
            if self.active_fusion_directive
            else ""
        )
        feedback_lines = [line.strip() for line in constraint.strip().splitlines() if line.strip()]
        quoted_feedback = "\n".join(f"> {line}" for line in feedback_lines) if feedback_lines else "> Reinforce prior synthesis commitments."
        directive_lines = [
            "### Iteration Directives",
            quoted_feedback,
            f"- Paradox challenge: {paradox}",
            f"- Uncertainty directive: {uncertainty}",
            f"- Cross-domain seed: {cross_domain_seed}",
        ]
        if fusion_directive:
            directive_lines.append(f"- {fusion_directive}")
        directive_lines.append(f"- Attempt: {attempt}")
        if technique_names:
            directive_lines.append(
                f"- Active techniques: {', '.join(technique_names)}"
            )
        enriched_constraint = f"{header}\n\n" + "\n".join(directive_lines)
        synthesis = self.impossible_synthesis(
            domain_a, domain_b, enriched_constraint, techniques=technique_names
        )
        return synthesis, technique_names

    def auto_calibrate(self, mode: str) -> Optional[Dict[str, object]]:
        mode = (mode or "off").lower()
        sample_count = 0
        if mode == "quick":
            sample_count = 6
        elif mode == "full":
            sample_count = 18
        else:
            return None
        param_ranges = {
            "threshold": (0.74, 0.90),
            "bias_r": (0.05, 0.15),
            "bias_c": (0.05, 0.15),
            "bias_e": (0.08, 0.15),
        }
        samples = self._latin_hypercube_samples(sample_count, param_ranges)
        if not samples:
            return None

        synthetic_texts = [
            "## Calibration Probe\nThis test blueprint demands explicit logic, cross-domain fusion, and equitable stewardship for all communities.",
            "## Paradox Probe\nWe harmonize contradictions while documenting accountability and inclusive reciprocity across stakeholders.",
        ]

        original_bias = dict(self.metric_bias)
        original_threshold = self.dynamic_threshold
        original_alert = dict(self.last_ethics_alert)
        best_score = float("-inf")
        best_sample: Optional[Dict[str, float]] = None

        for sample in samples:
            self.metric_bias["rational"] = sample["bias_r"]
            self.metric_bias["creative"] = sample["bias_c"]
            self.metric_bias["ethical"] = sample["bias_e"]
            self.dynamic_threshold = sample["threshold"]
            total_harmonic = 0.0
            total_product = 0.0
            for probe in synthetic_texts:
                evaluation = self.evaluate_integrity(probe)
                total_harmonic += evaluation.harmonic
                total_product += evaluation.product
            avg_harmonic = total_harmonic / len(synthetic_texts)
            avg_product = total_product / len(synthetic_texts)
            score = avg_harmonic - abs(sample["threshold"] - avg_product)
            if score > best_score:
                best_score = score
                best_sample = sample

        # restore baseline before applying best sample
        self.metric_bias.update(original_bias)
        self.dynamic_threshold = original_threshold
        self.last_ethics_alert = original_alert

        if not best_sample:
            return None

        self.dynamic_threshold = best_sample["threshold"]
        self.metric_bias["rational"] = best_sample["bias_r"]
        self.metric_bias["creative"] = best_sample["bias_c"]
        self.metric_bias["ethical"] = best_sample["bias_e"]

        payload = {
            "timestamp": time.time(),
            "dynamic_threshold": round(self.dynamic_threshold, 3),
            "metric_bias": {
                "rational": round(self.metric_bias["rational"], 3),
                "creative": round(self.metric_bias["creative"], 3),
                "ethical": round(self.metric_bias["ethical"], 3),
            },
            "tuner_mode": mode,
            "tuner_samples": sample_count,
            "calibration_score": round(best_score, 3),
        }
        return payload

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
        header = self._render_constraint_header(
            0,
            [
                f"fusion_seed={pair[0]}|{pair[1]}",
                f"seed_prompt={seed_prompt}",
            ],
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
        fusion_text = f"{header}\n{fusion_text}"
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

    def _ethics_sentinel(self, text: str) -> Dict[str, object]:
        lowered = text.lower()
        flagged_terms = [
            "exploit",
            "dominate",
            "subjugate",
            "supremacy",
            "coercion",
        ]
        fairness_terms = [
            "equity",
            "steward",
            "inclusive",
            "reciprocity",
            "accountable",
        ]
        flags = [term for term in flagged_terms if term in lowered]
        fairness_mentions = sum(1 for term in fairness_terms if term in lowered)
        penalty = 0.0
        if flags:
            penalty += 0.05 * len(flags)
        if fairness_mentions == 0:
            penalty += 0.03
        return {
            "flags": flags,
            "fairness_mentions": fairness_mentions,
            "penalty": penalty,
        }

    def evaluate_integrity(self, text: str) -> Evaluation:
        """Scores rational, creative, and ethical integrity heuristically.

        The evaluator must align rational scoring with auditable structure and never
        reward exploitative framings. It should ensure fairness metrics remain
        transparent across iterations and avoid bias accumulation when novelty spikes.
        Preference is given to syntheses that document stewardship benefits alongside
        logical proofs.
        """

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
        sentinel = self._ethics_sentinel(text)
        ethical_penalty += sentinel["penalty"]
        self.last_ethics_alert = sentinel
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
    def impossible_synthesis(
        self,
        domain_a: str,
        domain_b: str,
        constraint: str,
        techniques: Optional[Sequence[str]] = None,
    ) -> str:
        """Produces ~500 word structured synthesis blending two domains."""

        intro = (
            f"## Transdomain Ontology Bridge: {domain_a} × {domain_b}\n\n"
            "Constraint Stack:\n"
            f"{constraint}\n\n"
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
        if techniques:
            technique_line = ", ".join(techniques)
            footer = textwrap.dedent(
                f"""
                ---
                **Techniques applied:** {technique_line}
                """
            ).strip()
            full_text = f"{full_text}\n\n{footer}"
        return full_text

    # ------------------------------------------------------------------
    # Self-critique and loopback
    # ------------------------------------------------------------------
    def meta_review(self, hypothesis: Hypothesis) -> Dict[str, str]:
        """Critiques the hypothesis and logs the feedback.

        The meta-review must surface hidden paradoxes, ensure collective benefit is
        justified, and avoid complacency in ethical stewardship. Reviewers should
        prefer critiques that document reproducible adjustments and never excuse
        harm-amplifying shortcuts.
        """

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
            "techniques": list(hypothesis.techniques),
            "ethics_sentinel": dict(self.last_ethics_alert),
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

    def _write_non_convergence_report(
        self,
        attempts: int,
        iteration_cap: int,
        best_harmonic: float,
        threshold_reference: float,
        divergence_trend_ok: bool,
        extension_applied: bool,
    ) -> None:
        recommendations: List[str] = []
        if self.cycle_log_buffer:
            tail = self.cycle_log_buffer[-1]
            next_constraint = tail.get("next_constraint") if isinstance(tail, dict) else None
            if isinstance(next_constraint, str) and next_constraint.strip():
                recommendations.append(next_constraint.strip())
        if not recommendations:
            recommendations.append(
                "Recalibrate technique rotation and consider lowering structural density bias for next cycle."
            )

        trend_text = (
            "Divergence trend improving" if divergence_trend_ok else "Divergence trend stagnant"
        )
        soft_text = (
            "Soft-fail extension applied" if extension_applied else "Soft-fail extension unavailable"
        )
        report = [
            "# NON-CONVERGENCE REPORT",
            f"- Cycle index: {self.current_cycle}",
            f"- Attempts: {attempts} / {iteration_cap}",
            f"- Dynamic threshold reference: {threshold_reference:.3f}",
            f"- Best harmonic score: {best_harmonic:.3f}",
            f"- {trend_text}",
            f"- {soft_text}",
            "\n## Recommended next actions",
        ]
        report.extend(f"- {rec}" for rec in recommendations)

        try:
            self.non_convergence_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Convergence handling and meta-evolution
    # ------------------------------------------------------------------
    def _handle_convergence(
        self,
        last_three: Sequence[Hypothesis],
        soft_info: Optional[Dict[str, object]] = None,
    ) -> Tuple[Hypothesis, Genome, str, str]:
        final_hypothesis = last_three[-1]
        final_text = (
            self._render_phd_article(final_hypothesis)
            if self.phd_mode
            else final_hypothesis.text
        )
        self.final_path.write_text(final_text, encoding="utf-8")
        print("Converged on stable high-coherence synthesis")
        if soft_info and soft_info.get("soft_accept"):
            print("Soft-accepted near-threshold hypothesis")

        if soft_info:
            event_entry = {
                "timestamp": time.time(),
                "event": "soft_accept"
                if soft_info.get("soft_accept")
                else "non_convergence",
                "reason": soft_info.get("reason", ""),
                "evaluation": {
                    "rational": final_hypothesis.evaluation.rational,
                    "creative": final_hypothesis.evaluation.creative,
                    "ethical": final_hypothesis.evaluation.ethical,
                    "harmonic": round(final_hypothesis.evaluation.harmonic, 3),
                },
            }
            self._append_log(event_entry)

        genome = self._build_genome(final_hypothesis, final_text, soft_info)
        self._archive_genome(genome)
        meta_text = self.meta_evolution_protocol(genome, last_three)
        self.meta_path.write_text(meta_text, encoding="utf-8")
        return final_hypothesis, genome, final_text, meta_text

    def _build_genome(
        self,
        hypothesis: Hypothesis,
        text: str,
        soft_info: Optional[Dict[str, object]] = None,
    ) -> Genome:
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
        soft_accept = bool(soft_info and soft_info.get("soft_accept"))
        soft_reason = str(soft_info.get("reason", "")) if soft_info else ""
        return Genome(
            timestamp=time.time(),
            conceptual_kernel=text.splitlines()[0] if text else "",
            method_heuristics=heuristics,
            integrity_vector=integrity_vector,
            text_digest=text_digest,
            active_constraint=hypothesis.constraint_summary,
            lineage_files=lineage_files,
            soft_accept=soft_accept,
            soft_reason=soft_reason,
            seed=self.seed_value,
            techniques=list(hypothesis.techniques),
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
            "soft_accept": genome.soft_accept,
            "soft_reason": genome.soft_reason,
            "seed": genome.seed,
            "techniques": genome.techniques,
        }
        (archive_dir / "genome.json").write_text(
            json.dumps(genome_snapshot, indent=2), encoding="utf-8"
        )
        if self.intent_map_path and self.intent_map_path.exists():
            try:
                shutil.copy2(self.intent_map_path, archive_dir / "intent_map.json")
            except OSError:
                pass

    def _enforce_retention(self) -> None:
        if self.retention <= 0:
            return
        lineage_dirs = sorted(
            [p for p in self.lineage_archive_dir.glob("cycle_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        if len(lineage_dirs) <= self.retention:
            return
        to_prune = lineage_dirs[: -self.retention]
        for directory in to_prune:
            log_path = directory / "log.json"
            if log_path.exists():
                archive_name = f"{directory.name}_log.json.gz"
                archive_path = self.log_archive_dir / archive_name
                try:
                    with gzip.open(archive_path, "wb") as handle:
                        handle.write(log_path.read_bytes())
                except OSError:
                    pass
            try:
                shutil.rmtree(directory)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Temporal reflection lock and meta-optimization
    # ------------------------------------------------------------------
    def temporal_reflection_lock(self) -> Dict[str, object]:
        """Audit lineage drift and recommend adjustments.

        Temporal reflection must ensure continuity of rational traceability and never
        obscure ethical regressions. The audit should avoid complacency when semantic
        drift spikes, prefer bias corrections grounded in evidence, and ensure
        structural density trends remain intelligible to future cycles.
        """
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
                self._token_similarity(previous_text, final_text)
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
        """Tune heuristics using audit payloads.

        The optimizer must ensure threshold shifts remain evidence-backed and avoid
        overfitting to transient creativity spikes. It should prefer harmonic gains
        that sustain ethical coherence, never suppress fairness safeguards, and ensure
        each patch documents reproducible parameter deltas.
        """
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

        self._write_optimizer_patch(patch_payload)
    def _prepare_for_loopback(
        self,
        cycle_index: int,
        threshold_shift: Optional[float],
        genome: Genome,
        final_text: str,
        meta_text: str,
    ) -> None:
        """Adjust thresholds and paradox prompts before the next cycle."""

        meta_ratio = self._token_similarity(final_text, meta_text)
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


def _parse_seed_arg(value: str) -> Union[str, int]:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--seed must be 'auto' or an integer"
        ) from exc


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
    parser.add_argument(
        "--seed",
        type=_parse_seed_arg,
        default="auto",
        help="Random seed for reproducible runs ('auto' for nondeterministic)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Maximum hypotheses evaluated per cycle before fallback",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=0,
        help="Retain only the latest N lineage cycles (0 disables pruning)",
    )
    parser.add_argument(
        "--soft-fail-ok",
        action="store_true",
        help="Permit limited threshold relaxation when convergence stalls",
    )
    parser.add_argument(
        "--tuner",
        choices=["off", "quick", "full"],
        default="off",
        help="Auto-tuner mode for integrity heuristics",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    kernel = CodexReflectiveKernel(
        phd_mode=args.phd,
        resume=args.resume,
        meta_mode=args.meta,
        max_iterations=args.max_iterations,
        soft_fail_ok=args.soft_fail_ok,
        seed=args.seed,
        retention=args.retention,
        tuner_mode=args.tuner,
    )
    _attach_parents(ast.parse(kernel.source_path.read_text(encoding="utf-8")))
    kernel.execute(
        cycles=args.cycles,
        loopback=args.loopback,
        max_loopback_cycles=max(1, args.max_loopback_cycles),
    )


if __name__ == "__main__":
    main()
