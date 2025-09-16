"""Utilities for adapting KF-UVE directives to the platform environment.

This module provides a pragmatic, platform-compatible subset of the KF-UVE
"command" stack.  The real specification references a large number of external
systems (cryptographic ledgers, blockchain anchors, multi-compiler validation,
agent collectives, etc.) that are unavailable in the execution environment.

The :class:`AdaptedKFUVE` class simulates the most relevant behaviours with
Python standard library tooling so that higher-level workflows can still
benefit from repeatable validation, lightweight security checks, documentation
introspection, and provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple
import ast
import hashlib
import json
import logging
import sys
import time


LOGGER = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    """Simple chained ledger entry.

    Attributes
    ----------
    digest:
        SHA-256 digest for the stored artifact.
    previous:
        Digest of the previous entry in the chain (``"genesis"`` for the first
        element).
    artifact_type:
        Small descriptor used to identify which subsystem produced the entry
        (e.g. ``"static-analysis"`` or ``"security"``).
    metadata:
        Optional metadata stored alongside the entry.  The values must be JSON
        serialisable so that they can be emitted in reports.
    """

    digest: str
    previous: str
    artifact_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptedKFUVE:
    """Approximate implementation of the KF-UVE workflow.

    The class focuses on tasks that can be executed with the available
    platform tooling:

    * In-memory, hash-chained ledger entries for reproducible provenance.
    * ``ast``-based static analysis for syntax validation and lightweight
      structural metrics.
    * Simple security auditing that flags high-risk API usage patterns.
    * Test execution with coverage sampling via :mod:`trace`.
    * Performance sampling for callables to emulate profiling hooks.
    * Consensus simulation that emulates a voting collective using numeric
      scores.
    * Documentation extraction from the source ``AST``.
    """

    #: Default patterns that should trigger a security warning.
    DEFAULT_BANNED_PATTERNS: Tuple[str, ...] = (
        "eval",
        "exec",
        "os.system",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.run",
    )

    def __init__(
        self,
        *,
        consensus_threshold: float = 0.8,
        banned_patterns: Optional[Sequence[str]] = None,
    ) -> None:
        if consensus_threshold <= 0 or consensus_threshold > 1:
            raise ValueError("consensus_threshold must be between 0 and 1")

        self._consensus_threshold = consensus_threshold
        self._banned_patterns: Tuple[str, ...] = (
            tuple(banned_patterns) if banned_patterns else self.DEFAULT_BANNED_PATTERNS
        )
        self._ledger: List[LedgerEntry] = []
        self._previous_hash: str = "genesis"

    # ------------------------------------------------------------------
    # Ledger helpers
    # ------------------------------------------------------------------
    @property
    def ledger(self) -> List[LedgerEntry]:
        """Return a copy of the ledger entries collected so far."""

        return list(self._ledger)

    @staticmethod
    def _hash_data(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def ledger_seal(self, artifact_type: str, payload: MutableMapping[str, Any]) -> LedgerEntry:
        """Store an artifact in the in-memory ledger.

        Parameters
        ----------
        artifact_type:
            Label describing the stored artifact.
        payload:
            Mapping that will be serialised and included in the ledger entry.

        Returns
        -------
        LedgerEntry
            The newly created ledger entry.
        """

        serialised = json.dumps(payload, sort_keys=True, default=str)
        concatenated = f"{self._previous_hash}:{artifact_type}:{serialised}"
        digest = self._hash_data(concatenated)
        entry = LedgerEntry(
            digest=digest,
            previous=self._previous_hash,
            artifact_type=artifact_type,
            metadata=dict(payload),
        )
        self._ledger.append(entry)
        self._previous_hash = digest
        LOGGER.debug("Ledger entry appended: %s", entry)
        return entry

    # ------------------------------------------------------------------
    # Static analysis and auditing
    # ------------------------------------------------------------------
    def static_analysis(self, code: str) -> Dict[str, Any]:
        """Parse source code and compute lightweight metrics.

        Returns a dictionary describing the parsing outcome and including a
        handful of structural statistics that can be used for diagnostics.
        """

        try:
            module = ast.parse(code)
        except SyntaxError as exc:  # pragma: no cover - error branch exercised via tests
            message = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            LOGGER.debug("Static analysis failed: %s", message)
            return {"status": "failed", "error": message}

        functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(module) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(module) if isinstance(node, (ast.Import, ast.ImportFrom))]

        metrics = {
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
            "has_docstring": ast.get_docstring(module) is not None,
        }
        LOGGER.debug("Static analysis metrics: %s", metrics)
        return {"status": "passed", "metrics": metrics}

    def security_audit(self, code: str) -> Dict[str, Any]:
        """Perform pattern-based security checks.

        The method looks for direct usage of well-known dangerous primitives
        such as ``eval`` or shell execution functions.  The output lists the
        offending nodes together with their line numbers to assist manual
        review.
        """

        issues: List[Dict[str, Any]] = []

        try:
            module = ast.parse(code)
        except SyntaxError as exc:  # pragma: no cover - validated via static analysis tests
            return {
                "status": "failed",
                "error": f"SyntaxError: {exc.msg} (line {exc.lineno})",
            }

        banned = set(self._banned_patterns)

        class SecurityVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    parts: List[str] = []
                    current = node.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value  # type: ignore[assignment]
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                        name = ".".join(reversed(parts))
                if name and name in banned:
                    issues.append({"call": name, "line": node.lineno})
                self.generic_visit(node)

        SecurityVisitor().visit(module)
        status = "failed" if issues else "passed"
        LOGGER.debug("Security audit status=%s issues=%s", status, issues)
        return {"status": status, "issues": issues}

    # ------------------------------------------------------------------
    # Testing, profiling, consensus, documentation
    # ------------------------------------------------------------------
    def run_tests(self, code: str, tests: str) -> Dict[str, Any]:
        """Execute tests with coverage sampling.

        Parameters
        ----------
        code:
            Source code under evaluation.  It is executed once to populate the
            namespace for the tests.
        tests:
            Python statements (typically assertions) that exercise the source
            code.  They are executed in the same namespace as ``code``.
        """

        namespace: Dict[str, Any] = {}

        virtual_filename = "kfuve_virtual_module.py"

        try:
            compiled_code = compile(code, virtual_filename, "exec")
            exec(compiled_code, namespace)
        except Exception as exc:  # pragma: no cover - fails fast
            LOGGER.debug("Code execution failed during test setup: %s", exc)
            return {"status": "failed", "error": f"Code execution failed: {exc}"}

        executed_lines: set[int] = set()

        def tracer(frame, event, arg):  # type: ignore[override]
            if event == "line" and frame.f_code.co_filename == virtual_filename:
                executed_lines.add(frame.f_lineno)
            return tracer

        previous_tracer = sys.gettrace()
        try:
            sys.settrace(tracer)
            exec(compile(tests, "kfuve_virtual_tests.py", "exec"), namespace)
        except Exception as exc:
            LOGGER.debug("Tests failed with error: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            sys.settrace(previous_tracer)

        measured_lines = [
            lineno
            for lineno, line in enumerate(code.splitlines(), start=1)
            if line.strip() and not line.strip().startswith("#")
        ]
        total = len(measured_lines)
        coverage = (len(executed_lines) / total) if total else 1.0
        missing_lines = sorted(set(measured_lines) - executed_lines)

        report = {
            "status": "passed",
            "coverage": coverage,
            "executed_lines": sorted(executed_lines),
            "missing_lines": missing_lines,
        }
        LOGGER.debug("Test run report: %s", report)
        return report

    def profile_performance(self, func: Any, *, runs: int = 10) -> Dict[str, Any]:
        """Profile a callable by repeatedly executing it."""

        if runs <= 0:
            raise ValueError("runs must be a positive integer")
        if not callable(func):
            raise TypeError("func must be callable")

        start = time.perf_counter()
        for _ in range(runs):
            func()
        duration = time.perf_counter() - start
        average_time = duration / runs
        report = {"status": "profiled", "average_time": average_time, "runs": runs}
        LOGGER.debug("Performance profile report: %s", report)
        return report

    def simulate_consensus(self, scores: Sequence[float]) -> Dict[str, Any]:
        """Simulate a swarm consensus vote using numeric scores."""

        if not scores:
            return {"status": "failed", "approved": False, "reason": "no scores"}

        valid_scores = [score for score in scores if 0.0 <= score <= 1.0]
        if len(valid_scores) != len(scores):
            return {
                "status": "failed",
                "approved": False,
                "reason": "scores must be normalised between 0 and 1",
            }

        average = sum(valid_scores) / len(valid_scores)
        approved = average >= self._consensus_threshold
        report = {
            "status": "passed" if approved else "pending",
            "approved": approved,
            "average": average,
            "threshold": self._consensus_threshold,
        }
        LOGGER.debug("Consensus simulation report: %s", report)
        return report

    def generate_documentation(self, code: str) -> Dict[str, Any]:
        """Extract a lightweight documentation summary from ``code``."""

        try:
            module = ast.parse(code)
        except SyntaxError as exc:
            return {
                "status": "failed",
                "error": f"SyntaxError: {exc.msg} (line {exc.lineno})",
            }

        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "lineno": node.lineno,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                methods = [
                    method.name
                    for method in node.body
                    if isinstance(method, ast.FunctionDef)
                ]
                classes.append(
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "methods": methods,
                        "lineno": node.lineno,
                    }
                )

        documentation = {
            "module_docstring": ast.get_docstring(module),
            "functions": functions,
            "classes": classes,
        }
        LOGGER.debug("Documentation generated: %s", documentation)
        return {"status": "generated", "documentation": documentation}

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------
    def run_validation_pipeline(
        self,
        *,
        code: str,
        tests: str,
        performance_callable: Optional[Any] = None,
        consensus_scores: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Run the adapted KF-UVE pipeline and return the collected reports."""

        reports: Dict[str, Any] = {}

        analysis = self.static_analysis(code)
        reports["static_analysis"] = analysis
        self.ledger_seal("static-analysis", analysis)

        security = self.security_audit(code)
        reports["security_audit"] = security
        self.ledger_seal("security-audit", security)

        tests_report = self.run_tests(code, tests)
        reports["tests"] = tests_report
        self.ledger_seal("tests", tests_report)

        if performance_callable is not None:
            profile = self.profile_performance(performance_callable)
            reports["performance"] = profile
            self.ledger_seal("performance", profile)

        documentation = self.generate_documentation(code)
        reports["documentation"] = documentation
        self.ledger_seal("documentation", documentation)

        scores = consensus_scores if consensus_scores is not None else (1.0, 1.0, 1.0)
        consensus = self.simulate_consensus(scores)
        reports["consensus"] = consensus
        self.ledger_seal("consensus", consensus)

        reports["ledger_tail"] = self._previous_hash
        LOGGER.debug("Validation pipeline completed with reports: %s", reports)
        return reports


__all__ = ["AdaptedKFUVE", "LedgerEntry"]
