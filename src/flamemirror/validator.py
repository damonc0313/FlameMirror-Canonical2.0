"""Static validation helpers for FlameMirror."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class ValidationResult:
    """Represents the output of a validation pass."""

    passed: bool
    diagnostics: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class CodeValidator:
    """Performs lightweight static validation of generated code."""

    def __init__(
        self,
        rules: Optional[Sequence[Callable[[ast.AST], Iterable[str]]]] = None,
    ) -> None:
        self.rules = list(rules or [])

    def validate(self, code: str, path: Path) -> ValidationResult:
        diagnostics: List[str] = []
        try:
            tree = ast.parse(code, filename=str(path))
        except SyntaxError as exc:
            diagnostics.append(f"SyntaxError: {exc.msg} (line {exc.lineno})")
            return ValidationResult(
                passed=False,
                diagnostics=diagnostics,
                metrics={"complexity": float("inf")},
            )

        for rule in self.rules:
            diagnostics.extend(rule(tree))

        metrics = {
            "complexity": self._estimate_complexity(tree),
            "lines": float(len(code.splitlines())),
        }
        return ValidationResult(passed=not diagnostics, diagnostics=diagnostics, metrics=metrics)

    def _estimate_complexity(self, tree: ast.AST) -> float:
        walker = _ComplexityCounter()
        walker.visit(tree)
        return float(walker.score)


class _ComplexityCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score: float = 0.0

    def visit_If(self, node: ast.If) -> None:  # noqa: D401
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: D401
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: D401
        self.score += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: D401
        self.score += 0.5
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: D401
        self.score += 0.5
        self.generic_visit(node)
