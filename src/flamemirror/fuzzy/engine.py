"""Fuzzy guidance engine adjusting plan priorities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, cast

import yaml  # type: ignore[import-untyped]


class FuzzyGuidanceEngine:
    def __init__(self, rules_path: Path) -> None:
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, object]]:
        data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        return list(data.get("rules", []))

    def evaluate(self, metrics: Dict[str, float]) -> List[str]:
        advice: List[str] = []
        for rule in self.rules:
            conditions = cast(Dict[str, Dict[str, float]], rule.get("conditions", {}))
            if self._matches(conditions, metrics):
                advice.append(str(rule.get("advice", "")))
        return advice or ["No fuzzy adjustments"]

    def _matches(self, conditions: Dict[str, Dict[str, float]], metrics: Dict[str, float]) -> bool:
        for metric, constraint in conditions.items():
            value = metrics.get(metric, 0.0)
            if "gt" in constraint and not value > float(constraint["gt"]):
                return False
            if "lt" in constraint and not value < float(constraint["lt"]):
                return False
            if "ge" in constraint and not value >= float(constraint["ge"]):
                return False
            if "le" in constraint and not value <= float(constraint["le"]):
                return False
        return True
