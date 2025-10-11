"""Intent guidance utilities for Codex reflective kernels."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

PROMOTED_FUNCTIONS = {
    "evaluate_integrity",
    "meta_review",
    "temporal_reflection_lock",
    "autonomous_meta_optimization",
}

VERB_KEYWORDS = ("must", "never", "ensure", "avoid", "prefer")
ETHICAL_CUES = (
    "ethic",
    "justice",
    "care",
    "steward",
    "harm",
    "fair",
    "inclusive",
    "responsib",
)
RATIONAL_CUES = (
    "logic",
    "proof",
    "audit",
    "metric",
    "rational",
    "coheren",
    "threshold",
    "structur",
    "integrity",
    "score",
)
CREATIVE_CUES = (
    "improv",
    "novel",
    "metaphor",
    "synth",
    "uncert",
    "paradox",
    "diverg",
)


def _split_sentences(text: str) -> Iterable[str]:
    """Yield clean sentences from a blob of docstring/comment text."""

    sanitized = re.sub(r"[\n\r]+", " ", text)
    for fragment in re.split(r"(?<=[.!?])\s+", sanitized):
        cleaned = fragment.strip()
        if cleaned:
            yield cleaned


def _categorise_sentence(sentence: str) -> Tuple[List[str], str]:
    """Return (tags, verb) extracted from the sentence."""

    lower = sentence.lower()
    tags: List[str] = []
    if any(cue in lower for cue in ETHICAL_CUES):
        tags.append("ethical")
    if any(cue in lower for cue in RATIONAL_CUES):
        tags.append("rational")
    if any(cue in lower for cue in CREATIVE_CUES):
        tags.append("creative")
    verb = next((verb for verb in VERB_KEYWORDS if verb in lower), "")
    return tags, verb


def _iter_function_entries(functions: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for name, payload in functions.items():
        if not isinstance(payload, dict):
            continue
        yield name, payload
        methods = payload.get("methods")
        if isinstance(methods, dict):
            for method_name, method_payload in methods.items():
                if isinstance(method_payload, dict):
                    yield f"{name}.{method_name}", method_payload


def derive_constraints_from_intent(
    intent_map: Dict[str, Any], run_context: Dict[str, Any]
) -> List[str]:
    """Extract high-salience intent constraints from docstrings and comments."""

    functions = intent_map.get("functions") if isinstance(intent_map, dict) else None
    if not isinstance(functions, dict):
        return []

    constraints: List[Tuple[int, int, str]] = []
    seen: set[str] = set()
    index = 0
    context_tags = []
    if isinstance(run_context, dict):
        tags = run_context.get("tags")
        if isinstance(tags, (list, tuple)):
            context_tags = [str(tag) for tag in tags if tag]

    for name, payload in _iter_function_entries(functions):
        texts: List[str] = []
        docstring = payload.get("docstring")
        if isinstance(docstring, str) and docstring.strip():
            texts.append(docstring)
        comments = payload.get("comments")
        if isinstance(comments, list):
            texts.extend(str(comment) for comment in comments if comment)

        priority = 3 if name in PROMOTED_FUNCTIONS else 1
        for text in texts:
            for sentence in _split_sentences(text):
                tags, verb = _categorise_sentence(sentence)
                if not verb:
                    continue
                normalised = " ".join(sentence.split())
                lowered = normalised.lower()
                if lowered in seen:
                    continue
                metadata = [f"[source:{name}]", f"[priority:{priority}]", f"[verb:{verb}]"]
                if tags:
                    metadata.append(f"[tags:{','.join(sorted(set(tags)))}]")
                if context_tags:
                    metadata.append(
                        f"[context:{','.join(sorted(set(context_tags)))}]"
                    )
                constraints.append((priority, index, "".join(metadata) + f" {normalised}"))
                seen.add(lowered)
                index += 1

    # Sort by priority (desc) then insertion order
    constraints.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in constraints]
