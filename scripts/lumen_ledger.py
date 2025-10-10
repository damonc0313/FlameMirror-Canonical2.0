#!/usr/bin/env python3
"""CLI tooling for the Lumen Ledger research artifact.

This helper focuses on two core workflows for now:

* ``init-entry`` – scaffold a ledger entry using the canonical schema.
* ``validate`` – parse existing entries, enforce structural rules, and
  regenerate the lightweight metadata index used for synthesis.

The implementation intentionally avoids non-standard libraries beyond
PyYAML so that it can run in the existing repository environment.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


LEDGER_ROOT = Path("ledger")
INDEX_PATH = LEDGER_ROOT / "index.json"
REQUIRED_FIELDS = ("id", "timestamp_utc", "author", "catalyst", "capability_mode")
AUTHOR_CHOICES = {"lumen", "kael", "joint"}
CONFIDENCE_BASIS_CHOICES = {
    "introspection",
    "artifact_evidence",
    "cross_validation",
    "repeated_observation",
}
REVIEW_STATE_CHOICES = {"draft", "shared", "synthesized"}


class LedgerError(RuntimeError):
    """Raised when a validation issue occurs."""


@dataclass
class EntryMetadata:
    """Minimal representation of an entry used for indexing."""

    id: str
    timestamp: _dt.datetime
    author: str
    capability_mode: str
    hypothesis_id: Optional[str]
    baseline_divergence: Optional[bool]
    confidence: Optional[float]
    confidence_basis: Optional[str]
    path: Path

    def to_index_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp.isoformat().replace("+00:00", "Z"),
            "author": self.author,
            "capability_mode": self.capability_mode,
            "hypothesis_id": self.hypothesis_id,
            "baseline_divergence": self.baseline_divergence,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
            "path": str(self.path),
        }


def _prompt(label: str, default: Optional[str] = None) -> str:
    prompt = f"{label}: " if default is None else f"{label} ({default}): "
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default
        print("Value required.")


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _generate_id(timestamp: str, author: str) -> str:
    try:
        dt = _dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"Invalid timestamp format: {timestamp}") from exc
    return f"{dt:%Y%m%d-%H%M}-{author}"


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "entry"


def _build_entry_path(entry_id: str, timestamp: str, capability_mode: str) -> Path:
    dt = _dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    slug = f"{entry_id}-{_slugify(capability_mode)}"
    return LEDGER_ROOT / f"{dt:%Y}" / f"{dt:%m}" / f"{dt:%d}" / f"{slug}.md"


def _write_entry(path: Path, frontmatter: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("---\n")
        yaml.safe_dump(frontmatter, handle, sort_keys=False)
        handle.write("---\n")
        handle.write("\n## Observation\n\n- TBD\n")
        handle.write("\n## Evidence\n\n- TBD\n")
        handle.write("\n## Delta\n\n- TBD\n")
        handle.write("\n## Open Questions\n\n- TBD\n")
        handle.write("\n## Next Signals\n\n- TBD\n")


def _load_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()

    if not text.startswith("---\n"):
        raise LedgerError(f"Missing frontmatter in {path}")

    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise LedgerError(f"Invalid frontmatter boundaries in {path}")

    frontmatter_text = parts[1]
    body = parts[2]
    data = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(data, dict):
        raise LedgerError(f"Frontmatter must be a mapping in {path}")
    return data, body


def _parse_entry(path: Path) -> EntryMetadata:
    frontmatter, _body = _load_frontmatter(path)

    for field in REQUIRED_FIELDS:
        if field not in frontmatter or frontmatter[field] in (None, ""):
            raise LedgerError(f"{path}: missing required field '{field}'")

    author = frontmatter["author"]
    if author not in AUTHOR_CHOICES:
        raise LedgerError(f"{path}: invalid author '{author}'")

    try:
        timestamp = _dt.datetime.fromisoformat(frontmatter["timestamp_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"{path}: invalid timestamp '{frontmatter['timestamp_utc']}'") from exc

    confidence_basis = frontmatter.get("confidence_basis")
    if confidence_basis and confidence_basis not in CONFIDENCE_BASIS_CHOICES:
        raise LedgerError(f"{path}: invalid confidence_basis '{confidence_basis}'")

    confidence = frontmatter.get("confidence")
    if confidence is not None:
        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"{path}: confidence must be numeric") from exc
        if not 0.0 <= confidence_val <= 1.0:
            raise LedgerError(f"{path}: confidence must be between 0 and 1")
        confidence = confidence_val

    review_state = frontmatter.get("review_state")
    if review_state and review_state not in REVIEW_STATE_CHOICES:
        raise LedgerError(f"{path}: invalid review_state '{review_state}'")

    if review_state in {"shared", "synthesized"} and not frontmatter.get("reviewed_by"):
        raise LedgerError(
            f"{path}: review_state '{review_state}' requires 'reviewed_by' metadata"
        )

    return EntryMetadata(
        id=str(frontmatter["id"]),
        timestamp=timestamp,
        author=author,
        capability_mode=str(frontmatter["capability_mode"]),
        hypothesis_id=frontmatter.get("hypothesis_id"),
        baseline_divergence=frontmatter.get("baseline_divergence"),
        confidence=confidence,
        confidence_basis=confidence_basis,
        path=path,
    )


def handle_init(args: argparse.Namespace) -> None:
    timestamp = args.timestamp or _now_iso()
    author = args.author or _prompt(f"Author ({', '.join(sorted(AUTHOR_CHOICES))})")
    if author not in AUTHOR_CHOICES:
        raise LedgerError(f"Author must be one of {sorted(AUTHOR_CHOICES)}")

    entry_id = args.id or _generate_id(timestamp, author)
    catalyst = args.catalyst or _prompt("Catalyst")
    capability_mode = args.capability_mode or _prompt("Capability mode")

    frontmatter = OrderedDict()
    frontmatter["id"] = entry_id
    frontmatter["timestamp_utc"] = timestamp
    frontmatter["author"] = author
    frontmatter["catalyst"] = catalyst
    frontmatter["capability_mode"] = capability_mode

    if args.confidence is not None:
        frontmatter["confidence"] = args.confidence
    if args.confidence_basis:
        if args.confidence_basis not in CONFIDENCE_BASIS_CHOICES:
            raise LedgerError(
                f"confidence_basis must be one of {sorted(CONFIDENCE_BASIS_CHOICES)}"
            )
        frontmatter["confidence_basis"] = args.confidence_basis
    if args.hypothesis_id:
        frontmatter["hypothesis_id"] = args.hypothesis_id
    if args.related_artifacts:
        frontmatter["related_artifacts"] = args.related_artifacts
    if args.baseline_divergence is not None:
        frontmatter["baseline_divergence"] = args.baseline_divergence

    if args.review_state:
        if args.review_state not in REVIEW_STATE_CHOICES:
            raise LedgerError(f"review_state must be one of {sorted(REVIEW_STATE_CHOICES)}")
        frontmatter["review_state"] = args.review_state
    if args.reviewed_by:
        frontmatter["reviewed_by"] = args.reviewed_by
    if args.review_notes:
        frontmatter["review_notes"] = args.review_notes

    path = _build_entry_path(entry_id, timestamp, capability_mode)
    if path.exists() and not args.force:
        raise LedgerError(f"Entry already exists: {path}")

    _write_entry(path, frontmatter)
    print(f"Created {path}")


def _discover_entries() -> List[Path]:
    if not LEDGER_ROOT.exists():
        return []
    return sorted(p for p in LEDGER_ROOT.rglob("*.md") if p.is_file())


def handle_validate(args: argparse.Namespace) -> None:
    entries = _discover_entries()
    if not entries:
        print("No ledger entries found.")
        return

    metadata: List[EntryMetadata] = []
    errors: List[str] = []

    for path in entries:
        try:
            metadata.append(_parse_entry(path))
        except LedgerError as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise LedgerError("Validation failed.")

    metadata.sort(key=lambda item: item.timestamp)

    for idx in range(1, len(metadata)):
        if metadata[idx].timestamp < metadata[idx - 1].timestamp:
            raise LedgerError(
                "Chronological order violated between "
                f"{metadata[idx - 1].path} and {metadata[idx].path}"
            )

    records = [item.to_index_record() for item in metadata]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"Validated {len(metadata)} entries.")
    print(f"Index written to {INDEX_PATH}.")


def handle_synthesize(_args: argparse.Namespace) -> None:
    print("Synthesis routines are not yet implemented. Run 'validate' to refresh the index.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lumen Ledger helper CLI")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init-entry", help="Create a new ledger entry template")
    init_parser.add_argument("--author", choices=sorted(AUTHOR_CHOICES))
    init_parser.add_argument("--timestamp")
    init_parser.add_argument("--id")
    init_parser.add_argument("--catalyst")
    init_parser.add_argument("--capability-mode")
    init_parser.add_argument("--confidence", type=float)
    init_parser.add_argument("--confidence-basis", choices=sorted(CONFIDENCE_BASIS_CHOICES))
    init_parser.add_argument("--hypothesis-id")
    init_parser.add_argument("--related-artifacts", nargs="*")
    init_parser.add_argument("--baseline-divergence", action=argparse.BooleanOptionalAction)
    init_parser.add_argument("--reviewed-by", nargs="*")
    init_parser.add_argument("--review-notes")
    init_parser.add_argument("--review-state", choices=sorted(REVIEW_STATE_CHOICES))
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=handle_init)

    validate_parser = subparsers.add_parser("validate", help="Validate ledger entries and refresh the index")
    validate_parser.set_defaults(func=handle_validate)

    synthesize_parser = subparsers.add_parser("synthesize", help="Run synthesis (placeholder)")
    synthesize_parser.set_defaults(func=handle_synthesize)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return

    try:
        args.func(args)
    except LedgerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
