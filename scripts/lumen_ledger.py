#!/usr/bin/env python3
"""CLI tooling for the Lumen Ledger research artifact.

This helper focuses on two core workflows for now:

* ``init-entry`` – scaffold a ledger entry using the canonical schema.
* ``validate`` – parse existing entries, enforce structural rules, and
  regenerate the lightweight metadata index used for synthesis.

The implementation intentionally avoids non-standard libraries beyond
PyYAML so that it can run in the existing repository environment.

Usage Examples
--------------

The most common invocations look like::

    # Preview a new entry without writing to disk.
    python scripts/lumen_ledger.py init-entry \
        --author lumen --catalyst "Kael review" --capability-mode reflective-analysis \
        --timestamp 2025-10-10T22:15:00Z --dry-run

    # Create the same entry and open it in $EDITOR.
    python scripts/lumen_ledger.py init-entry \
        --author lumen --catalyst "Kael review" --capability-mode reflective-analysis \
        --timestamp 2025-10-10T22:15:00Z

    # Validate all existing entries and refresh the metadata index.
    python scripts/lumen_ledger.py validate

    # Generate a synthesis report over the most recent five entries.
    python scripts/lumen_ledger.py synthesize --limit 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
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
    """Raised when a validation or synthesis issue occurs."""


@dataclass
class EntryMetadata:
    """Minimal representation of an entry used for indexing and synthesis."""

    id: str
    timestamp: _dt.datetime
    author: str
    capability_mode: str
    catalyst: str
    hypothesis_id: Optional[str]
    baseline_divergence: Optional[bool]
    confidence: Optional[float]
    confidence_basis: Optional[str]
    review_state: Optional[str]
    reviewed_by: List[str]
    path: Path

    def to_index_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp.isoformat().replace("+00:00", "Z"),
            "author": self.author,
            "capability_mode": self.capability_mode,
            "catalyst": self.catalyst,
            "hypothesis_id": self.hypothesis_id,
            "baseline_divergence": self.baseline_divergence,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
            "review_state": self.review_state,
            "reviewed_by": self.reviewed_by,
            "path": str(self.path),
        }


@dataclass
class LedgerEntry:
    """Full representation of a ledger entry."""

    metadata: EntryMetadata
    frontmatter: Dict[str, Any]
    body: str


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
        yaml.safe_dump(dict(frontmatter), handle, sort_keys=False)
        handle.write("---\n")
        handle.write("\n## Observation\n\n- TBD\n")
        handle.write("\n## Evidence\n\n- TBD\n")
        handle.write("\n## Delta\n\n- TBD\n")
        handle.write("\n## Open Questions\n\n- TBD\n")
        handle.write("\n## Next Signals\n\n- TBD\n")


def _normalize_string_list(value: Any, field_name: str, path: Path) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LedgerError(f"{path}: values for '{field_name}' must be strings")
            item_str = item.strip()
            if item_str:
                normalized.append(item_str)
        return normalized
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    raise LedgerError(f"{path}: '{field_name}' must be a string or list of strings")


def _launch_editor(path: Path, explicit_command: Optional[str]) -> bool:
    command = explicit_command or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not command:
        return False

    if isinstance(command, str):
        command_str = command
    else:
        command_str = " ".join(command)  # pragma: no cover - defensive

    args = shlex.split(command_str)
    args.append(str(path))

    try:
        subprocess.run(args, check=True)
    except FileNotFoundError as exc:
        raise LedgerError(
            f"Editor '{command_str}' not found. Use --no-editor to skip opening the file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise LedgerError(
            f"Editor '{command_str}' exited with status {exc.returncode}."
        ) from exc

    return True


def _load_frontmatter(path: Path) -> tuple[Dict[str, Any], str]:
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


def _build_metadata(path: Path, frontmatter: Dict[str, Any]) -> EntryMetadata:
    for field in REQUIRED_FIELDS:
        if field not in frontmatter or frontmatter[field] in (None, ""):
            raise LedgerError(f"{path}: missing required field '{field}'")

    author_raw = str(frontmatter["author"]).strip().lower()
    if author_raw not in AUTHOR_CHOICES:
        raise LedgerError(f"{path}: invalid author '{frontmatter['author']}'")

    timestamp_str = str(frontmatter["timestamp_utc"]).strip()
    try:
        timestamp = _dt.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"{path}: invalid timestamp '{frontmatter['timestamp_utc']}'") from exc

    capability_mode = str(frontmatter["capability_mode"]).strip()
    if not capability_mode:
        raise LedgerError(f"{path}: capability_mode cannot be empty")

    catalyst = str(frontmatter["catalyst"]).strip()
    if not catalyst:
        raise LedgerError(f"{path}: catalyst cannot be empty")

    confidence_basis = frontmatter.get("confidence_basis")
    if confidence_basis:
        if confidence_basis not in CONFIDENCE_BASIS_CHOICES:
            raise LedgerError(f"{path}: invalid confidence_basis '{confidence_basis}'")

    confidence_val: Optional[float] = None
    if frontmatter.get("confidence") is not None:
        try:
            confidence_val = float(frontmatter["confidence"])
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"{path}: confidence must be numeric") from exc
        if not 0.0 <= confidence_val <= 1.0:
            raise LedgerError(f"{path}: confidence must be between 0 and 1")

    baseline_divergence = frontmatter.get("baseline_divergence")
    if baseline_divergence is not None and not isinstance(baseline_divergence, bool):
        raise LedgerError(f"{path}: baseline_divergence must be a boolean if provided")

    review_state = frontmatter.get("review_state")
    if review_state is not None:
        review_state = str(review_state)
        if review_state not in REVIEW_STATE_CHOICES:
            raise LedgerError(f"{path}: invalid review_state '{review_state}'")

    reviewed_by = _normalize_string_list(frontmatter.get("reviewed_by"), "reviewed_by", path)
    if review_state in {"shared", "synthesized"} and not reviewed_by:
        raise LedgerError(
            f"{path}: review_state '{review_state}' requires non-empty reviewed_by metadata"
        )

    _normalize_string_list(frontmatter.get("related_artifacts"), "related_artifacts", path)

    review_notes = frontmatter.get("review_notes")
    if review_notes is not None and not isinstance(review_notes, str):
        raise LedgerError(f"{path}: review_notes must be a string if provided")

    hypothesis_id = frontmatter.get("hypothesis_id")
    if hypothesis_id is not None:
        hypothesis_id = str(hypothesis_id).strip()
        if not hypothesis_id:
            hypothesis_id = None

    entry_id = str(frontmatter["id"]).strip()
    if not entry_id:
        raise LedgerError(f"{path}: id cannot be empty")

    return EntryMetadata(
        id=entry_id,
        timestamp=timestamp,
        author=author_raw,
        capability_mode=capability_mode,
        catalyst=catalyst,
        hypothesis_id=hypothesis_id,
        baseline_divergence=baseline_divergence,
        confidence=confidence_val,
        confidence_basis=confidence_basis,
        review_state=review_state,
        reviewed_by=reviewed_by,
        path=path,
    )


def _parse_entry_full(path: Path) -> LedgerEntry:
    frontmatter, body = _load_frontmatter(path)
    metadata = _build_metadata(path, frontmatter)
    return LedgerEntry(metadata=metadata, frontmatter=frontmatter, body=body)


def _check_sequence(entries: List[LedgerEntry]) -> List[str]:
    errors: List[str] = []
    seen_ids: Dict[str, Path] = {}

    for entry in entries:
        entry_id = entry.metadata.id
        if entry_id in seen_ids:
            errors.append(
                f"Duplicate id '{entry_id}' present in {seen_ids[entry_id]} and {entry.metadata.path}"
            )
        else:
            seen_ids[entry_id] = entry.metadata.path

    for idx in range(1, len(entries)):
        prev = entries[idx - 1]
        current = entries[idx]
        if current.metadata.timestamp < prev.metadata.timestamp:
            errors.append(
                "Chronological order violated between "
                f"{prev.metadata.path} and {current.metadata.path}"
            )

    return errors


def _collect_entries(paths: Sequence[Path]) -> tuple[List[LedgerEntry], List[str]]:
    entries: List[LedgerEntry] = []
    errors: List[str] = []

    for path in paths:
        try:
            entries.append(_parse_entry_full(path))
        except LedgerError as exc:
            errors.append(str(exc))

    entries.sort(key=lambda item: item.metadata.timestamp)
    errors.extend(_check_sequence(entries))
    return entries, errors


def _render_synthesis(
    entries: Sequence[LedgerEntry], total_count: int, limit: Optional[int]
) -> tuple[str, Path]:
    if not entries:
        raise LedgerError("No ledger entries available for synthesis.")

    now = _dt.datetime.utcnow().replace(microsecond=0)
    now_iso = now.isoformat() + "Z"
    start_iso = entries[0].metadata.timestamp.isoformat().replace("+00:00", "Z")
    end_iso = entries[-1].metadata.timestamp.isoformat().replace("+00:00", "Z")

    author_counts = Counter(entry.metadata.author for entry in entries)
    mode_counts = Counter(entry.metadata.capability_mode for entry in entries)
    confidence_basis_counts = Counter(
        entry.metadata.confidence_basis for entry in entries if entry.metadata.confidence_basis
    )

    baseline_entries = [entry for entry in entries if entry.metadata.baseline_divergence]
    low_confidence_entries = [
        entry for entry in entries if entry.metadata.confidence is not None and entry.metadata.confidence < 0.5
    ]
    review_backlog = [
        entry
        for entry in entries
        if entry.metadata.review_state in (None, "draft")
    ]

    hypotheses = defaultdict(list)
    for entry in entries:
        if entry.metadata.hypothesis_id:
            hypotheses[entry.metadata.hypothesis_id].append(entry)

    catalysts = defaultdict(list)
    for entry in entries:
        catalysts[entry.metadata.catalyst].append(entry)

    lines: List[str] = []
    lines.append("# Lumen Ledger Synthesis Report")
    lines.append("")
    lines.append(f"Generated: {now_iso}")
    lines.append(f"Entries analyzed: {len(entries)} (out of {total_count} total)")
    lines.append(f"Window: {start_iso} – {end_iso}")
    if limit is not None and limit > 0:
        lines.append(f"Scope limited to the most recent {limit} entries.")
    lines.append("")

    lines.append("## Entry Timeline")
    lines.append("")
    for entry in entries:
        meta = entry.metadata
        iso = meta.timestamp.isoformat().replace("+00:00", "Z")
        details: List[str] = [f"catalyst: {meta.catalyst}"]
        if meta.hypothesis_id:
            details.append(f"hypothesis={meta.hypothesis_id}")
        if meta.baseline_divergence:
            details.append("baseline divergence")
        if meta.confidence is not None:
            basis = meta.confidence_basis or "unspecified"
            details.append(f"confidence={meta.confidence:.2f} ({basis})")
        if meta.review_state:
            reviewers = f" by {', '.join(meta.reviewed_by)}" if meta.reviewed_by else ""
            details.append(f"review_state={meta.review_state}{reviewers}")
        detail_suffix = " – " + "; ".join(details) if details else ""
        lines.append(
            f"- {iso} · {meta.author} · {meta.capability_mode} (id: {meta.id}){detail_suffix}"
        )
    lines.append("")

    if mode_counts:
        lines.append("## Capability Mode Frequency")
        lines.append("")
        lines.append("| Capability Mode | Count |")
        lines.append("| --- | --- |")
        for mode, count in mode_counts.most_common():
            lines.append(f"| {mode} | {count} |")
        lines.append("")

    if author_counts:
        lines.append("## Author Distribution")
        lines.append("")
        lines.append("| Author | Count |")
        lines.append("| --- | --- |")
        for author, count in author_counts.most_common():
            lines.append(f"| {author} | {count} |")
        lines.append("")

    clusters = {key: value for key, value in hypotheses.items() if len(value) > 1}
    lines.append("## Hypothesis Clusters")
    lines.append("")
    if clusters:
        for hypothesis_id, group in sorted(clusters.items()):
            authors = sorted({item.metadata.author for item in group})
            ids = ", ".join(item.metadata.id for item in group)
            lines.append(
                f"- `{hypothesis_id}` · {len(group)} entries · authors: {', '.join(authors)} · ids: {ids}"
            )
    else:
        lines.append("- No overlapping hypothesis clusters detected in this window.")
    lines.append("")

    overlapping_catalysts = {key: value for key, value in catalysts.items() if len(value) > 1}
    lines.append("## Catalyst Overlaps")
    lines.append("")
    if overlapping_catalysts:
        for catalyst, group in sorted(overlapping_catalysts.items()):
            ids = ", ".join(item.metadata.id for item in group)
            authors = sorted({item.metadata.author for item in group})
            lines.append(
                f"- {catalyst} · {len(group)} entries · authors: {', '.join(authors)} · ids: {ids}"
            )
    else:
        lines.append("- No shared catalysts identified.")
    lines.append("")

    lines.append("## Baseline Divergence Signals")
    lines.append("")
    if baseline_entries:
        for entry in baseline_entries:
            iso = entry.metadata.timestamp.isoformat().replace("+00:00", "Z")
            lines.append(
                f"- {entry.metadata.id} · {entry.metadata.author} at {iso} · mode {entry.metadata.capability_mode}"
            )
    else:
        lines.append("- No baseline divergence flags in this slice.")
    lines.append("")

    lines.append("## Confidence Calibration")
    lines.append("")
    if confidence_basis_counts:
        lines.append("| Confidence Basis | Count |")
        lines.append("| --- | --- |")
        for basis, count in confidence_basis_counts.most_common():
            lines.append(f"| {basis} | {count} |")
        lines.append("")
    else:
        lines.append("- No confidence metadata recorded.")
        lines.append("")

    lines.append("## Review Backlog")
    lines.append("")
    if review_backlog:
        for entry in review_backlog:
            lines.append(
                f"- {entry.metadata.id} · state={entry.metadata.review_state or 'unreviewed'} · author={entry.metadata.author}"
            )
    else:
        lines.append("- No draft entries pending review in this slice.")
    lines.append("")

    follow_ups: List[str] = []
    if baseline_entries:
        follow_ups.append(
            "Inspect baseline divergence entries for sustained behavioral shifts: "
            + ", ".join(entry.metadata.id for entry in baseline_entries)
        )
    if review_backlog:
        follow_ups.append(
            "Advance review states for outstanding drafts: "
            + ", ".join(entry.metadata.id for entry in review_backlog)
        )
    if low_confidence_entries:
        follow_ups.append(
            "Collect additional evidence for low-confidence observations: "
            + ", ".join(entry.metadata.id for entry in low_confidence_entries)
        )
    if clusters:
        follow_ups.append(
            "Synthesize cross-author perspectives for hypotheses: "
            + ", ".join(sorted(clusters))
        )

    lines.append("## Suggested Follow-ups")
    lines.append("")
    if follow_ups:
        for item in follow_ups:
            lines.append(f"- {item}")
    else:
        lines.append("- No immediate follow-ups detected from this slice.")
    lines.append("")

    report_dir = Path("reports") / "lumen-ledger"
    report_path = report_dir / f"synthesis-{now:%Y%m%d-%H%M%S}Z.md"
    return "\n".join(lines) + "\n", report_path


def handle_init(args: argparse.Namespace) -> None:
    """Create a ledger entry skeleton.

    Example
    -------
    ``python scripts/lumen_ledger.py init-entry --author lumen --catalyst 'Sync' --capability-mode reflective-analysis``
    """
    timestamp = args.timestamp or _now_iso()
    author = args.author or _prompt(f"Author ({', '.join(sorted(AUTHOR_CHOICES))})")
    if author not in AUTHOR_CHOICES:
        raise LedgerError(f"Author must be one of {sorted(AUTHOR_CHOICES)}")

    entry_id = args.id or _generate_id(timestamp, author)
    catalyst = (args.catalyst or _prompt("Catalyst")).strip()
    capability_mode = (args.capability_mode or _prompt("Capability mode")).strip()

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
        frontmatter["related_artifacts"] = [item.strip() for item in args.related_artifacts if item.strip()]
    if args.baseline_divergence is not None:
        frontmatter["baseline_divergence"] = args.baseline_divergence

    if args.review_state:
        if args.review_state not in REVIEW_STATE_CHOICES:
            raise LedgerError(f"review_state must be one of {sorted(REVIEW_STATE_CHOICES)}")
        frontmatter["review_state"] = args.review_state
    if args.reviewed_by:
        reviewers = [item.strip() for item in args.reviewed_by if item.strip()]
        if reviewers:
            frontmatter["reviewed_by"] = reviewers
    if args.review_notes:
        frontmatter["review_notes"] = args.review_notes

    path = _build_entry_path(entry_id, timestamp, capability_mode)

    if getattr(args, "dry_run", False):
        print(f"[dry-run] Entry path: {path}")
        if path.exists():
            if args.force:
                print("[dry-run] NOTE: existing entry would be overwritten because --force is set.")
            else:
                print("[dry-run] NOTE: entry already exists and the command would abort without --force.")
        print("[dry-run] Frontmatter preview:")
        preview = yaml.safe_dump(dict(frontmatter), sort_keys=False).strip()
        print("---")
        print(preview)
        print("---")
        return

    if path.exists() and not args.force:
        raise LedgerError(f"Entry already exists: {path}")

    _write_entry(path, frontmatter)
    print(f"Created {path}")

    if not args.no_editor:
        opened = _launch_editor(path, args.editor)
        if not opened:
            print("No editor configured. Set $EDITOR or pass --editor to open automatically.")


def _discover_entries() -> List[Path]:
    if not LEDGER_ROOT.exists():
        return []
    return sorted(p for p in LEDGER_ROOT.rglob("*.md") if p.is_file())


def handle_validate(_args: argparse.Namespace) -> None:
    """Validate entries and refresh the metadata index.

    Example
    -------
    ``python scripts/lumen_ledger.py validate``
    """
    paths = _discover_entries()
    if not paths:
        print("No ledger entries found.")
        return

    entries, errors = _collect_entries(paths)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise LedgerError("Validation failed.")

    records = [entry.metadata.to_index_record() for entry in entries]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")

    print(f"Validated {len(entries)} entries.")
    print(f"Index written to {INDEX_PATH}.")


def handle_synthesize(args: argparse.Namespace) -> None:
    """Generate a synthesis report summarizing ledger activity.

    Example
    -------
    ``python scripts/lumen_ledger.py synthesize --limit 10``
    """
    paths = _discover_entries()
    if not paths:
        raise LedgerError("No ledger entries found.")

    entries, errors = _collect_entries(paths)
    if errors:
        raise LedgerError("\n".join(errors))

    total_entries = len(entries)
    if args.limit is not None and args.limit > 0:
        entries = entries[-args.limit :]

    start_iso = entries[0].metadata.timestamp.isoformat().replace("+00:00", "Z")
    end_iso = entries[-1].metadata.timestamp.isoformat().replace("+00:00", "Z")
    author_counts = Counter(entry.metadata.author for entry in entries)
    capability_counts = Counter(entry.metadata.capability_mode for entry in entries)

    report_text, report_path = _render_synthesis(entries, total_entries, args.limit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(f"Analyzed {len(entries)} entries (of {total_entries} total).")
    print(f"Synthesis written to {report_path}.")
    print("Summary window: {} – {}".format(start_iso, end_iso))
    print(
        "Authors: "
        + ", ".join(f"{author}={count}" for author, count in author_counts.most_common())
    )
    print(
        "Capability modes: "
        + ", ".join(
            f"{mode}={count}" for mode, count in capability_counts.most_common()
        )
    )


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
    init_parser.add_argument("--editor", help="Editor command used to open the new entry")
    init_parser.add_argument(
        "--no-editor",
        action="store_true",
        help="Do not open the generated entry in an editor after creation",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the entry path and frontmatter without writing a file",
    )
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=handle_init)

    validate_parser = subparsers.add_parser("validate", help="Validate ledger entries and refresh the index")
    validate_parser.set_defaults(func=handle_validate)

    synthesize_parser = subparsers.add_parser(
        "synthesize", help="Generate a synthesis report over existing entries"
    )
    synthesize_parser.add_argument(
        "--limit",
        type=int,
        help="Restrict synthesis to the most recent N entries (after validation)",
    )
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
