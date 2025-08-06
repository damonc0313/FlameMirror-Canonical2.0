#!/usr/bin/env python3
"""Command-line entry point to run a full MegaSystem cycle and persist its output.

Usage examples
--------------
Run with default payload and save report to *logs/* (created if necessary)::

    python -m mega_system.main

Pass a JSON payload string::

    python -m mega_system.main --payload '{"message": "hello"}'

Supply payload via file and custom log directory::

    python -m mega_system.main --payload payload.json --log-dir my_reports
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

from mega_system import MegaSystem


def _load_payload(raw: str) -> Dict[str, Any]:
    """Interpret *raw* either as JSON string or path to a JSON file."""
    path = pathlib.Path(raw)
    if path.is_file():
        with path.open() as fh:
            return json.load(fh)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MegaSystem cycle and save the report")
    parser.add_argument("--payload", default="{}", help="JSON string or path to JSON file with request data")
    parser.add_argument("--log-dir", type=pathlib.Path, default="logs", help="Directory to store reports")
    args = parser.parse_args()

    payload = _load_payload(args.payload)
    system = MegaSystem()
    report = system.run_cycle(payload)
    system.save_report(report, args.log_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()