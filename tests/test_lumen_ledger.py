"""Compatibility wrapper for the relocated Lumen Ledger tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.lumen_ledger.tests.lumen_ledger_core_tests import *  # noqa: F401,F403
