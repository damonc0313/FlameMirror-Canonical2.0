from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime

from core.autonomous_agent import create_autonomousagent
from core.code_generator import create_codegenerator
from core.documenter import create_documenter
from core.git_manager import create_gitmanager
from core.test_runner import create_testrunner
from core.validator import create_validator

from api.rest import create_rest, RestRequest
from api.graphql import create_graphql, GraphqlRequest
from api.websocket import create_websocket, WebsocketRequest

from hash_verifier import verify_hashes

logger = logging.getLogger(__name__)


class MegaSystem:
    """High-level orchestrator that coordinates every component in the repository.

    The class keeps instances of all *core* modules and *API* facades. A single
    call to :py:meth:`run_cycle` initializes every component (if necessary),
    executes them, gathers their results, performs a hash-integrity check on the
    sample data, and then returns a consolidated report.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.autonomous_agent = create_autonomousagent()
        self.code_generator = create_codegenerator()
        self.documenter = create_documenter()
        self.git_manager = create_gitmanager()
        self.test_runner = create_testrunner()
        self.validator = create_validator()

        self.rest_api = create_rest()
        self.graphql_api = create_graphql()
        self.websocket_api = create_websocket()

        self._components = [
            self.autonomous_agent,
            self.code_generator,
            self.documenter,
            self.git_manager,
            self.test_runner,
            self.validator,
            self.rest_api,
            self.graphql_api,
            self.websocket_api,
        ]

        # Directory containing test/data sample files for hash verification
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "tests" / "data"
        logger.debug("MegaSystem initialized with data_dir=%s", self.data_dir)

    # ---------------------------------------------------------------------
    # Lifecycle helpers
    # ---------------------------------------------------------------------
    def initialize_all(self) -> None:
        """Initialize every component exactly once."""
        for comp in self._components:
            try:
                comp.initialize()
            except Exception as exc:  # pragma: no cover – robustness helper
                logger.exception("Initialization failed for %s: %s", comp.__class__.__name__, exc)

    def cleanup_all(self) -> None:
        """Cleanup all components (best-effort)."""
        for comp in self._components:
            try:
                comp.cleanup()
            except Exception as exc:  # pragma: no cover
                logger.exception("Cleanup failed for %s: %s", comp.__class__.__name__, exc)

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------
    def run_cycle(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Run a full processing cycle across every component and API.

        Parameters
        ----------
        payload
            Arbitrary request data forwarded to all API endpoints. Defaults to
            an empty dict.
        Returns
        -------
        dict
            Merge of all individual results.
        """
        self.initialize_all()
        payload = payload or {}

        report: Dict[str, Any] = {}

        # Execute core components
        report["autonomous_agent"] = self.autonomous_agent.execute()
        report["code_generator"] = self.code_generator.execute()
        report["documenter"] = self.documenter.execute()
        report["git_manager"] = self.git_manager.execute()
        report["test_runner"] = self.test_runner.execute()
        report["validator"] = self.validator.execute()

        # Execute API facades
        report["rest"] = self.rest_api.execute(RestRequest(data=payload))
        report["graphql"] = self.graphql_api.execute(GraphqlRequest(data=payload))
        report["websocket"] = self.websocket_api.execute(WebsocketRequest(data=payload))

        # Integrity verification on sample hashes
        csv_path = self.data_dir / "sample_hashes.csv"
        if csv_path.exists():
            report["hash_verification"] = verify_hashes(csv_path, base_path=self.data_dir)
        else:
            report["hash_verification"] = {
                "error": f"CSV file not found at {csv_path}",
            }

        # After executing everything, perform cleanup so resources are released
        self.cleanup_all()
        return report

    def save_report(self, report: Dict[str, Any], log_dir: str | Path = "logs") -> Path:
        """Persist a single run report to *log_dir* as a timestamped JSON file.

        The file name format is ``YYYYMMDD_HHMMSS_report.json``.
        Returns the path to the saved file.
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = log_path / f"{ts}_report.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Saved report to %s", out_file)
        return out_file

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<MegaSystem components={len(self._components)}>"