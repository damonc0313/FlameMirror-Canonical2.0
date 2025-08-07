
"""Common base classes for core components."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ComponentConfig:
    """Base configuration shared by core components."""
    enabled: bool = True
    max_retries: int = 3
    timeout: float = 30.0


class BaseComponent:
    """Reusable base providing init, cleanup and health check logic."""

    def __init__(self, name: str, config: Optional[ComponentConfig] = None):
        self.name = name
        self.config = config or ComponentConfig()
        self.logger = logging.getLogger(self.name)
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the component."""
        try:
            self.logger.info("Initializing %s", self.name)
            self._initialized = True
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to initialize %s: %s", self.name, exc)
            return False

    def execute(self, *args, **kwargs) -> Dict[str, Any]:  # pragma: no cover - abstract
        """Execute component logic. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement execute")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up %s", self.name)
        self._initialized = False

    def health_check(self) -> Dict[str, Any]:
        """Return basic health status for monitoring and tests."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "timestamp": datetime.now().isoformat(),
            "component": self.name,
        }
