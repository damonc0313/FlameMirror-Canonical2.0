#!/usr/bin/env python3
"""
Configuration Manager with Environment Overrides
SYNTHESIZED from learned codebase patterns
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class ConfigSchema:
    """Schema for configuration validation."""
    required_keys: list = field(default_factory=list)
    optional_keys: dict = field(default_factory=dict)
    validators: dict = field(default_factory=dict)

class ConfigManager:
    """
    Hierarchical configuration manager with:
    - File-based config (JSON/ENV)
    - Environment variable overrides
    - Schema validation
    - Hot reload support
    """
    
    def __init__(
        self,
        config_file: Path,
        schema: Optional[ConfigSchema] = None,
        env_prefix: str = "APP_"
    ):
        self.config_file = Path(config_file)
        self.schema = schema
        self.env_prefix = env_prefix
        self._config: Dict[str, Any] = {}
        self._watchers = []
        
        self.load()
    
    def load(self):
        """Load config from file and apply env overrides."""
        # Load from file
        if self.config_file.exists():
            with open(self.config_file) as f:
                self._config = json.load(f)
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        # Validate if schema provided
        if self.schema:
            self._validate()
    
    def _apply_env_overrides(self):
        """Override config values from environment."""
        for key, value in self._config.items():
            env_key = f"{self.env_prefix}{key.upper()}"
            if env_key in os.environ:
                # Type coercion based on original type
                original_type = type(value)
                try:
                    self._config[key] = original_type(os.environ[env_key])
                except (ValueError, TypeError):
                    self._config[key] = os.environ[env_key]
    
    def _validate(self):
        """Validate config against schema."""
        if not self.schema:
            return
        
        # Check required keys
        for key in self.schema.required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Run custom validators
        for key, validator in self.schema.validators.items():
            if key in self._config:
                if not validator(self._config[key]):
                    raise ValueError(f"Validation failed for {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set config value."""
        self._config[key] = value
    
    def save(self):
        """Save current config to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def reload(self):
        """Hot reload config from file."""
        self.load()
        self._notify_watchers()
    
    def watch(self, callback):
        """Register callback for config changes."""
        self._watchers.append(callback)
    
    def _notify_watchers(self):
        """Notify all watchers of config change."""
        for callback in self._watchers:
            try:
                callback(self._config)
            except Exception:
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return self._config.copy()

if __name__ == '__main__':
    # Example usage
    schema = ConfigSchema(
        required_keys=['database_url', 'api_key'],
        validators={
            'max_connections': lambda x: x > 0,
            'timeout': lambda x: 0 < x < 300
        }
    )
    
    config = ConfigManager('config.json', schema=schema)
    print(f"Database: {config.get('database_url', 'localhost')}")
    print(f"Config loaded successfully")
