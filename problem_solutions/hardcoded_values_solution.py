"""
Centralized Configuration System
Single source of truth for all configuration values.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Centralized configuration.
    
    Usage:
        config = Config.load()
        timeout = config.get('timeout', default=30)
    """
    
    _data: Dict[str, Any] = field(default_factory=dict)
    _env_prefix: str = "APP_"
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """Load configuration from file and environment"""
        
        config = cls()
        
        # Load from file
        if config_path and config_path.exists():
            with open(config_path) as f:
                config._data = json.load(f)
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith(config._env_prefix):
                config_key = key[len(config._env_prefix):].lower()
                config._data[config_key] = value
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._data[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple values"""
        self._data.update(updates)
    
    def save(self, config_path: Path):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self._data, f, indent=2)


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config.load()
    return _global_config
