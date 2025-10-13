#!/usr/bin/env python3
"""
Dynamic Plugin System with Hot Reload
SYNTHESIZED from learned codebase patterns
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None

class Plugin(ABC):
    """Base class for plugins."""
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with context."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin logic."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup plugin resources."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        pass

class PluginManager:
    """
    Plugin manager with:
    - Dynamic plugin discovery
    - Hot reload support
    - Dependency resolution
    - Plugin lifecycle management
    """
    
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
    
    def discover_plugins(self):
        """Discover all plugins in plugin directory."""
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True)
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.stem.startswith('_'):
                continue
            
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                print(f"Failed to load {plugin_file}: {e}")
    
    def _load_plugin(self, plugin_file: Path):
        """Load a single plugin."""
        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find Plugin subclasses
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj != Plugin:
                plugin_instance = obj()
                self.plugins[plugin_instance.metadata.name] = plugin_instance
                print(f"Loaded plugin: {plugin_instance.metadata.name}")
    
    def initialize_all(self, context: Dict[str, Any]):
        """Initialize all loaded plugins."""
        for name, plugin in self.plugins.items():
            try:
                plugin.initialize(context)
            except Exception as e:
                print(f"Failed to initialize {name}: {e}")
    
    def get_plugin(self, name: str) -> Plugin:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def register_hook(self, event: str, callback: Callable):
        """Register hook for plugin events."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs):
        """Trigger all hooks for event."""
        if event in self.hooks:
            for callback in self.hooks[event]:
                callback(*args, **kwargs)
    
    def reload_plugin(self, name: str):
        """Hot reload a specific plugin."""
        if name in self.plugins:
            self.plugins[name].cleanup()
            del self.plugins[name]
        
        # Reload from disk
        self.discover_plugins()

if __name__ == '__main__':
    manager = PluginManager(Path('./plugins'))
    manager.discover_plugins()
    print(f"Loaded {len(manager.plugins)} plugins")
