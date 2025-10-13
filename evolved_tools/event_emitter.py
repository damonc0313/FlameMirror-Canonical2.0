#!/usr/bin/env python3
"""Event Emitter - Observer pattern implementation."""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Event:
    """Event data structure."""
    name: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

class EventEmitter:
    """
    Event emitter with:
    - Multiple listeners per event
    - Once-only listeners
    - Event history
    - Wildcard listeners
    """
    
    def __init__(self, max_history: int = 100):
        self._listeners: Dict[str, List[Callable]] = {}
        self._once_listeners: Dict[str, List[Callable]] = {}
        self._wildcard_listeners: List[Callable] = []
        self._history: List[Event] = []
        self._max_history = max_history
    
    def on(self, event_name: str, callback: Callable):
        """Register persistent listener."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)
    
    def once(self, event_name: str, callback: Callable):
        """Register one-time listener."""
        if event_name not in self._once_listeners:
            self._once_listeners[event_name] = []
        self._once_listeners[event_name].append(callback)
    
    def on_any(self, callback: Callable):
        """Register wildcard listener (all events)."""
        self._wildcard_listeners.append(callback)
    
    def emit(self, event_name: str, data: Dict[str, Any] = None, source: str = "unknown"):
        """Emit event to all listeners."""
        event = Event(name=event_name, data=data or {}, source=source)
        
        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Call wildcard listeners
        for callback in self._wildcard_listeners:
            try:
                callback(event)
            except Exception as e:
                print(f"Wildcard listener error: {e}")
        
        # Call specific listeners
        if event_name in self._listeners:
            for callback in self._listeners[event_name]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Listener error: {e}")
        
        # Call once listeners and remove
        if event_name in self._once_listeners:
            for callback in self._once_listeners[event_name]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Once listener error: {e}")
            del self._once_listeners[event_name]
    
    def off(self, event_name: str, callback: Callable = None):
        """Remove listener(s)."""
        if callback is None:
            # Remove all listeners for event
            self._listeners.pop(event_name, None)
            self._once_listeners.pop(event_name, None)
        else:
            # Remove specific listener
            if event_name in self._listeners:
                self._listeners[event_name] = [
                    cb for cb in self._listeners[event_name] if cb != callback
                ]
    
    def get_history(self, event_name: str = None) -> List[Event]:
        """Get event history, optionally filtered."""
        if event_name:
            return [e for e in self._history if e.name == event_name]
        return self._history.copy()

if __name__ == '__main__':
    emitter = EventEmitter()
    
    def on_user_action(event):
        print(f"User action: {event.data}")
    
    emitter.on('user_action', on_user_action)
    emitter.emit('user_action', {'action': 'login', 'user': 'alice'})
    print(f"History: {len(emitter.get_history())} events")
