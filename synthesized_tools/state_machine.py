#!/usr/bin/env python3
"""
Generic State Machine with Transition Guards
SYNTHESIZED from learned codebase patterns
"""

from typing import Dict, Callable, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

@dataclass
class Transition:
    """State transition with guard and action."""
    from_state: str
    to_state: str
    trigger: str
    guard: Optional[Callable] = None
    action: Optional[Callable] = None

class StateMachine:
    """
    Generic state machine with:
    - Transition guards
    - Transition actions
    - State entry/exit hooks
    - Transition history
    """
    
    def __init__(self, initial_state: str):
        self.current_state = initial_state
        self.transitions: Dict[str, Dict[str, Transition]] = {}
        self.entry_hooks: Dict[str, Callable] = {}
        self.exit_hooks: Dict[str, Callable] = {}
        self.history: list = [(initial_state, None, None)]
    
    def add_transition(self, transition: Transition):
        """Register a state transition."""
        if transition.from_state not in self.transitions:
            self.transitions[transition.from_state] = {}
        
        self.transitions[transition.from_state][transition.trigger] = transition
    
    def on_enter(self, state: str, callback: Callable):
        """Register entry hook for state."""
        self.entry_hooks[state] = callback
    
    def on_exit(self, state: str, callback: Callable):
        """Register exit hook for state."""
        self.exit_hooks[state] = callback
    
    def trigger(self, event: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger state transition.
        
        Returns True if transition succeeded, False otherwise.
        """
        if self.current_state not in self.transitions:
            return False
        
        if event not in self.transitions[self.current_state]:
            return False
        
        transition = self.transitions[self.current_state][event]
        
        # Check guard
        if transition.guard and not transition.guard(context):
            return False
        
        # Exit current state
        if self.current_state in self.exit_hooks:
            self.exit_hooks[self.current_state](context)
        
        # Execute transition action
        if transition.action:
            transition.action(context)
        
        # Move to new state
        old_state = self.current_state
        self.current_state = transition.to_state
        self.history.append((self.current_state, event, context))
        
        # Enter new state
        if self.current_state in self.entry_hooks:
            self.entry_hooks[self.current_state](context)
        
        return True
    
    def can_trigger(self, event: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if event can be triggered from current state."""
        if self.current_state not in self.transitions:
            return False
        
        if event not in self.transitions[self.current_state]:
            return False
        
        transition = self.transitions[self.current_state][event]
        
        if transition.guard:
            return transition.guard(context)
        
        return True
    
    def get_available_transitions(self) -> Set[str]:
        """Get all available transitions from current state."""
        if self.current_state not in self.transitions:
            return set()
        
        return set(self.transitions[self.current_state].keys())

if __name__ == '__main__':
    # Example: Traffic light
    sm = StateMachine('red')
    
    sm.add_transition(Transition('red', 'green', 'next'))
    sm.add_transition(Transition('green', 'yellow', 'next'))
    sm.add_transition(Transition('yellow', 'red', 'next'))
    
    print(f"Initial: {sm.current_state}")
    sm.trigger('next')
    print(f"After next: {sm.current_state}")
