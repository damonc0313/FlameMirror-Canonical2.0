#!/usr/bin/env python3
"""Exponential backoff retry mechanism for resilient systems."""

import time
import random
from typing import Callable, Any, Type
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryHandler:
    """Handles retries with exponential backoff and jitter."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def execute(
        self,
        func: Callable,
        *args,
        retry_on: Type[Exception] = Exception,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except retry_on as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    time.sleep(delay)
        
        # All retries exhausted
        raise last_exception


if __name__ == '__main__':
    # Example usage
    handler = RetryHandler()
    
    def flaky_function():
        import random
        if random.random() < 0.7:
            raise ValueError("Temporary failure")
        return "Success!"
    
    try:
        result = handler.execute(flaky_function)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Failed after retries: {e}")
