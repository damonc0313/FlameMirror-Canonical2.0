#!/usr/bin/env python3
"""Token bucket rate limiter for API protection."""

import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    tokens_per_second: float = 10.0
    burst_size: int = 20


class RateLimiter:
    """Token bucket algorithm rate limiter."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.tokens = float(self.config.burst_size)
        self.last_update = time.time()
    
    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on rate
        self.tokens = min(
            self.config.burst_size,
            self.tokens + elapsed * self.config.tokens_per_second
        )
        
        self.last_update = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens, return True if successful."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, blocking until available."""
        while not self.try_acquire(tokens):
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def reset(self) -> None:
        """Reset limiter to full capacity."""
        self.tokens = float(self.config.burst_size)
        self.last_update = time.time()


if __name__ == '__main__':
    # Example usage
    limiter = RateLimiter(RateLimitConfig(tokens_per_second=5.0, burst_size=10))
    
    for i in range(15):
        if limiter.try_acquire():
            print(f"Request {i+1}: Allowed")
        else:
            print(f"Request {i+1}: Rate limited")
        time.sleep(0.1)
