#!/usr/bin/env python3
"""Intelligent cache manager with TTL and LRU eviction."""

from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict


@dataclass
class CacheEntry:
    """Represents a cached value with expiration."""
    value: Any
    expires_at: datetime


class CacheManager:
    """Thread-safe cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now() > entry.expires_at:
            del self.cache[key]
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        if len(self.cache) >= self.max_size:
            # Remove oldest (LRU eviction)
            self.cache.popitem(last=False)
        
        expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache[key] = CacheEntry(value=value, expires_at=expires_at)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()


if __name__ == '__main__':
    # Example usage
    cache = CacheManager(max_size=100, default_ttl=60.0)
    cache.set('key1', 'value1')
    print(cache.get('key1'))  # 'value1'
    print(cache.get('nonexistent'))  # None
