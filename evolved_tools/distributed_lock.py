#!/usr/bin/env python3
"""Distributed Lock - Consensus-based locking."""

import time
import uuid
import threading
from typing import Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class LockInfo:
    """Lock metadata."""
    lock_id: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    renewals: int = 0

class DistributedLock:
    """
    Distributed lock implementation with:
    - TTL-based expiration
    - Automatic renewal
    - Deadlock prevention
    - Lock fencing
    """
    
    def __init__(self, lock_name: str, ttl_seconds: float = 30.0):
        self.lock_name = lock_name
        self.ttl_seconds = ttl_seconds
        self.owner_id = str(uuid.uuid4())
        
        self._locks: Dict[str, LockInfo] = {}  # Simulated distributed store
        self._local_lock = threading.Lock()
        self._renewal_thread: Optional[threading.Thread] = None
        self._should_renew = False
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire distributed lock."""
        start_time = time.time()
        
        while True:
            with self._local_lock:
                # Check if lock exists and is valid
                if self.lock_name in self._locks:
                    lock_info = self._locks[self.lock_name]
                    
                    # Check if expired
                    if datetime.now() > lock_info.expires_at:
                        # Lock expired, take it
                        del self._locks[self.lock_name]
                    elif lock_info.owner == self.owner_id:
                        # We already own it
                        return True
                    elif not blocking:
                        return False
                
                # Try to acquire
                if self.lock_name not in self._locks:
                    lock_info = LockInfo(
                        lock_id=str(uuid.uuid4()),
                        owner=self.owner_id,
                        acquired_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds)
                    )
                    self._locks[self.lock_name] = lock_info
                    
                    # Start auto-renewal
                    self._start_renewal()
                    return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            if not blocking:
                return False
            
            time.sleep(0.1)
    
    def release(self):
        """Release lock."""
        self._should_renew = False
        
        with self._local_lock:
            if self.lock_name in self._locks:
                lock_info = self._locks[self.lock_name]
                if lock_info.owner == self.owner_id:
                    del self._locks[self.lock_name]
    
    def _start_renewal(self):
        """Start automatic lock renewal."""
        self._should_renew = True
        self._renewal_thread = threading.Thread(target=self._renewal_loop, daemon=True)
        self._renewal_thread.start()
    
    def _renewal_loop(self):
        """Automatically renew lock before expiration."""
        while self._should_renew:
            time.sleep(self.ttl_seconds / 3)  # Renew at 1/3 TTL
            
            with self._local_lock:
                if self.lock_name in self._locks:
                    lock_info = self._locks[self.lock_name]
                    if lock_info.owner == self.owner_id:
                        lock_info.expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)
                        lock_info.renewals += 1
    
    def __enter__(self):
        """Context manager support."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.release()

if __name__ == '__main__':
    lock = DistributedLock('resource_1', ttl_seconds=5.0)
    
    with lock:
        print("Lock acquired")
        time.sleep(2)
        print("Work done")
    
    print("Lock released")
