#!/usr/bin/env python3
"""Distributed logging with remote aggregation and buffering."""

import os
import gzip
import json
import queue
import socket
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    context: Dict
    hostname: str
    process_id: int

class DistributedLogger:
    """Production distributed logger with buffering and remote shipping."""
    
    def __init__(
        self,
        log_file: str,
        max_size: int = 10_000_000,
        backup_count: int = 5,
        level: LogLevel = LogLevel.INFO,
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        buffer_size: int = 1000,
        flush_interval: float = 5.0
    ):
        self.log_file = Path(log_file)
        self.max_size = max_size
        self.backup_count = backup_count
        self.level = level
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        
        # Buffering
        self.buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def _should_rotate(self) -> bool:
        if not self.log_file.exists():
            return False
        return self.log_file.stat().st_size >= self.max_size
    
    def _rotate(self):
        """Rotate and compress log files."""
        for i in range(self.backup_count - 1, 0, -1):
            old = Path(f"{self.log_file}.{i}.gz")
            new = Path(f"{self.log_file}.{i+1}.gz")
            if old.exists():
                old.rename(new)
        
        if self.log_file.exists():
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(f"{self.log_file}.1.gz", 'wb', compresslevel=6) as f_out:
                    f_out.write(f_in.read())
            self.log_file.unlink()
    
    def _ship_remote(self, entry: LogEntry):
        """Ship log entry to remote aggregator."""
        if not self.remote_host or not self.remote_port:
            return
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.connect((self.remote_host, self.remote_port))
                data = json.dumps(asdict(entry)).encode() + b'\n'
                sock.sendall(data)
        except (socket.error, socket.timeout):
            pass  # Failed to ship, local log still has it
    
    def _flush_worker(self):
        """Background thread to flush buffer."""
        while True:
            entries: List[LogEntry] = []
            
            # Collect entries for flush_interval seconds
            deadline = datetime.now().timestamp() + self.flush_interval
            while datetime.now().timestamp() < deadline:
                try:
                    entry = self.buffer.get(timeout=0.1)
                    entries.append(entry)
                except queue.Empty:
                    continue
            
            # Flush to disk and remote
            if entries:
                self._flush_entries(entries)
    
    def _flush_entries(self, entries: List[LogEntry]):
        """Write buffered entries."""
        if self._should_rotate():
            self._rotate()
        
        with open(self.log_file, 'a') as f:
            for entry in entries:
                context_str = f" {entry.context}" if entry.context else ""
                log_line = f"{entry.timestamp} [{entry.level}] {entry.hostname}:{entry.process_id}{context_str} {entry.message}\n"
                f.write(log_line)
        
        # Ship to remote
        for entry in entries:
            self._ship_remote(entry)
    
    def log(self, message: str, level: LogLevel, context: Optional[Dict] = None):
        """Buffer log entry."""
        if level.value < self.level.value:
            return
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            message=message,
            context=context or {},
            hostname=self.hostname,
            process_id=self.process_id
        )
        
        try:
            self.buffer.put_nowait(entry)
        except queue.Full:
            # Force flush if buffer full
            self._flush_entries([entry])
    
    def debug(self, message: str, **context):
        self.log(message, LogLevel.DEBUG, context or None)
    
    def info(self, message: str, **context):
        self.log(message, LogLevel.INFO, context or None)
    
    def warning(self, message: str, **context):
        self.log(message, LogLevel.WARNING, context or None)
    
    def error(self, message: str, **context):
        self.log(message, LogLevel.ERROR, context or None)
    
    def critical(self, message: str, **context):
        self.log(message, LogLevel.CRITICAL, context or None)

if __name__ == '__main__':
    logger = DistributedLogger(
        "distributed.log",
        max_size=5000,
        buffer_size=10,
        flush_interval=2.0
    )
    
    logger.info("Distributed system started", cluster="prod", region="us-east")
    logger.warning("High latency", service="api", latency_ms=250)
    logger.error("Replication lag", database="primary", lag_seconds=30)
    
    import time
    time.sleep(3)  # Let buffer flush
    print("Distributed logging with buffering complete")
