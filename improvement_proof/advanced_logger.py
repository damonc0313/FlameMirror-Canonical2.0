#!/usr/bin/env python3
"""Advanced logging with rotation and formatting."""

import os
import gzip
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class AdvancedLogger:
    """Logger with rotation, levels, and formatting."""
    
    def __init__(
        self,
        log_file: str,
        max_size: int = 10_000_000,  # 10MB
        backup_count: int = 5,
        level: LogLevel = LogLevel.INFO
    ):
        self.log_file = Path(log_file)
        self.max_size = max_size
        self.backup_count = backup_count
        self.level = level
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _should_rotate(self) -> bool:
        """Check if log file needs rotation."""
        if not self.log_file.exists():
            return False
        return self.log_file.stat().st_size >= self.max_size
    
    def _rotate(self):
        """Rotate log files, compress old ones."""
        # Shift existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_file = Path(f"{self.log_file}.{i}.gz")
            new_file = Path(f"{self.log_file}.{i+1}.gz")
            if old_file.exists():
                old_file.rename(new_file)
        
        # Compress current log
        if self.log_file.exists():
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(f"{self.log_file}.1.gz", 'wb') as f_out:
                    f_out.write(f_in.read())
            self.log_file.unlink()
    
    def log(self, message: str, level: LogLevel, context: Optional[Dict] = None):
        """Write formatted log message."""
        if level.value < self.level.value:
            return
        
        if self._should_rotate():
            self._rotate()
        
        timestamp = datetime.now().isoformat()
        context_str = f" {context}" if context else ""
        log_line = f"{timestamp} [{level.name}]{context_str} {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
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
    logger = AdvancedLogger("app.log", max_size=1000, backup_count=3)
    logger.info("System started", version="1.0")
    logger.warning("High memory usage", memory_mb=512)
    logger.error("Connection failed", host="db.example.com")
    print("Logged with rotation and context")
