#!/usr/bin/env python3
"""Simple logging utility."""

from pathlib import Path
from datetime import datetime
from typing import Optional

class SimpleLogger:
    """Basic file logger."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Write log message."""
        timestamp = datetime.now().isoformat()
        log_line = f"{timestamp} [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def error(self, message: str):
        self.log(message, "ERROR")

if __name__ == '__main__':
    logger = SimpleLogger("app.log")
    logger.info("Application started")
    logger.error("Test error")
    print("Logged 2 messages")
