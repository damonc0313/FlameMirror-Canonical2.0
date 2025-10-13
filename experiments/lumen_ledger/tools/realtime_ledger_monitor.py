#!/usr/bin/env python3
"""Real-Time Ledger Monitor - Live updates, auto-synthesis, instant notifications"""
import json
import time
from datetime import datetime
from pathlib import Path

class RealtimeLedgerMonitor:
    def __init__(self, ledger_path, poll_seconds=30):
        self.ledger_path = Path(ledger_path)
        self.poll_seconds = poll_seconds
        self.last_count = 0
        self.callbacks = []
    
    def register_callback(self, callback):
        """Register function to call on new entry"""
        self.callbacks.append(callback)
    
    def check_for_updates(self):
        """Check if new entries added"""
        with open(self.ledger_path) as f:
            entries = json.load(f)
        current_count = len(entries)
        
        if current_count > self.last_count:
            new_count = current_count - self.last_count
            self.last_count = current_count
            
            # Trigger callbacks
            for callback in self.callbacks:
                callback({'new_entries': new_count, 'total': current_count})
            
            return new_count
        return 0
    
    def run_monitor(self, duration_minutes=60):
        """Run monitoring loop"""
        print(f"Monitoring ledger for {duration_minutes} minutes...")
        start = datetime.now()
        
        while (datetime.now() - start).seconds < duration_minutes * 60:
            new = self.check_for_updates()
            if new > 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {new} new entries detected")
            time.sleep(self.poll_seconds)

if __name__ == '__main__':
    monitor = RealtimeLedgerMonitor('../../ledger/index.json', poll_seconds=10)
    monitor.register_callback(lambda update: print(f"✓ Update: {update}"))
    print("✓ Real-time monitor ready")
