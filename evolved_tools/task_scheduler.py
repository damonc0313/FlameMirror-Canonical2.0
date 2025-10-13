#!/usr/bin/env python3
"""Task Scheduler - Cron-like scheduling system."""

import time
import threading
from datetime import datetime, timedelta
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum

class ScheduleType(Enum):
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"

@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    name: str
    action: Callable
    schedule_type: ScheduleType
    next_run: datetime
    interval_seconds: Optional[float] = None
    last_run: Optional[datetime] = None
    run_count: int = 0

class TaskScheduler:
    """
    Production task scheduler with:
    - One-time scheduling
    - Interval-based repeating
    - Daily execution at specific time
    - Task history
    - Thread-safe execution
    """
    
    def __init__(self):
        self.tasks: List[ScheduledTask] = []
        self.running = False
        self._lock = threading.Lock()
        self._scheduler_thread = None
    
    def schedule_once(self, name: str, action: Callable, run_at: datetime):
        """Schedule task to run once at specific time."""
        task = ScheduledTask(
            name=name,
            action=action,
            schedule_type=ScheduleType.ONCE,
            next_run=run_at
        )
        with self._lock:
            self.tasks.append(task)
    
    def schedule_interval(self, name: str, action: Callable, interval_seconds: float):
        """Schedule task to repeat at interval."""
        task = ScheduledTask(
            name=name,
            action=action,
            schedule_type=ScheduleType.INTERVAL,
            next_run=datetime.now() + timedelta(seconds=interval_seconds),
            interval_seconds=interval_seconds
        )
        with self._lock:
            self.tasks.append(task)
    
    def schedule_daily(self, name: str, action: Callable, hour: int, minute: int = 0):
        """Schedule task to run daily at specific time."""
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        if next_run <= now:
            next_run += timedelta(days=1)
        
        task = ScheduledTask(
            name=name,
            action=action,
            schedule_type=ScheduleType.DAILY,
            next_run=next_run,
            interval_seconds=86400  # 24 hours
        )
        with self._lock:
            self.tasks.append(task)
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            now = datetime.now()
            
            with self._lock:
                for task in self.tasks[:]:  # Copy to allow modification
                    if task.next_run <= now:
                        # Execute task
                        self._execute_task(task)
                        
                        # Schedule next run
                        if task.schedule_type == ScheduleType.ONCE:
                            self.tasks.remove(task)
                        elif task.schedule_type in [ScheduleType.INTERVAL, ScheduleType.DAILY]:
                            task.next_run = now + timedelta(seconds=task.interval_seconds)
            
            time.sleep(0.1)  # Check every 100ms
    
    def _execute_task(self, task: ScheduledTask):
        """Execute scheduled task."""
        try:
            task.action()
            task.last_run = datetime.now()
            task.run_count += 1
        except Exception as e:
            print(f"Task {task.name} failed: {e}")
    
    def start(self):
        """Start scheduler."""
        if self.running:
            return
        
        self.running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
    
    def stop(self):
        """Stop scheduler."""
        self.running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=1.0)
    
    def get_status(self) -> List[Dict]:
        """Get status of all scheduled tasks."""
        with self._lock:
            return [
                {
                    'name': task.name,
                    'type': task.schedule_type.value,
                    'next_run': task.next_run.isoformat(),
                    'run_count': task.run_count,
                    'last_run': task.last_run.isoformat() if task.last_run else None
                }
                for task in self.tasks
            ]

if __name__ == '__main__':
    scheduler = TaskScheduler()
    
    def job():
        print(f"Job executed at {datetime.now()}")
    
    scheduler.schedule_interval('test_job', job, interval_seconds=1.0)
    scheduler.start()
    
    time.sleep(3)
    print(f"Status: {scheduler.get_status()}")
    scheduler.stop()
