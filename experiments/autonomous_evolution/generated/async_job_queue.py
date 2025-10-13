#!/usr/bin/env python3
"""
ASYNC JOB QUEUE - Iteration 2.3
================================
Background job processing with retry, scheduling, priorities
"""

import asyncio
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import heapq

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Job:
    job_id: str
    task_name: str
    payload: Dict[str, Any]
    priority: int = 5
    max_retries: int = 3
    retry_count: int = 0
    status: JobStatus = JobStatus.PENDING
    scheduled_at: Optional[datetime] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def __lt__(self, other):
        # Higher priority (lower number) first, then earlier scheduled
        if self.priority != other.priority:
            return self.priority < other.priority
        return (self.scheduled_at or datetime.min) < (other.scheduled_at or datetime.min)

class JobQueue:
    """Async job queue with priorities and scheduling"""
    
    def __init__(self, workers: int = 5):
        self.workers = workers
        self.tasks: Dict[str, Callable] = {}
        self.queue: List[Job] = []
        self.jobs: Dict[str, Job] = {}
        self.running = False
        
    def register_task(self, name: str, handler: Callable):
        """Register task handler"""
        self.tasks[name] = handler
        
    def enqueue(self, job: Job):
        """Add job to queue"""
        if not job.created_at:
            job.created_at = datetime.now(timezone.utc)
        self.jobs[job.job_id] = job
        heapq.heappush(self.queue, job)
        
    async def start(self):
        """Start processing jobs"""
        self.running = True
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.workers)
        ]
        await asyncio.gather(*workers)
        
    async def _worker(self, worker_id: int):
        """Job worker"""
        while self.running:
            if not self.queue:
                await asyncio.sleep(0.1)
                continue
                
            job = heapq.heappop(self.queue)
            
            # Check if scheduled for future
            if job.scheduled_at and job.scheduled_at > datetime.now(timezone.utc):
                heapq.heappush(self.queue, job)  # Put back
                await asyncio.sleep(1)
                continue
            
            # Execute job
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            
            try:
                handler = self.tasks.get(job.task_name)
                if handler:
                    await handler(job.payload)
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
                    
            except Exception as e:
                job.error = str(e)
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    job.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=2 ** job.retry_count)
                    heapq.heappush(self.queue, job)
                else:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)

# 120 lines - Focused job queue
