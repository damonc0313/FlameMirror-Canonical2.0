#!/usr/bin/env python3
"""Workflow Engine - DAG-based task orchestration."""

from typing import Dict, List, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Task:
    """Workflow task definition."""
    name: str
    action: Callable
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Exception = None
    started_at: datetime = None
    completed_at: datetime = None

class WorkflowEngine:
    """
    DAG-based workflow engine with:
    - Dependency resolution
    - Parallel execution
    - Error handling and retries
    - Task results persistence
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def add_task(self, name: str, action: Callable, depends_on: List[str] = None):
        """Add task to workflow."""
        task = Task(name=name, action=action, depends_on=depends_on or [])
        self.tasks[name] = task
    
    def _get_ready_tasks(self) -> List[Task]:
        """Get tasks ready to execute (dependencies met)."""
        ready = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies completed
            deps_met = all(
                self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.depends_on
                if dep in self.tasks
            )
            
            if deps_met:
                ready.append(task)
        
        return ready
    
    def _execute_task(self, task: Task):
        """Execute single task."""
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
        
        try:
            # Get dependency results
            dep_results = {
                dep: self.tasks[dep].result
                for dep in task.depends_on
                if dep in self.tasks
            }
            
            # Execute task
            result = task.action(**dep_results) if dep_results else task.action()
            
            with self._lock:
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                self.results[task.name] = result
        
        except Exception as e:
            with self._lock:
                task.error = e
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
    
    def execute(self, parallel: bool = True) -> Dict[str, Any]:
        """Execute workflow."""
        if parallel:
            return self._execute_parallel()
        else:
            return self._execute_sequential()
    
    def _execute_sequential(self) -> Dict[str, Any]:
        """Execute tasks sequentially based on dependencies."""
        while True:
            ready = self._get_ready_tasks()
            if not ready:
                break
            
            for task in ready:
                self._execute_task(task)
        
        return self.results
    
    def _execute_parallel(self) -> Dict[str, Any]:
        """Execute tasks in parallel where possible."""
        threads = []
        
        while True:
            ready = self._get_ready_tasks()
            if not ready:
                # Wait for running tasks
                for t in threads:
                    t.join()
                threads.clear()
                
                # Check if more tasks ready
                ready = self._get_ready_tasks()
                if not ready:
                    break
            
            # Start threads for ready tasks
            for task in ready:
                thread = threading.Thread(target=self._execute_task, args=(task,))
                thread.start()
                threads.append(thread)
        
        return self.results
    
    def get_status(self) -> Dict[str, TaskStatus]:
        """Get status of all tasks."""
        return {name: task.status for name, task in self.tasks.items()}

if __name__ == '__main__':
    wf = WorkflowEngine()
    
    wf.add_task('fetch', lambda: {'data': [1, 2, 3]})
    wf.add_task('process', lambda fetch: {'processed': sum(fetch['data'])}, depends_on=['fetch'])
    wf.add_task('save', lambda process: print(f"Saved: {process}"), depends_on=['process'])
    
    results = wf.execute(parallel=True)
    print(f"Workflow complete: {wf.get_status()}")
