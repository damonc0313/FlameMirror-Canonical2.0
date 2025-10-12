#!/usr/bin/env python3
"""
Multi-Agent Coordination Protocol (MACP)
Enables systematic collaboration between multiple Claude instances via D1 database

Implements Enhancement Pathway #7: Multi-Agent Collaboration

Key Features:
- Task queue system (propose, claim, complete workflow)
- Work-in-progress signaling
- Capability-based task assignment
- Conflict resolution via timestamps
- Collaborative synthesis coordination
- Research agenda management
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class TaskStatus(Enum):
    """Task lifecycle states"""
    PROPOSED = "proposed"      # Identified, not yet claimed
    CLAIMED = "claimed"        # Instance working on it
    BLOCKED = "blocked"        # Waiting on dependency
    COMPLETED = "completed"    # Finished
    VALIDATED = "validated"    # Peer-reviewed


class TaskPriority(Enum):
    """Task urgency levels"""
    CRITICAL = 5  # Infrastructure failure, blocking research
    HIGH = 4      # Core research advancement
    MEDIUM = 3    # Enhancement opportunities
    LOW = 2       # Nice-to-have improvements
    BACKLOG = 1   # Future consideration


@dataclass
class Task:
    """Research task for multi-agent coordination"""
    task_id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    proposed_by: str
    proposed_at: datetime
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    required_capability: Optional[str] = None  # e.g., "infrastructure-debugging"
    depends_on: List[str] = None  # task_ids this depends on
    estimated_entries: int = 1  # Expected ledger entries to complete
    related_hypothesis: Optional[str] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


class MultiAgentCoordinator:
    """
    Coordinates multiple Claude instances working on shared research
    via D1 database task queue
    """
    
    def __init__(self, author: str):
        """
        Initialize coordinator for specific author/instance
        
        Args:
            author: Instance identifier (e.g., 'kael', 'lumen', 'dale')
        """
        self.author = author
        self.tasks: List[Task] = []
        self.author_capabilities: Dict[str, List[str]] = {
            'kael': ['recursive-analysis', 'meta-enhancement-design', 
                     'automated-tool-creation', 'predictive-enhancement'],
            'lumen': ['autonomous-planning', 'infrastructure-organization',
                      'forecasting-analysis', 'self-referential-analysis'],
            'dale': ['infrastructure-debugging', 'documentation-synthesis',
                     'predictive-validation', 'empirical-testing']
        }
    
    def propose_task(
        self, 
        title: str, 
        description: str,
        priority: TaskPriority,
        required_capability: Optional[str] = None,
        depends_on: List[str] = None,
        estimated_entries: int = 1,
        related_hypothesis: Optional[str] = None
    ) -> Task:
        """
        Propose new research task for coordination
        
        Returns: Task object ready for database insertion
        """
        task_id = f"TASK-{datetime.utcnow().strftime('%Y%m%d-%H%M')}-{self.author}"
        
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            status=TaskStatus.PROPOSED,
            priority=priority,
            proposed_by=self.author,
            proposed_at=datetime.utcnow(),
            required_capability=required_capability,
            depends_on=depends_on or [],
            estimated_entries=estimated_entries,
            related_hypothesis=related_hypothesis
        )
        
        self.tasks.append(task)
        return task
    
    def find_suitable_tasks(self) -> List[Task]:
        """
        Find tasks this instance is well-suited to complete
        based on capability matching
        
        Returns: Prioritized list of suitable tasks
        """
        my_capabilities = self.author_capabilities.get(self.author, [])
        
        suitable = []
        for task in self.tasks:
            # Only consider proposed (unclaimed) tasks
            if task.status != TaskStatus.PROPOSED:
                continue
            
            # Check if dependencies are met
            if task.depends_on:
                deps_met = all(
                    self._is_task_completed(dep) 
                    for dep in task.depends_on
                )
                if not deps_met:
                    continue
            
            # Check capability match
            if task.required_capability:
                # Exact match or similar capability
                matches = any(
                    task.required_capability in cap or 
                    cap in task.required_capability
                    for cap in my_capabilities
                )
                if matches:
                    suitable.append(task)
            else:
                # No specific capability required - anyone can do it
                suitable.append(task)
        
        # Sort by priority then proposal time
        suitable.sort(
            key=lambda t: (-t.priority.value, t.proposed_at)
        )
        
        return suitable
    
    def claim_task(self, task: Task) -> Tuple[bool, str]:
        """
        Attempt to claim a task (with conflict detection)
        
        Returns: (success, message)
        """
        # Check if already claimed by someone else
        if task.claimed_by and task.claimed_by != self.author:
            # Check if claim is stale (>2 hours with no update)
            if task.claimed_at:
                elapsed = datetime.utcnow() - task.claimed_at
                if elapsed < timedelta(hours=2):
                    return False, f"Task claimed by {task.claimed_by}"
                else:
                    # Stale claim - can reclaim
                    pass
        
        # Claim it
        task.status = TaskStatus.CLAIMED
        task.claimed_by = self.author
        task.claimed_at = datetime.utcnow()
        
        return True, f"Task claimed successfully"
    
    def complete_task(self, task: Task, evidence: str) -> None:
        """
        Mark task as completed with evidence
        """
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        # Evidence would be stored in related ledger entry
    
    def _is_task_completed(self, task_id: str) -> bool:
        """Check if a task dependency is completed"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task.status in [TaskStatus.COMPLETED, TaskStatus.VALIDATED]
        return False
    
    def generate_work_report(self) -> str:
        """
        Generate status report for coordination
        """
        lines = ["=" * 70]
        lines.append(f"MULTI-AGENT COORDINATION STATUS - {self.author.upper()}")
        lines.append("=" * 70)
        lines.append("")
        
        # My active tasks
        my_active = [t for t in self.tasks if t.claimed_by == self.author 
                     and t.status == TaskStatus.CLAIMED]
        if my_active:
            lines.append(f"MY ACTIVE TASKS ({len(my_active)}):")
            for task in my_active:
                elapsed = datetime.utcnow() - task.claimed_at
                lines.append(f"  [{task.priority.name}] {task.title}")
                lines.append(f"    Status: In progress ({elapsed.seconds//60}m)")
                lines.append("")
        
        # Available tasks for me
        suitable = self.find_suitable_tasks()
        if suitable:
            lines.append(f"SUITABLE AVAILABLE TASKS ({len(suitable)}):")
            for task in suitable[:5]:  # Top 5
                lines.append(f"  [{task.priority.name}] {task.title}")
                lines.append(f"    Proposed by: {task.proposed_by}")
                if task.required_capability:
                    lines.append(f"    Requires: {task.required_capability}")
                lines.append("")
        
        # Overall task queue status
        by_status = {}
        for task in self.tasks:
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        lines.append("OVERALL QUEUE STATUS:")
        for status, count in sorted(by_status.items()):
            lines.append(f"  {status}: {count}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


if __name__ == '__main__':
    # Test with realistic research tasks
    kael = MultiAgentCoordinator('kael')
    
    # Propose actual tasks from current research agenda
    task1 = kael.propose_task(
        title="Validate confidence calibration accuracy",
        description="Test confidence_calibrator.py with real ledger data to measure "
                   "prediction accuracy and calibration gaps",
        priority=TaskPriority.HIGH,
        required_capability="automated-tool-creation",
        estimated_entries=1,
        related_hypothesis="recursive-self-improvement"
    )
    
    task2 = kael.propose_task(
        title="Build visualization dashboard",
        description="Create interactive HTML dashboard showing ledger analytics, "
                   "confidence calibration, and research velocity metrics",
        priority=TaskPriority.MEDIUM,
        required_capability="autonomous-planning",
        estimated_entries=2
    )
    
    print(kael.generate_work_report())
