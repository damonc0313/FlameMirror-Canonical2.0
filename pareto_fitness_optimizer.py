#!/usr/bin/env python3
"""
Autonomous Pareto Fitness Optimization System
Multi-Criteria Decision Making for Code Evolution

This module implements autonomous multi-objective optimization using Pareto 
frontiers for code mutation fitness evaluation without human intervention.
"""

from __future__ import annotations

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum
import time
from datetime import datetime
from pathlib import Path


class OptimizationObjective(Enum):
    """Multi-objective optimization criteria."""
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    MAINTAINABILITY = "maintainability"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SECURITY = "security"
    READABILITY = "readability"


@dataclass
class MultiObjectiveScore:
    """Multi-dimensional fitness score for Pareto analysis."""
    performance: float = 0.0
    correctness: float = 0.0
    maintainability: float = 0.0
    resource_efficiency: float = 0.0
    security: float = 0.0
    readability: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for mathematical operations."""
        return np.array([
            self.performance,
            self.correctness,
            self.maintainability,
            self.resource_efficiency,
            self.security,
            self.readability
        ])
    
    def weighted_sum(self, weights: np.ndarray) -> float:
        """Calculate weighted sum with autonomous weight adaptation."""
        return np.dot(self.to_vector(), weights)


@dataclass
class ParetoSolution:
    """Individual solution in Pareto frontier."""
    mutation_id: str
    scores: MultiObjectiveScore
    rank: int = 0
    crowding_distance: float = 0.0
    domination_count: int = 0
    dominated_solutions: List[str] = field(default_factory=list)
    
    def dominates(self, other: 'ParetoSolution') -> bool:
        """Check if this solution dominates another (autonomous comparison)."""
        self_vector = self.scores.to_vector()
        other_vector = other.scores.to_vector()
        
        # Pareto dominance: all objectives >= other, at least one strictly >
        return (np.all(self_vector >= other_vector) and 
                np.any(self_vector > other_vector))


class AutonomousParetoOptimizer:
    """Autonomous multi-objective optimization using Pareto frontiers."""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.population = []
        self.pareto_frontiers = []
        self.weights = self._initialize_weights_autonomously()
        self.adaptation_history = []
        
    def _initialize_weights_autonomously(self) -> np.ndarray:
        """Autonomously initialize objective weights."""
        # Equal weights initially, will adapt based on success patterns
        num_objectives = len(OptimizationObjective)
        return np.ones(num_objectives) / num_objectives
    
    def evaluate_solution_autonomously(self, 
                                     mutation_result: Any,
                                     repository_metrics: Dict[str, Any]) -> MultiObjectiveScore:
        """Autonomously evaluate solution across multiple objectives."""
        
        # Performance evaluation
        performance = self._evaluate_performance_autonomously(mutation_result)
        
        # Correctness evaluation
        correctness = self._evaluate_correctness_autonomously(mutation_result)
        
        # Maintainability evaluation
        maintainability = self._evaluate_maintainability_autonomously(
            mutation_result, repository_metrics
        )
        
        # Resource efficiency evaluation
        resource_efficiency = self._evaluate_resource_efficiency_autonomously(mutation_result)
        
        # Security evaluation
        security = self._evaluate_security_autonomously(mutation_result)
        
        # Readability evaluation
        readability = self._evaluate_readability_autonomously(mutation_result)
        
        return MultiObjectiveScore(
            performance=performance,
            correctness=correctness,
            maintainability=maintainability,
            resource_efficiency=resource_efficiency,
            security=security,
            readability=readability
        )
    
    def _evaluate_performance_autonomously(self, mutation_result: Any) -> float:
        """Autonomous performance evaluation."""
        try:
            # Extract timing information from mutation result
            base_time = getattr(mutation_result, 'execution_time', 1.0)
            
            # Autonomous performance scoring (lower time = higher score)
            if base_time <= 0:
                return 0.0
            
            # Normalized performance score (0-1 scale)
            max_acceptable_time = 60.0  # 1 minute baseline
            performance_ratio = min(max_acceptable_time / base_time, 1.0)
            
            return performance_ratio
            
        except Exception:
            # Autonomous fallback
            return 0.5
    
    def _evaluate_correctness_autonomously(self, mutation_result: Any) -> float:
        """Autonomous correctness evaluation."""
        try:
            # Basic correctness from mutation success
            if hasattr(mutation_result, 'success') and mutation_result.success:
                base_score = 0.8
            else:
                base_score = 0.2
            
            # Additional correctness factors
            test_results = getattr(mutation_result, 'test_results', {})
            if test_results and not test_results.get('error'):
                base_score = min(1.0, base_score + 0.2)
            
            return base_score
            
        except Exception:
            return 0.0
    
    def _evaluate_maintainability_autonomously(self, 
                                             mutation_result: Any,
                                             repository_metrics: Dict[str, Any]) -> float:
        """Autonomous maintainability evaluation."""
        try:
            # Complexity-based maintainability
            complexity_metrics = repository_metrics.get('complexity_scores', {})
            target_file = getattr(mutation_result, 'target_file', '')
            
            file_metrics = complexity_metrics.get(target_file, {})
            code_complexity = file_metrics.get('code_complexity', 50)
            
            # Lower complexity = higher maintainability
            # Normalize to 0-1 scale
            max_complexity = 200  # Baseline for complex files
            maintainability = max(0.0, 1.0 - (code_complexity / max_complexity))
            
            return maintainability
            
        except Exception:
            return 0.5
    
    def _evaluate_resource_efficiency_autonomously(self, mutation_result: Any) -> float:
        """Autonomous resource efficiency evaluation."""
        try:
            # Memory usage evaluation
            memory_usage = getattr(mutation_result, 'memory_usage', 0.0)
            
            # Execution time factor
            exec_time = getattr(mutation_result, 'execution_time', 1.0)
            
            # Combined efficiency score
            # Lower resource usage = higher efficiency
            time_efficiency = max(0.0, 1.0 - min(exec_time / 10.0, 1.0))
            memory_efficiency = max(0.0, 1.0 - min(memory_usage / 1000.0, 1.0))
            
            return (time_efficiency + memory_efficiency) / 2.0
            
        except Exception:
            return 0.5
    
    def _evaluate_security_autonomously(self, mutation_result: Any) -> float:
        """Autonomous security evaluation."""
        try:
            # Basic security scoring based on mutation type
            mutation_type = getattr(mutation_result, 'mutation_type', None)
            
            if mutation_type:
                type_name = str(mutation_type).lower()
                
                # Security-positive mutations
                if 'logging' in type_name or 'validation' in type_name:
                    return 0.8
                
                # Neutral mutations
                if 'memoization' in type_name or 'caching' in type_name:
                    return 0.6
                
                # Potentially risky mutations
                if 'parallelization' in type_name:
                    return 0.4
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _evaluate_readability_autonomously(self, mutation_result: Any) -> float:
        """Autonomous readability evaluation."""
        try:
            # Readability based on mutation type and success
            mutation_type = getattr(mutation_result, 'mutation_type', None)
            success = getattr(mutation_result, 'success', False)
            
            base_score = 0.6 if success else 0.3
            
            if mutation_type:
                type_name = str(mutation_type).lower()
                
                # Readability-improving mutations
                if 'refactoring' in type_name or 'logging' in type_name:
                    base_score = min(1.0, base_score + 0.3)
                
                # Potentially readability-reducing mutations
                if 'memoization' in type_name or 'caching' in type_name:
                    base_score = max(0.0, base_score - 0.1)
            
            return base_score
            
        except Exception:
            return 0.5
    
    def add_solution(self, mutation_result: Any, repository_metrics: Dict[str, Any]):
        """Autonomously add solution to population."""
        scores = self.evaluate_solution_autonomously(mutation_result, repository_metrics)
        mutation_id = getattr(mutation_result, 'mutation_id', f"mutation_{time.time()}")
        
        solution = ParetoSolution(
            mutation_id=mutation_id,
            scores=scores
        )
        
        self.population.append(solution)
    
    def compute_pareto_frontiers_autonomously(self) -> List[List[ParetoSolution]]:
        """Autonomously compute Pareto frontiers using NSGA-II algorithm."""
        if not self.population:
            return []
        
        # Reset domination information
        for solution in self.population:
            solution.domination_count = 0
            solution.dominated_solutions = []
        
        # Compute domination relationships
        for i, solution_i in enumerate(self.population):
            for j, solution_j in enumerate(self.population):
                if i != j:
                    if solution_i.dominates(solution_j):
                        solution_i.dominated_solutions.append(solution_j.mutation_id)
                    elif solution_j.dominates(solution_i):
                        solution_i.domination_count += 1
        
        # Identify frontiers
        frontiers = []
        current_frontier = []
        
        # First frontier (non-dominated solutions)
        for solution in self.population:
            if solution.domination_count == 0:
                solution.rank = 0
                current_frontier.append(solution)
        
        frontiers.append(current_frontier.copy())
        
        # Subsequent frontiers
        frontier_index = 0
        while current_frontier:
            next_frontier = []
            
            for solution in current_frontier:
                for dominated_id in solution.dominated_solutions:
                    # Find dominated solution
                    dominated_solution = next(
                        (s for s in self.population if s.mutation_id == dominated_id),
                        None
                    )
                    
                    if dominated_solution:
                        dominated_solution.domination_count -= 1
                        if dominated_solution.domination_count == 0:
                            dominated_solution.rank = frontier_index + 1
                            next_frontier.append(dominated_solution)
            
            current_frontier = next_frontier
            if next_frontier:
                frontiers.append(next_frontier.copy())
            frontier_index += 1
        
        # Compute crowding distances
        for frontier in frontiers:
            self._compute_crowding_distance_autonomously(frontier)
        
        self.pareto_frontiers = frontiers
        return frontiers
    
    def _compute_crowding_distance_autonomously(self, frontier: List[ParetoSolution]):
        """Autonomously compute crowding distance for frontier diversity."""
        if len(frontier) <= 2:
            for solution in frontier:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for solution in frontier:
            solution.crowding_distance = 0.0
        
        # Compute distance for each objective
        objectives_count = len(OptimizationObjective)
        
        for obj_index in range(objectives_count):
            # Sort by objective
            frontier.sort(key=lambda s: s.scores.to_vector()[obj_index])
            
            # Boundary solutions get infinite distance
            frontier[0].crowding_distance = float('inf')
            frontier[-1].crowding_distance = float('inf')
            
            # Compute distances for interior solutions
            obj_min = frontier[0].scores.to_vector()[obj_index]
            obj_max = frontier[-1].scores.to_vector()[obj_index]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, len(frontier) - 1):
                    distance = (frontier[i + 1].scores.to_vector()[obj_index] - 
                              frontier[i - 1].scores.to_vector()[obj_index])
                    frontier[i].crowding_distance += distance / obj_range
    
    def select_best_solutions_autonomously(self, count: int) -> List[ParetoSolution]:
        """Autonomously select best solutions using Pareto ranking."""
        if not self.pareto_frontiers:
            self.compute_pareto_frontiers_autonomously()
        
        selected = []
        
        for frontier in self.pareto_frontiers:
            if len(selected) + len(frontier) <= count:
                # Take entire frontier
                selected.extend(frontier)
            else:
                # Partial frontier selection by crowding distance
                remaining = count - len(selected)
                frontier.sort(key=lambda s: s.crowding_distance, reverse=True)
                selected.extend(frontier[:remaining])
                break
        
        return selected
    
    def adapt_weights_autonomously(self, success_feedback: List[Tuple[str, bool]]):
        """Autonomously adapt objective weights based on success patterns."""
        if not success_feedback:
            return
        
        # Analyze successful vs failed mutations
        successful_solutions = []
        failed_solutions = []
        
        for mutation_id, success in success_feedback:
            solution = next((s for s in self.population if s.mutation_id == mutation_id), None)
            if solution:
                if success:
                    successful_solutions.append(solution)
                else:
                    failed_solutions.append(solution)
        
        if successful_solutions and failed_solutions:
            # Compare objective patterns
            successful_means = np.mean([s.scores.to_vector() for s in successful_solutions], axis=0)
            failed_means = np.mean([s.scores.to_vector() for s in failed_solutions], axis=0)
            
            # Adapt weights toward successful patterns
            difference = successful_means - failed_means
            adaptation_rate = 0.1
            
            self.weights += adaptation_rate * difference
            self.weights = np.clip(self.weights, 0.1, 1.0)  # Maintain minimum weight
            self.weights /= np.sum(self.weights)  # Normalize
            
            # Record adaptation
            self.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "new_weights": self.weights.tolist(),
                "successful_count": len(successful_solutions),
                "failed_count": len(failed_solutions)
            })
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate autonomous optimization analysis report."""
        if not self.pareto_frontiers:
            self.compute_pareto_frontiers_autonomously()
        
        report = {
            "optimization_timestamp": datetime.now().isoformat(),
            "total_solutions": len(self.population),
            "pareto_frontiers_count": len(self.pareto_frontiers),
            "current_weights": self.weights.tolist(),
            "adaptation_count": len(self.adaptation_history),
            "frontier_sizes": [len(f) for f in self.pareto_frontiers],
            "best_solutions": []
        }
        
        # Add best solutions from first frontier
        if self.pareto_frontiers:
            first_frontier = self.pareto_frontiers[0]
            for solution in first_frontier[:5]:  # Top 5
                report["best_solutions"].append({
                    "mutation_id": solution.mutation_id,
                    "rank": solution.rank,
                    "crowding_distance": solution.crowding_distance,
                    "scores": {
                        "performance": solution.scores.performance,
                        "correctness": solution.scores.correctness,
                        "maintainability": solution.scores.maintainability,
                        "resource_efficiency": solution.scores.resource_efficiency,
                        "security": solution.scores.security,
                        "readability": solution.scores.readability
                    }
                })
        
        return report


class AutonomousMultiCriteriaDecisionMaker:
    """Autonomous decision making system using multiple criteria."""
    
    def __init__(self):
        self.pareto_optimizer = AutonomousParetoOptimizer(list(OptimizationObjective))
        self.decision_history = []
        self.learning_model = None
    
    def make_autonomous_decision(self, 
                               alternatives: List[Any],
                               repository_metrics: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Make autonomous decision among alternatives."""
        
        # Evaluate all alternatives
        for alternative in alternatives:
            self.pareto_optimizer.add_solution(alternative, repository_metrics)
        
        # Compute Pareto frontiers
        frontiers = self.pareto_optimizer.compute_pareto_frontiers_autonomously()
        
        # Select best solution
        best_solutions = self.pareto_optimizer.select_best_solutions_autonomously(1)
        
        if best_solutions:
            best_solution = best_solutions[0]
            
            # Find corresponding alternative
            selected_alternative = next(
                (alt for alt in alternatives 
                 if getattr(alt, 'mutation_id', '') == best_solution.mutation_id),
                alternatives[0] if alternatives else None
            )
            
            decision_info = {
                "selection_method": "pareto_optimization",
                "rank": best_solution.rank,
                "crowding_distance": best_solution.crowding_distance,
                "scores": best_solution.scores.__dict__,
                "alternatives_count": len(alternatives),
                "frontiers_count": len(frontiers)
            }
            
            # Record decision
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "selected_id": best_solution.mutation_id,
                "decision_info": decision_info
            })
            
            return selected_alternative, decision_info
        
        # Fallback to first alternative
        return alternatives[0] if alternatives else None, {"selection_method": "fallback"}
    
    def update_decision_learning(self, decision_id: str, outcome_success: bool):
        """Update autonomous learning based on decision outcomes."""
        # Find decision in history
        decision_record = next(
            (d for d in self.decision_history if d.get("selected_id") == decision_id),
            None
        )
        
        if decision_record:
            decision_record["outcome_success"] = outcome_success
            decision_record["feedback_timestamp"] = datetime.now().isoformat()
            
            # Update Pareto optimizer weights
            self.pareto_optimizer.adapt_weights_autonomously([(decision_id, outcome_success)])
    
    def export_decision_analytics(self, output_path: Path):
        """Export comprehensive decision analytics."""
        analytics = {
            "decision_count": len(self.decision_history),
            "successful_decisions": len([d for d in self.decision_history 
                                       if d.get("outcome_success", False)]),
            "current_weights": self.pareto_optimizer.weights.tolist(),
            "adaptation_history": self.pareto_optimizer.adaptation_history,
            "optimization_report": self.pareto_optimizer.generate_optimization_report(),
            "decision_history": self.decision_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)


def main():
    """Demonstration of autonomous Pareto optimization."""
    # Mock mutation results for testing
    from types import SimpleNamespace
    
    # Create mock alternatives
    alternatives = [
        SimpleNamespace(
            mutation_id="mut_001",
            mutation_type="memoization",
            target_file="test.py",
            success=True,
            execution_time=0.5,
            memory_usage=100.0
        ),
        SimpleNamespace(
            mutation_id="mut_002", 
            mutation_type="logging_enhancement",
            target_file="test.py",
            success=True,
            execution_time=1.2,
            memory_usage=150.0
        ),
        SimpleNamespace(
            mutation_id="mut_003",
            mutation_type="parallelization", 
            target_file="test.py",
            success=False,
            execution_time=2.0,
            memory_usage=300.0
        )
    ]
    
    repository_metrics = {
        "complexity_scores": {
            "test.py": {
                "code_complexity": 45,
                "line_count": 100
            }
        }
    }
    
    # Create decision maker
    decision_maker = AutonomousMultiCriteriaDecisionMaker()
    
    # Make autonomous decision
    selected, info = decision_maker.make_autonomous_decision(alternatives, repository_metrics)
    
    print("Autonomous Decision Results:")
    print(f"Selected: {selected.mutation_id}")
    print(f"Decision Info: {json.dumps(info, indent=2)}")
    
    # Export analytics
    output_path = Path("autonomous_decision_analytics.json")
    decision_maker.export_decision_analytics(output_path)
    print(f"Analytics exported to: {output_path}")


if __name__ == "__main__":
    main()