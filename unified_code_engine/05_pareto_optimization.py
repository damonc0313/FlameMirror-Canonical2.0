"""
Cell 5: Pareto Fitness Optimization System
===========================================

Multi-objective optimization using Pareto frontiers for autonomous
code evolution without human intervention. This system optimizes
code across multiple conflicting objectives simultaneously.

Objectives:
- Performance (speed, efficiency)
- Correctness (test passing, error-free)
- Maintainability (readability, structure)
- Resource Efficiency (memory, CPU usage)
- Security (vulnerability resistance)
- Readability (documentation, clarity)
"""

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
        """Calculate weighted sum of objectives."""
        return np.dot(self.to_vector(), weights)
    
    def dominates(self, other: 'MultiObjectiveScore') -> bool:
        """Check if this score dominates another (Pareto dominance)."""
        self_vec = self.to_vector()
        other_vec = other.to_vector()
        return np.all(self_vec >= other_vec) and np.any(self_vec > other_vec)


@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier."""
    code: str
    objectives: MultiObjectiveScore
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_solutions: List[str] = field(default_factory=list)


class ParetoFitnessOptimizer:
    """Multi-objective Pareto optimization for code evolution."""
    
    def __init__(self):
        self.pareto_front = []
        self.dominated_solutions = []
        self.objective_weights = self._initialize_adaptive_weights()
        self.optimization_history = []
        
    def _initialize_adaptive_weights(self) -> np.ndarray:
        """Initialize adaptive weights for objectives."""
        # Equal weights initially, will adapt based on success rates
        return np.ones(6) / 6
    
    def evaluate_multi_objective_fitness(self, code: str) -> MultiObjectiveScore:
        """Comprehensive multi-objective fitness evaluation."""
        
        # Performance evaluation
        performance = self._evaluate_performance(code)
        
        # Correctness evaluation
        correctness = self._evaluate_correctness(code)
        
        # Maintainability evaluation
        maintainability = self._evaluate_maintainability(code)
        
        # Resource efficiency evaluation
        resource_efficiency = self._evaluate_resource_efficiency(code)
        
        # Security evaluation
        security = self._evaluate_security(code)
        
        # Readability evaluation
        readability = self._evaluate_readability(code)
        
        return MultiObjectiveScore(
            performance=performance,
            correctness=correctness,
            maintainability=maintainability,
            resource_efficiency=resource_efficiency,
            security=security,
            readability=readability
        )
    
    def _evaluate_performance(self, code: str) -> float:
        """Evaluate code performance characteristics."""
        score = 0.5  # Base score
        
        # Algorithmic complexity analysis
        if "O(n)" in code or "linear" in code.lower():
            score += 0.2
        elif "O(log n)" in code or "logarithmic" in code.lower():
            score += 0.3
        elif "O(n²)" in code or "quadratic" in code.lower():
            score -= 0.2
        
        # Performance patterns
        performance_patterns = {
            "numpy": 0.15,
            "vectorized": 0.1,
            "cache": 0.1,
            "memoize": 0.1,
            "parallel": 0.15,
            "async": 0.1,
            "generator": 0.05,
            "lazy": 0.05
        }
        
        for pattern, bonus in performance_patterns.items():
            if pattern in code.lower():
                score += bonus
        
        # Performance anti-patterns
        anti_patterns = {
            "nested loops": -0.1,
            "global": -0.05,
            "recursive": -0.05,  # Can be inefficient if not optimized
        }
        
        for pattern, penalty in anti_patterns.items():
            if pattern in code.lower():
                score += penalty
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_correctness(self, code: str) -> float:
        """Evaluate code correctness."""
        score = 0.5
        
        try:
            # Syntax check
            ast.parse(code)
            score += 0.3
        except SyntaxError:
            return 0.0
        
        # Error handling patterns
        if "try:" in code and "except" in code:
            score += 0.1
        if "raise" in code:
            score += 0.05
        if "assert" in code:
            score += 0.05
        
        # Type hints
        if ":" in code and "->" in code:
            score += 0.1
        
        # Input validation
        if "validate" in code.lower() or "check" in code.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_maintainability(self, code: str) -> float:
        """Evaluate code maintainability."""
        score = 0.5
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Function length
        if len(non_empty_lines) <= 20:
            score += 0.1
        elif len(non_empty_lines) > 50:
            score -= 0.1
        
        # Complexity indicators
        cyclomatic_complexity = self._estimate_cyclomatic_complexity(code)
        if cyclomatic_complexity <= 10:
            score += 0.1
        elif cyclomatic_complexity > 20:
            score -= 0.2
        
        # Naming conventions
        if any(char.isupper() for char in code if char.isalpha()):
            score += 0.05  # CamelCase or constants
        
        # Modularity
        if "class" in code:
            score += 0.1
        if "def " in code:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_resource_efficiency(self, code: str) -> float:
        """Evaluate resource efficiency."""
        score = 0.5
        
        # Memory efficiency patterns
        if "generator" in code or "yield" in code:
            score += 0.2
        if "del " in code:
            score += 0.05
        if "gc.collect" in code:
            score += 0.05
        
        # Memory inefficiency patterns
        if "list(" in code and "range(" in code:
            score -= 0.1  # Could use generator instead
        if "*" in code and "[" in code:  # List multiplication
            score -= 0.05
        
        # CPU efficiency
        if "multiprocessing" in code or "threading" in code:
            score += 0.1
        if "pool" in code.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_security(self, code: str) -> float:
        """Evaluate code security."""
        score = 0.8  # Start high, penalize for issues
        
        # Security vulnerabilities
        vulnerabilities = {
            "eval(": -0.3,
            "exec(": -0.3,
            "os.system(": -0.2,
            "subprocess.call(": -0.1,
            "__import__": -0.2,
            "pickle.load": -0.1,
            "input(": -0.05,  # Potential for injection
        }
        
        for vuln, penalty in vulnerabilities.items():
            if vuln in code:
                score += penalty
        
        # Security best practices
        if "sanitize" in code.lower():
            score += 0.1
        if "escape" in code.lower():
            score += 0.05
        if "validate" in code.lower():
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability."""
        score = 0.5
        
        lines = code.split('\n')
        
        # Documentation
        if '"""' in code or "'''" in code:
            score += 0.2
        
        # Comments
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if len(comment_lines) > 0:
            comment_ratio = len(comment_lines) / max(1, len(lines))
            score += min(0.2, comment_ratio * 2)
        
        # Spacing and formatting
        if any("  " in line for line in lines):  # Proper indentation
            score += 0.1
        
        # Variable naming
        if any(len(word) > 3 for word in code.split() if word.isidentifier()):
            score += 0.1  # Descriptive names
        
        return max(0.0, min(1.0, score))
    
    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        # Decision points
        decision_keywords = ['if', 'elif', 'while', 'for', 'except', 'and', 'or']
        for keyword in decision_keywords:
            complexity += code.lower().count(keyword)
        
        return complexity
    
    def update_pareto_front(self, solution: ParetoSolution):
        """Update the Pareto front with a new solution."""
        # Check if solution is dominated by any existing solution
        is_dominated = False
        for existing in self.pareto_front:
            if existing.objectives.dominates(solution.objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove solutions dominated by the new solution
            self.pareto_front = [
                existing for existing in self.pareto_front
                if not solution.objectives.dominates(existing.objectives)
            ]
            
            # Add the new solution
            self.pareto_front.append(solution)
            
            # Limit front size to prevent memory issues
            if len(self.pareto_front) > 100:
                self.pareto_front = self._prune_pareto_front()
    
    def _prune_pareto_front(self) -> List[ParetoSolution]:
        """Prune Pareto front to maintain diversity."""
        if len(self.pareto_front) <= 100:
            return self.pareto_front
        
        # Use crowding distance to maintain diversity
        crowding_distances = self._calculate_crowding_distances()
        
        # Sort by crowding distance and keep top solutions
        sorted_solutions = sorted(
            zip(self.pareto_front, crowding_distances),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [sol for sol, _ in sorted_solutions[:100]]
    
    def _calculate_crowding_distances(self) -> List[float]:
        """Calculate crowding distances for diversity preservation."""
        if len(self.pareto_front) <= 2:
            return [float('inf')] * len(self.pareto_front)
        
        distances = [0.0] * len(self.pareto_front)
        
        # For each objective
        for obj_idx in range(6):
            # Sort by objective value
            sorted_indices = sorted(
                range(len(self.pareto_front)),
                key=lambda i: self.pareto_front[i].objectives.to_vector()[obj_idx]
            )
            
            # Set boundary distances to infinity
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_range = (
                self.pareto_front[sorted_indices[-1]].objectives.to_vector()[obj_idx] -
                self.pareto_front[sorted_indices[0]].objectives.to_vector()[obj_idx]
            )
            
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        self.pareto_front[sorted_indices[i+1]].objectives.to_vector()[obj_idx] -
                        self.pareto_front[sorted_indices[i-1]].objectives.to_vector()[obj_idx]
                    ) / obj_range
        
        return distances
    
    def select_for_reproduction(self, tournament_size: int = 3) -> ParetoSolution:
        """Select solution for reproduction using tournament selection."""
        if not self.pareto_front:
            raise ValueError("No solutions in Pareto front")
        
        # Tournament selection based on crowding distance
        crowding_distances = self._calculate_crowding_distances()
        
        tournament_indices = random.sample(
            range(len(self.pareto_front)),
            min(tournament_size, len(self.pareto_front))
        )
        
        best_idx = max(tournament_indices, key=lambda i: crowding_distances[i])
        return self.pareto_front[best_idx]
    
    def get_preferred_solution(self, preference_weights: Optional[np.ndarray] = None) -> Optional[ParetoSolution]:
        """Get preferred solution based on weights."""
        if not self.pareto_front:
            return None
        
        if preference_weights is None:
            preference_weights = self.objective_weights
        
        # Calculate weighted sum for each solution
        weighted_scores = []
        for solution in self.pareto_front:
            score = solution.objectives.weighted_sum(preference_weights)
            weighted_scores.append(score)
        
        # Return solution with highest weighted score
        best_idx = np.argmax(weighted_scores)
        return self.pareto_front[best_idx]
    
    def adapt_objective_weights(self, success_rates: Dict[str, float]):
        """Adapt objective weights based on optimization success rates."""
        objective_names = ['performance', 'correctness', 'maintainability', 
                          'resource_efficiency', 'security', 'readability']
        
        # Increase weights for objectives with low success rates
        for i, obj_name in enumerate(objective_names):
            success_rate = success_rates.get(obj_name, 0.5)
            if success_rate < 0.3:
                self.objective_weights[i] *= 1.1
            elif success_rate > 0.7:
                self.objective_weights[i] *= 0.95
        
        # Normalize weights
        self.objective_weights /= np.sum(self.objective_weights)
    
    def get_pareto_front_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current Pareto front."""
        if not self.pareto_front:
            return {"size": 0}
        
        # Collect objective values
        objectives_matrix = np.array([
            sol.objectives.to_vector() for sol in self.pareto_front
        ])
        
        return {
            "size": len(self.pareto_front),
            "objective_means": np.mean(objectives_matrix, axis=0).tolist(),
            "objective_stds": np.std(objectives_matrix, axis=0).tolist(),
            "objective_ranges": (np.max(objectives_matrix, axis=0) - np.min(objectives_matrix, axis=0)).tolist(),
            "hypervolume": self._calculate_hypervolume(objectives_matrix),
            "spread": self._calculate_spread(objectives_matrix)
        }
    
    def _calculate_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified)."""
        if len(objectives_matrix) == 0:
            return 0.0
        
        # Use reference point as origin
        reference_point = np.zeros(objectives_matrix.shape[1])
        
        # Simple hypervolume approximation
        hypervolume = 0.0
        for point in objectives_matrix:
            volume = np.prod(np.maximum(point - reference_point, 0))
            hypervolume += volume
        
        return hypervolume / len(objectives_matrix)
    
    def _calculate_spread(self, objectives_matrix: np.ndarray) -> float:
        """Calculate spread indicator for diversity measurement."""
        if len(objectives_matrix) <= 1:
            return 0.0
        
        # Calculate distances between consecutive points in sorted order
        distances = []
        for obj_idx in range(objectives_matrix.shape[1]):
            sorted_values = np.sort(objectives_matrix[:, obj_idx])
            obj_distances = np.diff(sorted_values)
            distances.extend(obj_distances)
        
        # Calculate spread as standard deviation of distances
        return np.std(distances) if distances else 0.0


# Initialize Pareto fitness optimizer
pareto_optimizer = ParetoFitnessOptimizer()

logger.info("🎯 Pareto fitness optimization system initialized")
logger.info("📊 Multi-objective optimization ready")
logger.info(f"⚖️ Initial objective weights: {pareto_optimizer.objective_weights}")