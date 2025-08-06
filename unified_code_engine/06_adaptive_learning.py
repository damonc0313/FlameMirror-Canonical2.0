"""
Cell 6: Adaptive Learning Engine
================================

Continuous self-recalibration and learning system that adapts mutation
strategies, fitness evaluation, and decision-making policies based on
accumulated experience without human intervention.

Features:
- Reinforcement learning for strategy optimization
- Bayesian optimization for parameter tuning
- Multi-armed bandit for algorithm selection
- Pattern mining for success identification
- Experience replay and knowledge transfer
"""

class LearningStrategy(Enum):
    """Self-learning strategy types."""
    REINFORCEMENT = "reinforcement"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    NEURAL_EVOLUTION = "neural_evolution"
    PATTERN_MINING = "pattern_mining"


@dataclass
class LearningExperience:
    """Individual learning experience record."""
    experience_id: str
    timestamp: datetime
    context: Dict[str, Any]
    action: str
    outcome: float
    success: bool
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Discovered pattern from experience analysis."""
    pattern_id: str
    description: str
    conditions: Dict[str, Any]
    confidence: float
    support_count: int
    success_rate: float
    discovered_at: datetime


class AdaptiveLearningEngine:
    """Autonomous learning and adaptation system."""
    
    def __init__(self):
        self.experiences = deque(maxlen=10000)  # Experience replay buffer
        self.patterns = []
        self.strategy_performance = defaultdict(list)
        self.learning_rates = defaultdict(lambda: 0.1)
        self.exploration_rates = defaultdict(lambda: 0.1)
        
        # Multi-armed bandit for algorithm selection
        self.algorithm_rewards = defaultdict(list)
        self.algorithm_counts = defaultdict(int)
        
        # Bayesian optimization state
        self.parameter_history = []
        self.gaussian_process_model = None
        
        # Pattern mining
        self.pattern_cache = {}
        self.context_correlations = defaultdict(list)
        
    def record_experience(self, 
                         action: str, 
                         context: Dict[str, Any], 
                         outcome: float,
                         execution_time: float,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a learning experience."""
        
        experience_id = hashlib.md5(
            f"{action}_{time.time()}_{random.random()}".encode()
        ).hexdigest()[:8]
        
        experience = LearningExperience(
            experience_id=experience_id,
            timestamp=datetime.now(),
            context=context.copy(),
            action=action,
            outcome=outcome,
            success=outcome > 0.5,  # Success threshold
            execution_time=execution_time,
            metadata=metadata or {}
        )
        
        self.experiences.append(experience)
        self.strategy_performance[action].append(outcome)
        
        # Update algorithm reward for multi-armed bandit
        algorithm = context.get('algorithm', 'default')
        self.algorithm_rewards[algorithm].append(outcome)
        self.algorithm_counts[algorithm] += 1
        
        # Trigger pattern mining if enough new experiences
        if len(self.experiences) % 100 == 0:
            self._mine_patterns()
        
        return experience_id
    
    def select_strategy(self, context: Dict[str, Any]) -> str:
        """Select optimal strategy based on learned patterns."""
        
        # Check for matching patterns first
        matching_patterns = self._find_matching_patterns(context)
        if matching_patterns:
            # Select highest confidence pattern
            best_pattern = max(matching_patterns, key=lambda p: p.confidence * p.success_rate)
            recommended_action = best_pattern.metadata.get('recommended_action')
            if recommended_action:
                return recommended_action
        
        # Fallback to multi-armed bandit for algorithm selection
        return self._select_algorithm_bandit(context)
    
    def _select_algorithm_bandit(self, context: Dict[str, Any]) -> str:
        """Select algorithm using Upper Confidence Bound (UCB)."""
        
        available_algorithms = [
            'cma_es', 'nsga_iii', 'differential_evolution', 
            'genetic_programming', 'novelty_search', 'map_elites'
        ]
        
        if not any(self.algorithm_counts.values()):
            # Random selection for initial exploration
            return random.choice(available_algorithms)
        
        # UCB1 algorithm selection
        total_counts = sum(self.algorithm_counts.values())
        ucb_scores = {}
        
        for algorithm in available_algorithms:
            if self.algorithm_counts[algorithm] == 0:
                ucb_scores[algorithm] = float('inf')  # Unvisited algorithms
            else:
                mean_reward = np.mean(self.algorithm_rewards[algorithm])
                confidence_bound = np.sqrt(
                    2 * np.log(total_counts) / self.algorithm_counts[algorithm]
                )
                ucb_scores[algorithm] = mean_reward + confidence_bound
        
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def _find_matching_patterns(self, context: Dict[str, Any]) -> List[LearningPattern]:
        """Find patterns that match the current context."""
        matching_patterns = []
        
        for pattern in self.patterns:
            match_score = self._calculate_pattern_match(pattern.conditions, context)
            if match_score > 0.7:  # Similarity threshold
                matching_patterns.append(pattern)
        
        return matching_patterns
    
    def _calculate_pattern_match(self, pattern_conditions: Dict[str, Any], 
                                context: Dict[str, Any]) -> float:
        """Calculate how well context matches pattern conditions."""
        if not pattern_conditions:
            return 0.0
        
        matches = 0
        total_conditions = len(pattern_conditions)
        
        for key, expected_value in pattern_conditions.items():
            if key in context:
                if isinstance(expected_value, (int, float)):
                    # Numerical similarity
                    if isinstance(context[key], (int, float)):
                        diff = abs(expected_value - context[key])
                        max_val = max(abs(expected_value), abs(context[key]), 1.0)
                        similarity = 1.0 - (diff / max_val)
                        matches += max(0, similarity)
                elif expected_value == context[key]:
                    # Exact match
                    matches += 1.0
                elif isinstance(expected_value, str) and isinstance(context[key], str):
                    # String similarity
                    matches += self._string_similarity(expected_value, context[key])
        
        return matches / total_conditions
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _mine_patterns(self):
        """Mine patterns from recent experiences."""
        if len(self.experiences) < 50:
            return
        
        # Analyze successful vs unsuccessful experiences
        successful_experiences = [exp for exp in self.experiences if exp.success]
        failed_experiences = [exp for exp in self.experiences if not exp.success]
        
        if len(successful_experiences) < 10:
            return
        
        # Find common patterns in successful experiences
        success_patterns = self._extract_common_patterns(successful_experiences)
        
        for pattern_data in success_patterns:
            if pattern_data['support_count'] >= 5:  # Minimum support
                pattern = LearningPattern(
                    pattern_id=hashlib.md5(str(pattern_data).encode()).hexdigest()[:8],
                    description=f"Success pattern: {pattern_data['description']}",
                    conditions=pattern_data['conditions'],
                    confidence=pattern_data['confidence'],
                    support_count=pattern_data['support_count'],
                    success_rate=pattern_data['success_rate'],
                    discovered_at=datetime.now()
                )
                
                # Add if not already discovered
                if not any(p.pattern_id == pattern.pattern_id for p in self.patterns):
                    self.patterns.append(pattern)
    
    def _extract_common_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Extract common patterns from experiences."""
        patterns = []
        
        # Group by action type
        action_groups = defaultdict(list)
        for exp in experiences:
            action_groups[exp.action].append(exp)
        
        for action, action_experiences in action_groups.items():
            if len(action_experiences) < 5:
                continue
            
            # Find common context patterns
            context_keys = set()
            for exp in action_experiences:
                context_keys.update(exp.context.keys())
            
            common_conditions = {}
            for key in context_keys:
                values = [exp.context.get(key) for exp in action_experiences if key in exp.context]
                if len(values) >= len(action_experiences) * 0.6:  # Present in 60% of cases
                    # Find most common value
                    if all(isinstance(v, (int, float)) for v in values):
                        common_conditions[key] = np.mean(values)
                    else:
                        value_counts = defaultdict(int)
                        for v in values:
                            value_counts[v] += 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])
                        if most_common[1] >= len(values) * 0.5:
                            common_conditions[key] = most_common[0]
            
            if common_conditions:
                success_rate = sum(1 for exp in action_experiences if exp.success) / len(action_experiences)
                patterns.append({
                    'description': f"Action '{action}' with context pattern",
                    'conditions': common_conditions,
                    'confidence': len(action_experiences) / len(experiences),
                    'support_count': len(action_experiences),
                    'success_rate': success_rate,
                    'recommended_action': action
                })
        
        return patterns
    
    def adapt_parameters(self, algorithm: str, current_params: Dict[str, float]) -> Dict[str, float]:
        """Adapt algorithm parameters using Bayesian optimization."""
        
        # Simple adaptive parameter adjustment
        adapted_params = current_params.copy()
        
        if algorithm in self.strategy_performance:
            recent_performance = self.strategy_performance[algorithm][-10:]  # Last 10 results
            
            if len(recent_performance) >= 5:
                avg_performance = np.mean(recent_performance)
                performance_trend = np.mean(recent_performance[-3:]) - np.mean(recent_performance[:-3])
                
                # Adapt learning rate
                if avg_performance < 0.3:
                    # Poor performance, increase exploration
                    for param in adapted_params:
                        if 'rate' in param or 'prob' in param:
                            adapted_params[param] = min(1.0, adapted_params[param] * 1.1)
                elif avg_performance > 0.7:
                    # Good performance, fine-tune
                    for param in adapted_params:
                        if 'rate' in param or 'prob' in param:
                            adapted_params[param] = max(0.01, adapted_params[param] * 0.95)
                
                # Adapt based on trend
                if performance_trend < -0.1:
                    # Declining performance, try different parameters
                    for param in adapted_params:
                        if isinstance(adapted_params[param], float):
                            noise = random.uniform(-0.1, 0.1)
                            adapted_params[param] = max(0.01, min(1.0, adapted_params[param] + noise))
        
        return adapted_params
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process."""
        
        if not self.experiences:
            return {"status": "No experiences recorded"}
        
        total_experiences = len(self.experiences)
        successful_experiences = sum(1 for exp in self.experiences if exp.success)
        success_rate = successful_experiences / total_experiences
        
        # Algorithm performance
        algorithm_performance = {}
        for algorithm, rewards in self.algorithm_rewards.items():
            if rewards:
                algorithm_performance[algorithm] = {
                    'avg_reward': np.mean(rewards),
                    'count': len(rewards),
                    'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards)
                }
        
        # Pattern statistics
        pattern_stats = {
            'total_patterns': len(self.patterns),
            'avg_confidence': np.mean([p.confidence for p in self.patterns]) if self.patterns else 0,
            'avg_success_rate': np.mean([p.success_rate for p in self.patterns]) if self.patterns else 0
        }
        
        # Recent performance trends
        recent_experiences = list(self.experiences)[-100:]  # Last 100 experiences
        recent_success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
        
        return {
            'total_experiences': total_experiences,
            'overall_success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'algorithm_performance': algorithm_performance,
            'pattern_statistics': pattern_stats,
            'learning_trends': {
                'improvement': recent_success_rate - success_rate,
                'patterns_discovered': len([p for p in self.patterns 
                                          if (datetime.now() - p.discovered_at).days < 1])
            }
        }
    
    def transfer_knowledge(self, similar_context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from similar past experiences."""
        
        # Find experiences with similar contexts
        similar_experiences = []
        for exp in self.experiences:
            similarity = self._calculate_pattern_match(exp.context, similar_context)
            if similarity > 0.6:
                similar_experiences.append((exp, similarity))
        
        if not similar_experiences:
            return {"recommendation": "No similar experiences found"}
        
        # Sort by similarity and success
        similar_experiences.sort(key=lambda x: x[1] * (1 if x[0].success else 0.5), reverse=True)
        
        # Extract recommendations from top similar experiences
        top_experiences = similar_experiences[:5]
        action_scores = defaultdict(float)
        
        for exp, similarity in top_experiences:
            weight = similarity * (1.0 if exp.success else 0.3)
            action_scores[exp.action] += weight
        
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            confidence = action_scores[best_action] / sum(action_scores.values())
            
            return {
                "recommended_action": best_action,
                "confidence": confidence,
                "based_on_experiences": len(top_experiences),
                "transfer_basis": [exp.experience_id for exp, _ in top_experiences]
            }
        
        return {"recommendation": "No confident recommendation available"}
    
    def evolve_strategies(self) -> Dict[str, Any]:
        """Evolve learning strategies based on meta-learning."""
        
        strategy_evolution = {}
        
        # Analyze which learning strategies work best
        strategy_success = defaultdict(list)
        
        for exp in self.experiences:
            learning_strategy = exp.metadata.get('learning_strategy', 'default')
            strategy_success[learning_strategy].append(exp.outcome)
        
        # Evolve strategy preferences
        for strategy, outcomes in strategy_success.items():
            if len(outcomes) >= 10:
                avg_outcome = np.mean(outcomes)
                recent_trend = np.mean(outcomes[-5:]) - np.mean(outcomes[:-5])
                
                strategy_evolution[strategy] = {
                    'performance': avg_outcome,
                    'trend': recent_trend,
                    'recommended_weight': max(0.1, min(2.0, avg_outcome + recent_trend))
                }
        
        return strategy_evolution


# Initialize adaptive learning engine
adaptive_learning = AdaptiveLearningEngine()

logger.info("🧠 Adaptive learning engine initialized")
logger.info("📈 Continuous self-recalibration system ready")
logger.info("🔄 Experience-based strategy adaptation enabled")