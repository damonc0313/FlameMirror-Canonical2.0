#!/usr/bin/env python3
"""
Adaptive Learning Engine for Autonomous Code Evolution
Continuous Self-Recalibration Without Human Intervention

This module implements self-learning algorithms that adapt mutation strategies,
fitness evaluation, and decision-making policies based on accumulated experience.
"""

from __future__ import annotations

import numpy as np
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
import random


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
    """Discovered learning pattern."""
    pattern_id: str
    context_signature: str
    action_type: str
    success_rate: float
    average_outcome: float
    confidence: float
    frequency: int
    last_updated: datetime
    feature_weights: Dict[str, float] = field(default_factory=dict)


class AutonomousReinforcementLearner:
    """Self-directing reinforcement learning for mutation strategy."""
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.experience_buffer = deque(maxlen=10000)
        self.adaptation_history = []
        
    def get_state_signature(self, context: Dict[str, Any]) -> str:
        """Generate autonomous state signature from context."""
        try:
            # Extract key features for state representation
            features = []
            
            # Repository characteristics
            if 'repository_metrics' in context:
                metrics = context['repository_metrics']
                complexity = metrics.get('average_complexity', 0)
                file_count = metrics.get('file_count', 0)
                features.extend([f"complexity_{int(complexity/10)}", f"files_{int(file_count/10)}"])
            
            # Previous mutation outcomes
            if 'recent_outcomes' in context:
                outcomes = context['recent_outcomes']
                success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.5
                features.append(f"success_rate_{int(success_rate*10)}")
            
            # Time context
            hour = datetime.now().hour
            features.append(f"hour_{hour}")
            
            return "_".join(sorted(features))
            
        except Exception:
            return "default_state"
    
    def select_action_autonomously(self, 
                                 state: str, 
                                 available_actions: List[str]) -> str:
        """Autonomously select action using epsilon-greedy strategy."""
        if not available_actions:
            return "default_action"
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best known action
            action_values = {action: self.q_table[state][action] 
                           for action in available_actions}
            
            if not action_values or all(v == 0 for v in action_values.values()):
                return random.choice(available_actions)
            
            return max(action_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value_autonomously(self, 
                                  state: str, 
                                  action: str, 
                                  reward: float, 
                                  next_state: str):
        """Autonomously update Q-values using Q-learning."""
        # Get maximum Q-value for next state
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Q-learning update
        current_q = self.q_table[state][action]
        discount_factor = 0.95
        
        new_q = current_q + self.learning_rate * (reward + discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        # Update visit counts
        self.state_action_counts[state][action] += 1
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "action": action,
            "old_q": current_q,
            "new_q": new_q,
            "reward": reward
        })
    
    def decay_epsilon_autonomously(self):
        """Autonomously decay exploration rate over time."""
        min_epsilon = 0.01
        decay_rate = 0.995
        
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get autonomous learning progress statistics."""
        return {
            "q_table_size": len(self.q_table),
            "total_experiences": len(self.experience_buffer),
            "current_epsilon": self.epsilon,
            "adaptation_count": len(self.adaptation_history),
            "most_visited_states": self._get_top_states(5),
            "best_actions": self._get_best_actions(5)
        }
    
    def _get_top_states(self, count: int) -> List[Tuple[str, int]]:
        """Get most frequently visited states."""
        state_visits = defaultdict(int)
        for state, actions in self.state_action_counts.items():
            state_visits[state] = sum(actions.values())
        
        return sorted(state_visits.items(), key=lambda x: x[1], reverse=True)[:count]
    
    def _get_best_actions(self, count: int) -> List[Tuple[str, str, float]]:
        """Get best state-action pairs by Q-value."""
        best_pairs = []
        for state, actions in self.q_table.items():
            for action, q_value in actions.items():
                best_pairs.append((state, action, q_value))
        
        return sorted(best_pairs, key=lambda x: x[2], reverse=True)[:count]


class AutonomousBayesianOptimizer:
    """Bayesian optimization for autonomous parameter tuning."""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = parameter_bounds
        self.observations = []
        self.best_parameters = None
        self.best_score = float('-inf')
        self.acquisition_history = []
        
    def suggest_parameters_autonomously(self) -> Dict[str, float]:
        """Autonomously suggest next parameter configuration."""
        if len(self.observations) < 3:
            # Random exploration for initial observations
            return self._random_parameters()
        else:
            # Gaussian Process-based suggestion (simplified)
            return self._gp_suggest_parameters()
    
    def _random_parameters(self) -> Dict[str, float]:
        """Generate random parameters within bounds."""
        parameters = {}
        for param, (low, high) in self.parameter_bounds.items():
            parameters[param] = random.uniform(low, high)
        return parameters
    
    def _gp_suggest_parameters(self) -> Dict[str, float]:
        """Simplified Gaussian Process parameter suggestion."""
        # This is a simplified version - a full implementation would use 
        # libraries like scikit-optimize or GPyOpt
        
        if not self.observations:
            return self._random_parameters()
        
        # Get best parameters from history
        best_obs = max(self.observations, key=lambda x: x['score'])
        best_params = best_obs['parameters']
        
        # Add Gaussian noise for exploration
        suggested = {}
        for param, value in best_params.items():
            low, high = self.parameter_bounds[param]
            
            # Add noise proportional to parameter range
            noise_scale = (high - low) * 0.1
            noisy_value = value + random.gauss(0, noise_scale)
            
            # Clip to bounds
            suggested[param] = max(low, min(high, noisy_value))
        
        return suggested
    
    def update_observation_autonomously(self, 
                                      parameters: Dict[str, float], 
                                      score: float):
        """Autonomously update observations with new result."""
        observation = {
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters.copy(),
            "score": score
        }
        
        self.observations.append(observation)
        
        # Update best parameters
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = parameters.copy()
            
        # Record acquisition
        self.acquisition_history.append({
            "timestamp": datetime.now().isoformat(),
            "suggested_params": parameters,
            "observed_score": score,
            "is_best": score > self.best_score
        })
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get autonomous optimization progress summary."""
        return {
            "total_evaluations": len(self.observations),
            "best_score": self.best_score,
            "best_parameters": self.best_parameters,
            "parameter_bounds": self.parameter_bounds,
            "recent_improvements": len([obs for obs in self.observations[-10:] 
                                      if obs['score'] > self.best_score * 0.9])
        }


class AutonomousPatternMiner:
    """Autonomous pattern discovery from mutation experiences."""
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns = {}
        self.frequent_itemsets = []
        self.association_rules = []
        
    def mine_patterns_autonomously(self, experiences: List[LearningExperience]) -> List[LearningPattern]:
        """Autonomously mine patterns from learning experiences."""
        if len(experiences) < 10:
            return []
        
        # Extract features and outcomes
        feature_sets = []
        outcomes = []
        
        for exp in experiences:
            features = self._extract_features(exp)
            feature_sets.append(features)
            outcomes.append(exp.success)
        
        # Mine frequent patterns
        patterns = self._mine_frequent_patterns(feature_sets, outcomes)
        
        # Convert to LearningPattern objects
        learning_patterns = []
        for pattern_data in patterns:
            pattern = LearningPattern(
                pattern_id=f"pattern_{len(learning_patterns)}",
                context_signature=pattern_data['signature'],
                action_type=pattern_data.get('action_type', 'unknown'),
                success_rate=pattern_data['success_rate'],
                average_outcome=pattern_data['average_outcome'],
                confidence=pattern_data['confidence'],
                frequency=pattern_data['frequency'],
                last_updated=datetime.now(),
                feature_weights=pattern_data.get('feature_weights', {})
            )
            learning_patterns.append(pattern)
        
        self.patterns = {p.pattern_id: p for p in learning_patterns}
        return learning_patterns
    
    def _extract_features(self, experience: LearningExperience) -> Set[str]:
        """Extract features from learning experience."""
        features = set()
        
        # Context features
        context = experience.context
        if 'mutation_type' in context:
            features.add(f"mutation_{context['mutation_type']}")
        
        if 'file_type' in context:
            features.add(f"file_{context['file_type']}")
        
        if 'complexity' in context:
            complexity_level = "low" if context['complexity'] < 50 else "high"
            features.add(f"complexity_{complexity_level}")
        
        # Temporal features
        hour = experience.timestamp.hour
        if 6 <= hour < 12:
            features.add("time_morning")
        elif 12 <= hour < 18:
            features.add("time_afternoon")
        else:
            features.add("time_evening")
        
        # Outcome features
        if experience.success:
            features.add("outcome_success")
        else:
            features.add("outcome_failure")
        
        return features
    
    def _mine_frequent_patterns(self, 
                              feature_sets: List[Set[str]], 
                              outcomes: List[bool]) -> List[Dict[str, Any]]:
        """Mine frequent patterns using simplified Apriori algorithm."""
        patterns = []
        
        # Count feature frequencies
        feature_counts = defaultdict(int)
        total_transactions = len(feature_sets)
        
        for feature_set in feature_sets:
            for feature in feature_set:
                feature_counts[feature] += 1
        
        # Find frequent single features
        frequent_features = []
        for feature, count in feature_counts.items():
            support = count / total_transactions
            if support >= self.min_support:
                frequent_features.append(feature)
                
                # Calculate success rate for this feature
                success_count = sum(1 for i, fs in enumerate(feature_sets) 
                                  if feature in fs and outcomes[i])
                success_rate = success_count / count if count > 0 else 0
                
                if success_rate >= self.min_confidence:
                    patterns.append({
                        'signature': feature,
                        'success_rate': success_rate,
                        'average_outcome': success_rate,
                        'confidence': success_rate,
                        'frequency': count,
                        'feature_weights': {feature: 1.0}
                    })
        
        # Find frequent pairs (simplified - would extend to larger itemsets)
        for i, feat1 in enumerate(frequent_features):
            for feat2 in frequent_features[i+1:]:
                pair_count = sum(1 for fs in feature_sets if feat1 in fs and feat2 in fs)
                support = pair_count / total_transactions
                
                if support >= self.min_support:
                    success_count = sum(1 for i, fs in enumerate(feature_sets)
                                      if feat1 in fs and feat2 in fs and outcomes[i])
                    success_rate = success_count / pair_count if pair_count > 0 else 0
                    
                    if success_rate >= self.min_confidence:
                        patterns.append({
                            'signature': f"{feat1}&{feat2}",
                            'success_rate': success_rate,
                            'average_outcome': success_rate,
                            'confidence': success_rate,
                            'frequency': pair_count,
                            'feature_weights': {feat1: 0.5, feat2: 0.5}
                        })
        
        return patterns
    
    def predict_success_probability(self, context: Dict[str, Any]) -> float:
        """Predict success probability based on discovered patterns."""
        if not self.patterns:
            return 0.5  # Default probability
        
        # Extract features from context
        features = self._extract_features_from_context(context)
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.patterns.values():
            pattern_features = set(pattern.context_signature.split('&'))
            if pattern_features.issubset(features):
                matching_patterns.append(pattern)
        
        if not matching_patterns:
            return 0.5
        
        # Weighted average of matching patterns
        total_weight = sum(p.confidence * p.frequency for p in matching_patterns)
        weighted_prob = sum(p.success_rate * p.confidence * p.frequency 
                          for p in matching_patterns)
        
        return weighted_prob / total_weight if total_weight > 0 else 0.5
    
    def _extract_features_from_context(self, context: Dict[str, Any]) -> Set[str]:
        """Extract features from context dictionary."""
        features = set()
        
        if 'mutation_type' in context:
            features.add(f"mutation_{context['mutation_type']}")
        
        if 'file_type' in context:
            features.add(f"file_{context['file_type']}")
        
        if 'complexity' in context:
            complexity_level = "low" if context['complexity'] < 50 else "high"
            features.add(f"complexity_{complexity_level}")
        
        # Add temporal context
        hour = datetime.now().hour
        if 6 <= hour < 12:
            features.add("time_morning")
        elif 12 <= hour < 18:
            features.add("time_afternoon")
        else:
            features.add("time_evening")
        
        return features


class AutonomousAdaptiveLearningEngine:
    """Main adaptive learning coordination engine."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize learning components
        self.rl_learner = AutonomousReinforcementLearner()
        self.bayesian_optimizer = AutonomousBayesianOptimizer({
            'learning_rate': (0.01, 0.3),
            'epsilon': (0.01, 0.5),
            'mutation_probability': (0.1, 0.9)
        })
        self.pattern_miner = AutonomousPatternMiner()
        
        # Experience storage
        self.experiences = deque(maxlen=50000)
        self.learning_statistics = {}
        self.adaptation_schedule = []
        
        # Persistence
        self.model_file = workspace_dir / "learning_models.pkl"
        self.experience_file = workspace_dir / "experiences.json"
        
        # Load existing models
        self.load_learning_state()
        
        # Start background learning thread
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_active = True
        self.learning_thread.start()
    
    def record_experience_autonomously(self, 
                                     context: Dict[str, Any],
                                     action: str,
                                     outcome: float,
                                     success: bool,
                                     execution_time: float,
                                     metadata: Optional[Dict[str, Any]] = None):
        """Autonomously record learning experience."""
        experience = LearningExperience(
            experience_id=f"exp_{time.time()}_{random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            context=context.copy(),
            action=action,
            outcome=outcome,
            success=success,
            execution_time=execution_time,
            metadata=metadata or {}
        )
        
        self.experiences.append(experience)
        
        # Trigger immediate learning updates for significant experiences
        if outcome > 0.8 or outcome < 0.2:
            self._update_learning_models_autonomously([experience])
    
    def suggest_next_action_autonomously(self, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Autonomously suggest next action based on learned patterns."""
        # Get state representation
        state = self.rl_learner.get_state_signature(context)
        
        # Available actions (mutation types)
        available_actions = [
            "memoization", "timing_optimization", "logging_enhancement",
            "refactoring", "caching", "parallelization", "algorithm_substitution"
        ]
        
        # Get RL suggestion
        rl_action = self.rl_learner.select_action_autonomously(state, available_actions)
        
        # Get pattern-based prediction
        success_probability = self.pattern_miner.predict_success_probability(context)
        
        # Get optimized parameters
        suggested_params = self.bayesian_optimizer.suggest_parameters_autonomously()
        
        suggestion_info = {
            "primary_action": rl_action,
            "success_probability": success_probability,
            "suggested_parameters": suggested_params,
            "state_signature": state,
            "confidence": self._calculate_suggestion_confidence(context)
        }
        
        return rl_action, suggestion_info
    
    def update_learning_autonomously(self, 
                                   context: Dict[str, Any],
                                   action: str,
                                   outcome: float,
                                   success: bool):
        """Autonomously update all learning models based on outcome."""
        # Update reinforcement learning
        state = self.rl_learner.get_state_signature(context)
        reward = self._calculate_reward(outcome, success)
        
        # For RL, we need the next state (simplified - using current state)
        next_state = state  # In practice, would be the state after action
        self.rl_learner.update_q_value_autonomously(state, action, reward, next_state)
        
        # Decay exploration
        self.rl_learner.decay_epsilon_autonomously()
        
        # Update Bayesian optimization if parameters were suggested
        if hasattr(self, '_last_suggested_params'):
            self.bayesian_optimizer.update_observation_autonomously(
                self._last_suggested_params, outcome
            )
        
        # Update learning statistics
        self._update_learning_statistics()
    
    def _calculate_reward(self, outcome: float, success: bool) -> float:
        """Calculate reward signal for reinforcement learning."""
        base_reward = outcome if success else -outcome
        
        # Bonus for high performance
        if outcome > 0.8:
            base_reward += 0.2
        
        # Penalty for failures
        if not success:
            base_reward -= 0.5
        
        return np.clip(base_reward, -1.0, 1.0)
    
    def _calculate_suggestion_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence in suggestion based on experience."""
        state = self.rl_learner.get_state_signature(context)
        
        # Base confidence from Q-table visits
        state_visits = sum(self.rl_learner.state_action_counts[state].values())
        visit_confidence = min(1.0, state_visits / 10.0)  # Saturate at 10 visits
        
        # Pattern match confidence
        pattern_confidence = self.pattern_miner.predict_success_probability(context)
        
        # Bayesian optimization confidence
        bo_confidence = min(1.0, len(self.bayesian_optimizer.observations) / 20.0)
        
        # Weighted combination
        overall_confidence = (
            0.4 * visit_confidence +
            0.4 * pattern_confidence +
            0.2 * bo_confidence
        )
        
        return overall_confidence
    
    def _continuous_learning_loop(self):
        """Background thread for continuous learning updates."""
        while self.learning_active:
            try:
                # Periodic learning updates
                if len(self.experiences) >= 10:
                    recent_experiences = list(self.experiences)[-100:]  # Last 100 experiences
                    self._update_learning_models_autonomously(recent_experiences)
                
                # Save learning state periodically
                self.save_learning_state()
                
                # Sleep before next update
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                # Autonomous error handling
                logging.warning(f"Learning loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_learning_models_autonomously(self, experiences: List[LearningExperience]):
        """Update all learning models with new experiences."""
        if not experiences:
            return
        
        # Update pattern mining
        patterns = self.pattern_miner.mine_patterns_autonomously(experiences)
        
        # Update learning statistics
        self.learning_statistics.update({
            "last_update": datetime.now().isoformat(),
            "total_experiences": len(self.experiences),
            "discovered_patterns": len(patterns),
            "rl_stats": self.rl_learner.get_learning_statistics(),
            "bo_stats": self.bayesian_optimizer.get_optimization_summary()
        })
    
    def _update_learning_statistics(self):
        """Update comprehensive learning statistics."""
        self.learning_statistics = {
            "timestamp": datetime.now().isoformat(),
            "total_experiences": len(self.experiences),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "learning_progress": {
                "rl_q_table_size": len(self.rl_learner.q_table),
                "patterns_discovered": len(self.pattern_miner.patterns),
                "bo_evaluations": len(self.bayesian_optimizer.observations)
            },
            "adaptation_metrics": {
                "epsilon": self.rl_learner.epsilon,
                "best_bo_score": self.bayesian_optimizer.best_score,
                "pattern_confidence": self._average_pattern_confidence()
            }
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate from recent experiences."""
        if len(self.experiences) < 10:
            return 0.5
        
        recent = list(self.experiences)[-50:]  # Last 50 experiences
        successes = sum(1 for exp in recent if exp.success)
        return successes / len(recent)
    
    def _average_pattern_confidence(self) -> float:
        """Calculate average confidence of discovered patterns."""
        if not self.pattern_miner.patterns:
            return 0.0
        
        confidences = [p.confidence for p in self.pattern_miner.patterns.values()]
        return sum(confidences) / len(confidences)
    
    def save_learning_state(self):
        """Save learning models to persistent storage."""
        try:
            # Save models
            models_data = {
                'rl_q_table': dict(self.rl_learner.q_table),
                'rl_state_counts': dict(self.rl_learner.state_action_counts),
                'rl_epsilon': self.rl_learner.epsilon,
                'bo_observations': self.bayesian_optimizer.observations,
                'bo_best_params': self.bayesian_optimizer.best_parameters,
                'bo_best_score': self.bayesian_optimizer.best_score,
                'patterns': {pid: {
                    'pattern_id': p.pattern_id,
                    'context_signature': p.context_signature,
                    'action_type': p.action_type,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence,
                    'frequency': p.frequency,
                    'feature_weights': p.feature_weights
                } for pid, p in self.pattern_miner.patterns.items()}
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(models_data, f)
            
            # Save recent experiences
            experiences_data = [
                {
                    'experience_id': exp.experience_id,
                    'timestamp': exp.timestamp.isoformat(),
                    'context': exp.context,
                    'action': exp.action,
                    'outcome': exp.outcome,
                    'success': exp.success,
                    'execution_time': exp.execution_time,
                    'metadata': exp.metadata
                }
                for exp in list(self.experiences)[-1000:]  # Save last 1000 experiences
            ]
            
            with open(self.experience_file, 'w') as f:
                json.dump(experiences_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save learning state: {e}")
    
    def load_learning_state(self):
        """Load learning models from persistent storage."""
        try:
            # Load models
            if self.model_file.exists():
                with open(self.model_file, 'rb') as f:
                    models_data = pickle.load(f)
                
                # Restore RL learner
                if 'rl_q_table' in models_data:
                    self.rl_learner.q_table = defaultdict(lambda: defaultdict(float), models_data['rl_q_table'])
                if 'rl_state_counts' in models_data:
                    self.rl_learner.state_action_counts = defaultdict(lambda: defaultdict(int), models_data['rl_state_counts'])
                if 'rl_epsilon' in models_data:
                    self.rl_learner.epsilon = models_data['rl_epsilon']
                
                # Restore Bayesian optimizer
                if 'bo_observations' in models_data:
                    self.bayesian_optimizer.observations = models_data['bo_observations']
                if 'bo_best_params' in models_data:
                    self.bayesian_optimizer.best_parameters = models_data['bo_best_params']
                if 'bo_best_score' in models_data:
                    self.bayesian_optimizer.best_score = models_data['bo_best_score']
                
                # Restore patterns
                if 'patterns' in models_data:
                    patterns = {}
                    for pid, pdata in models_data['patterns'].items():
                        pattern = LearningPattern(
                            pattern_id=pdata['pattern_id'],
                            context_signature=pdata['context_signature'],
                            action_type=pdata['action_type'],
                            success_rate=pdata['success_rate'],
                            average_outcome=pdata.get('average_outcome', pdata['success_rate']),
                            confidence=pdata['confidence'],
                            frequency=pdata['frequency'],
                            last_updated=datetime.now(),
                            feature_weights=pdata.get('feature_weights', {})
                        )
                        patterns[pid] = pattern
                    self.pattern_miner.patterns = patterns
            
            # Load experiences
            if self.experience_file.exists():
                with open(self.experience_file, 'r') as f:
                    experiences_data = json.load(f)
                
                for exp_data in experiences_data:
                    experience = LearningExperience(
                        experience_id=exp_data['experience_id'],
                        timestamp=datetime.fromisoformat(exp_data['timestamp']),
                        context=exp_data['context'],
                        action=exp_data['action'],
                        outcome=exp_data['outcome'],
                        success=exp_data['success'],
                        execution_time=exp_data['execution_time'],
                        metadata=exp_data.get('metadata', {})
                    )
                    self.experiences.append(experience)
                    
        except Exception as e:
            logging.warning(f"Failed to load learning state: {e}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning progress report."""
        return {
            "learning_engine_status": "active" if self.learning_active else "inactive",
            "statistics": self.learning_statistics,
            "reinforcement_learning": self.rl_learner.get_learning_statistics(),
            "bayesian_optimization": self.bayesian_optimizer.get_optimization_summary(),
            "pattern_mining": {
                "total_patterns": len(self.pattern_miner.patterns),
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "success_rate": p.success_rate,
                        "confidence": p.confidence,
                        "frequency": p.frequency
                    }
                    for p in list(self.pattern_miner.patterns.values())[:10]  # Top 10
                ]
            },
            "experience_summary": {
                "total_experiences": len(self.experiences),
                "recent_success_rate": self._calculate_recent_success_rate(),
                "experience_diversity": self._calculate_experience_diversity()
            }
        }
    
    def _calculate_experience_diversity(self) -> float:
        """Calculate diversity of learning experiences."""
        if len(self.experiences) < 10:
            return 0.0
        
        # Count unique actions
        actions = set(exp.action for exp in self.experiences)
        action_diversity = len(actions) / 10  # Normalize assuming max 10 action types
        
        # Count unique contexts (simplified)
        contexts = set(str(sorted(exp.context.items())) for exp in self.experiences)
        context_diversity = min(1.0, len(contexts) / len(self.experiences))
        
        return (action_diversity + context_diversity) / 2
    
    def shutdown(self):
        """Gracefully shutdown learning engine."""
        self.learning_active = False
        if self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        self.save_learning_state()


def main():
    """Demonstration of adaptive learning engine."""
    workspace = Path("./adaptive_learning_workspace")
    
    # Initialize learning engine
    learning_engine = AutonomousAdaptiveLearningEngine(workspace)
    
    # Simulate learning experiences
    contexts = [
        {"mutation_type": "memoization", "complexity": 45, "file_type": "python"},
        {"mutation_type": "logging_enhancement", "complexity": 78, "file_type": "python"},
        {"mutation_type": "refactoring", "complexity": 120, "file_type": "python"}
    ]
    
    print("Simulating autonomous learning...")
    
    for i in range(20):
        context = random.choice(contexts)
        
        # Get suggestion
        action, info = learning_engine.suggest_next_action_autonomously(context)
        
        # Simulate outcome
        outcome = random.uniform(0.3, 0.9)
        success = outcome > 0.6
        execution_time = random.uniform(0.5, 3.0)
        
        # Record experience
        learning_engine.record_experience_autonomously(
            context, action, outcome, success, execution_time
        )
        
        # Update learning
        learning_engine.update_learning_autonomously(context, action, outcome, success)
        
        print(f"Iteration {i+1}: Action={action}, Outcome={outcome:.2f}, Success={success}")
    
    # Generate report
    report = learning_engine.get_comprehensive_report()
    print("\nLearning Progress Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Shutdown
    learning_engine.shutdown()


if __name__ == "__main__":
    main()