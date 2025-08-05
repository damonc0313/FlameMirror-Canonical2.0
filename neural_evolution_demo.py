#!/usr/bin/env python3
"""
Neural Code Evolution Demo
==========================

This demo showcases the neural code evolution engine with various examples
of AI-powered code generation, mutation, and optimization.

Author: Neural Evolution Demo Agent
License: MIT
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from neural_code_evolution_engine import (
    NeuralCodeEvolutionEngine,
    NeuralEvolutionConfig,
    CodeEvolutionResult
)


class NeuralEvolutionDemo:
    """Demo class for showcasing neural code evolution capabilities."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = self._create_demo_config()
        self.engine = NeuralCodeEvolutionEngine(self.config)
        
        # Demo code samples
        self.demo_samples = self._load_demo_samples()
    
    def _create_demo_config(self) -> NeuralEvolutionConfig:
        """Create configuration for the demo."""
        return NeuralEvolutionConfig(
            provider_type="openai",
            model_name="gpt-4",
            api_key=self.api_key,
            max_concurrent_evolutions=3,
            enable_quality_analysis=True,
            enable_parallel_evolution=True,
            fitness_threshold=0.6,  # Lower threshold for demo
            max_evolution_attempts=2
        )
    
    def _load_demo_samples(self) -> Dict[str, str]:
        """Load demo code samples for evolution."""
        return {
            "fibonacci": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
""",
            "data_processor": """
def process_data(data_list):
    result = []
    for item in data_list:
        if item > 0:
            result.append(item * 2)
    return result

def validate_input(data):
    if not isinstance(data, list):
        return False
    return True
""",
            "api_handler": """
def handle_request(request_data):
    response = {}
    if 'action' in request_data:
        if request_data['action'] == 'get':
            response['data'] = get_data()
        elif request_data['action'] == 'post':
            response['data'] = post_data(request_data)
    return response

def get_data():
    return {"message": "Hello World"}

def post_data(data):
    return {"status": "success"}
""",
            "ml_predictor": """
def predict_value(features):
    # Simple linear prediction
    weights = [0.1, 0.2, 0.3, 0.4]
    prediction = 0
    for i, feature in enumerate(features):
        if i < len(weights):
            prediction += feature * weights[i]
    return prediction

def train_model(training_data):
    # Placeholder for training
    return {"accuracy": 0.85}
"""
        }
    
    async def run_comprehensive_demo(self):
        """Run a comprehensive demo of neural code evolution."""
        print("🧠 Neural Code Evolution Engine Demo")
        print("=" * 60)
        print(f"🤖 LLM Provider: {self.config.provider_type}")
        print(f"📊 Model: {self.config.model_name}")
        print(f"🎯 Fitness Threshold: {self.config.fitness_threshold}")
        print("=" * 60)
        
        # Demo 1: Performance Optimization
        await self._demo_performance_optimization()
        
        # Demo 2: Readability Improvement
        await self._demo_readability_improvement()
        
        # Demo 3: Bug Fix and Error Handling
        await self._demo_bug_fix()
        
        # Demo 4: Feature Addition
        await self._demo_feature_addition()
        
        # Demo 5: Parallel Evolution
        await self._demo_parallel_evolution()
        
        # Demo 6: Code Quality Analysis
        await self._demo_quality_analysis()
        
        # Print final statistics
        await self._print_final_statistics()
    
    async def _demo_performance_optimization(self):
        """Demo performance optimization."""
        print(f"\n🚀 Demo 1: Performance Optimization")
        print("-" * 40)
        
        code = self.demo_samples["fibonacci"]
        print("📝 Original Code:")
        print(code)
        
        result = await self.engine.evolve_code(
            code=code,
            fitness_goal="Optimize for maximum performance and speed",
            context={"target": "performance", "constraints": ["maintainability"]}
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📊 Fitness Score: {result.fitness_score:.3f}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.quality_metrics:
            print(f"🎯 Performance Score: {result.quality_metrics.get('performance_score', 0):.1f}/10")
        
        print(f"\n🧠 Optimized Code:")
        print(result.evolved_code)
    
    async def _demo_readability_improvement(self):
        """Demo readability improvement."""
        print(f"\n📖 Demo 2: Readability Improvement")
        print("-" * 40)
        
        code = self.demo_samples["data_processor"]
        print("📝 Original Code:")
        print(code)
        
        result = await self.engine.evolve_code(
            code=code,
            fitness_goal="Improve code readability and maintainability",
            context={"target": "readability", "constraints": ["performance"]}
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📊 Fitness Score: {result.fitness_score:.3f}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.quality_metrics:
            print(f"📖 Readability Score: {result.quality_metrics.get('readability_score', 0):.1f}/10")
        
        print(f"\n🧠 Improved Code:")
        print(result.evolved_code)
    
    async def _demo_bug_fix(self):
        """Demo bug fix and error handling."""
        print(f"\n🐛 Demo 3: Bug Fix and Error Handling")
        print("-" * 40)
        
        code = self.demo_samples["api_handler"]
        print("📝 Original Code:")
        print(code)
        
        result = await self.engine.evolve_code(
            code=code,
            fitness_goal="Add comprehensive error handling and input validation",
            context={"target": "robustness", "constraints": ["performance"]}
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📊 Fitness Score: {result.fitness_score:.3f}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.quality_metrics:
            print(f"🔒 Security Score: {result.quality_metrics.get('security_score', 0):.1f}/10")
        
        print(f"\n🧠 Enhanced Code:")
        print(result.evolved_code)
    
    async def _demo_feature_addition(self):
        """Demo feature addition."""
        print(f"\n✨ Demo 4: Feature Addition")
        print("-" * 40)
        
        code = self.demo_samples["ml_predictor"]
        print("📝 Original Code:")
        print(code)
        
        result = await self.engine.evolve_code(
            code=code,
            fitness_goal="Add advanced ML features including cross-validation and hyperparameter tuning",
            context={"target": "feature_addition", "constraints": ["readability"]}
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📊 Fitness Score: {result.fitness_score:.3f}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.quality_metrics:
            print(f"🎯 Overall Quality: {result.quality_metrics.get('overall_score', 0):.1f}/10")
        
        print(f"\n🧠 Enhanced Code:")
        print(result.evolved_code)
    
    async def _demo_parallel_evolution(self):
        """Demo parallel evolution of multiple code samples."""
        print(f"\n🔄 Demo 5: Parallel Evolution")
        print("-" * 40)
        
        # Prepare multiple evolution tasks
        evolution_tasks = [
            (self.demo_samples["fibonacci"], "Optimize for performance", {"target": "speed"}),
            (self.demo_samples["data_processor"], "Improve readability", {"target": "clarity"}),
            (self.demo_samples["api_handler"], "Add security features", {"target": "security"})
        ]
        
        print(f"🔄 Running {len(evolution_tasks)} parallel evolutions...")
        start_time = time.time()
        
        results = await self.engine.parallel_evolution(evolution_tasks)
        
        total_time = time.time() - start_time
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📊 Average Time per Evolution: {total_time/len(results):.2f}s")
        
        for i, result in enumerate(results, 1):
            print(f"\n🔄 Evolution {i}:")
            print(f"  ✅ Success: {result.success}")
            print(f"  📊 Fitness: {result.fitness_score:.3f}")
            print(f"  🎯 Quality: {result.quality_metrics.get('overall_score', 0):.1f}/10")
    
    async def _demo_quality_analysis(self):
        """Demo code quality analysis."""
        print(f"\n🔍 Demo 6: Code Quality Analysis")
        print("-" * 40)
        
        # Analyze the evolved code from previous demos
        evolved_code = """
def fibonacci_optimized(n):
    if n <= 1:
        return n
    
    # Use dynamic programming for better performance
    fib_cache = {0: 0, 1: 1}
    
    for i in range(2, n + 1):
        fib_cache[i] = fib_cache[i-1] + fib_cache[i-2]
    
    return fib_cache[n]

def calculate_sum_optimized(numbers):
    # Use built-in sum function for better performance
    return sum(numbers)
"""
        
        print("📝 Code to Analyze:")
        print(evolved_code)
        
        quality_metrics = await self.engine.provider.analyze_code_quality(evolved_code)
        
        print(f"\n🔍 Quality Analysis Results:")
        print(f"  📊 Overall Score: {quality_metrics.get('overall_score', 0):.1f}/10")
        print(f"  ⚡ Performance Score: {quality_metrics.get('performance_score', 0):.1f}/10")
        print(f"  📖 Readability Score: {quality_metrics.get('readability_score', 0):.1f}/10")
        print(f"  🔧 Maintainability Score: {quality_metrics.get('maintainability_score', 0):.1f}/10")
        print(f"  🔒 Security Score: {quality_metrics.get('security_score', 0):.1f}/10")
        print(f"  🧮 Complexity Score: {quality_metrics.get('complexity_score', 0):.1f}/10")
        
        if quality_metrics.get('issues'):
            print(f"\n⚠️  Issues Identified:")
            for issue in quality_metrics['issues']:
                print(f"  - {issue}")
        
        if quality_metrics.get('suggestions'):
            print(f"\n💡 Suggestions:")
            for suggestion in quality_metrics['suggestions']:
                print(f"  - {suggestion}")
    
    async def _print_final_statistics(self):
        """Print final evolution statistics."""
        print(f"\n📈 Final Evolution Statistics")
        print("=" * 60)
        
        stats = self.engine.get_evolution_statistics()
        
        print(f"🔄 Total Evolutions: {stats.get('total_evolutions', 0)}")
        print(f"✅ Successful Evolutions: {stats.get('successful_evolutions', 0)}")
        print(f"❌ Failed Evolutions: {stats.get('failed_evolutions', 0)}")
        print(f"📊 Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"🎯 Average Fitness Score: {stats.get('avg_fitness_score', 0):.3f}")
        print(f"⏱️  Average Execution Time: {stats.get('avg_execution_time', 0):.2f}s")
        
        if stats.get('evolution_types'):
            print(f"\n🔄 Evolution Types:")
            for evo_type, count in stats['evolution_types'].items():
                print(f"  - {evo_type}: {count}")
        
        if stats.get('quality_metrics_summary'):
            print(f"\n🎯 Quality Metrics Summary:")
            for metric, values in stats['quality_metrics_summary'].items():
                print(f"  - {metric}: {values['mean']:.1f} ± {values['std']:.1f}")
        
        if stats.get('learning_patterns', {}).get('success_patterns'):
            print(f"\n🧠 Learning Patterns:")
            for pattern, data in stats['learning_patterns']['success_patterns'].items():
                print(f"  - {pattern}: {data['count']} successes, avg fitness {data['avg_fitness']:.3f}")


async def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Code Evolution Demo")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--provider", choices=["openai", "codellama"], 
                       default="openai", help="LLM provider to use")
    parser.add_argument("--model", default="gpt-4", help="Model name to use")
    parser.add_argument("--demo", choices=["all", "performance", "readability", "bugfix", "feature", "parallel", "quality"],
                       default="all", help="Specific demo to run")
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OpenAI API key required!")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    # Create demo instance
    demo = NeuralEvolutionDemo(api_key=api_key)
    
    # Update config if different provider/model specified
    if args.provider != "openai":
        demo.config.provider_type = args.provider
    if args.model != "gpt-4":
        demo.config.model_name = args.model
    
    try:
        if args.demo == "all":
            await demo.run_comprehensive_demo()
        else:
            # Run specific demo
            if args.demo == "performance":
                await demo._demo_performance_optimization()
            elif args.demo == "readability":
                await demo._demo_readability_improvement()
            elif args.demo == "bugfix":
                await demo._demo_bug_fix()
            elif args.demo == "feature":
                await demo._demo_feature_addition()
            elif args.demo == "parallel":
                await demo._demo_parallel_evolution()
            elif args.demo == "quality":
                await demo._demo_quality_analysis()
            
            # Print statistics even for single demos
            await demo._print_final_statistics()
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())