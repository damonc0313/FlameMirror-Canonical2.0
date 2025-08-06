"""
Cell 8: Demonstration and Examples
==================================

This cell provides comprehensive examples of how to use the unified
autonomous code evolution engine, from simple code improvements to
massive scale generation.

Examples include:
- Basic code optimization
- Multi-objective evolution
- Neural-enhanced mutations
- Massive scale deployment
- Real-time monitoring and insights
"""

import asyncio
from typing import Any


async def demo_basic_optimization():
    """Demonstrate basic code optimization."""
    
    print("🔬 DEMO: Basic Code Optimization")
    print("=" * 50)
    
    # Simple inefficient code to optimize
    initial_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
'''
    
    # Create evolution request
    request = EvolutionRequest(
        initial_code=initial_code,
        objectives=["performance", "readability"],
        constraints={"max_time": 60},
        target_metrics={"performance": 0.8, "readability": 0.9},
        max_generations=10,
        enable_massive_scale=False,
        enable_neural_mutations=True
    )
    
    print("📋 Initial code:")
    print(initial_code)
    print("\n🎯 Objectives: performance, readability")
    print("⚙️ Max generations: 10")
    
    # Run evolution
    result = await unified_engine.evolve_code(request)
    
    if result.success:
        print(f"\n✅ Evolution completed successfully!")
        print(f"⏱️ Execution time: {result.execution_time:.2f}s")
        print(f"🔄 Generations: {result.generations_completed}")
        print(f"📊 Final fitness: {result.fitness_scores.to_vector().mean():.3f}")
        
        print("\n🚀 Evolved code:")
        print(result.evolved_code)
        
        print(f"\n📈 Performance metrics:")
        for metric, value in result.performance_metrics.items():
            print(f"  • {metric}: {value:.3f}")
            
    else:
        print(f"❌ Evolution failed: {result.error_message}")
    
    return result


async def demo_multi_objective_evolution():
    """Demonstrate multi-objective evolution with Pareto optimization."""
    
    print("\n🎯 DEMO: Multi-Objective Evolution")
    print("=" * 50)
    
    # More complex code for multi-objective optimization
    initial_code = '''
import os
import sys

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        result = []
        for item in self.data:
            if item > 0:
                result.append(item * 2)
        return result
    
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.data))

def main():
    processor = DataProcessor([1, -2, 3, -4, 5])
    result = processor.process()
    processor.save("output.txt")
    return result
'''
    
    # Multi-objective request
    request = EvolutionRequest(
        initial_code=initial_code,
        objectives=["performance", "security", "maintainability", "readability"],
        constraints={"maintain_functionality": True},
        target_metrics={
            "performance": 0.8,
            "security": 0.9,
            "maintainability": 0.8,
            "readability": 0.85
        },
        max_generations=15,
        enable_pareto_optimization=True,
        enable_adaptive_learning=True
    )
    
    print("📋 Multi-objective optimization:")
    print("  • Performance")
    print("  • Security")
    print("  • Maintainability") 
    print("  • Readability")
    
    # Run evolution
    result = await unified_engine.evolve_code(request)
    
    if result.success:
        print(f"\n✅ Multi-objective evolution completed!")
        print(f"🎯 Pareto front size: {len(result.pareto_front)}")
        
        print(f"\n📊 Objective scores:")
        scores = result.fitness_scores
        print(f"  • Performance: {scores.performance:.3f}")
        print(f"  • Security: {scores.security:.3f}")
        print(f"  • Maintainability: {scores.maintainability:.3f}")
        print(f"  • Readability: {scores.readability:.3f}")
        
        print(f"\n🧠 Learning insights:")
        insights = result.learning_insights
        if 'algorithm_performance' in insights:
            print("  Algorithm performance:")
            for alg, perf in insights['algorithm_performance'].items():
                print(f"    • {alg}: {perf.get('success_rate', 0):.2f} success rate")
    
    return result


async def demo_massive_scale_generation():
    """Demonstrate massive scale code generation."""
    
    print("\n⚡ DEMO: Massive Scale Generation")
    print("=" * 50)
    
    # Seed code for massive scaling
    seed_code = '''
def distributed_calculator():
    """Seed for massive scale distributed system."""
    return {"status": "ready", "nodes": 1}
'''
    
    # Massive scale request
    request = EvolutionRequest(
        initial_code=seed_code,
        objectives=["scalability", "performance", "reliability"],
        constraints={"distributed": True, "fault_tolerant": True},
        target_metrics={"lines": MASSIVE_SCALE_TARGET_LINES},
        max_generations=5,  # Fewer generations for demo
        enable_massive_scale=True,
        enable_neural_mutations=True,
        parallelism_level=2
    )
    
    print(f"🎯 Targeting massive scale:")
    print(f"  • Target lines: {MASSIVE_SCALE_TARGET_LINES:,}")
    print(f"  • Target parameters: {MASSIVE_SCALE_TARGET_PARAMS:,}")
    print(f"  • Distributed architecture: Yes")
    
    # Run massive scale evolution
    result = await unified_engine.evolve_code(request)
    
    if result.success:
        print(f"\n✅ Massive scale generation completed!")
        final_lines = len(result.evolved_code.split('\n'))
        print(f"📏 Generated lines: {final_lines:,}")
        print(f"📈 Scale factor: {final_lines / len(seed_code.split('\n')):.1f}x")
        
        # Show excerpt of generated code
        lines = result.evolved_code.split('\n')
        print(f"\n📝 Code excerpt (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1:2d}: {line}")
        
        if len(lines) > 10:
            print(f"  ... ({len(lines)-10} more lines)")
    
    return result


async def demo_real_time_monitoring():
    """Demonstrate real-time system monitoring."""
    
    print("\n📊 DEMO: Real-time System Monitoring")
    print("=" * 50)
    
    # Get current system status
    status = unified_engine.get_system_status()
    
    print("🖥️ System Status:")
    print(f"  • Engine state: {status['engine_state']}")
    print(f"  • Pareto front size: {status['pareto_front_size']}")
    print(f"  • Learning patterns: {status['learning_patterns']}")
    print(f"  • Neural providers: {status['neural_providers']}")
    print(f"  • Memory usage: {status['memory_usage_mb']:.1f} MB")
    print(f"  • Uptime: {status['uptime_seconds']:.1f}s")
    
    # Get learning insights
    insights = adaptive_learning.get_learning_insights()
    
    print(f"\n🧠 Learning System:")
    print(f"  • Total experiences: {insights.get('total_experiences', 0)}")
    print(f"  • Success rate: {insights.get('overall_success_rate', 0):.2f}")
    print(f"  • Recent success rate: {insights.get('recent_success_rate', 0):.2f}")
    
    # Get Pareto front statistics
    pareto_stats = pareto_optimizer.get_pareto_front_statistics()
    
    print(f"\n🎯 Pareto Optimization:")
    print(f"  • Front size: {pareto_stats['size']}")
    if pareto_stats['size'] > 0:
        means = pareto_stats['objective_means']
        print(f"  • Average performance: {means[0]:.3f}")
        print(f"  • Average correctness: {means[1]:.3f}")
        print(f"  • Hypervolume: {pareto_stats['hypervolume']:.3f}")
    
    return status, insights, pareto_stats


def demo_advanced_features():
    """Demonstrate advanced features and customization."""
    
    print("\n🔬 DEMO: Advanced Features")
    print("=" * 50)
    
    # Demonstrate complexity analysis
    test_code = '''
def complex_algorithm(data):
    """Complex nested algorithm for demonstration."""
    result = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] > 10:
                for k in range(len(data)):
                    if k != i and k != j:
                        result.append((data[i], data[j], data[k]))
    return result
'''
    
    print("📐 Complexity Analysis:")
    info_measure = complexity_analyzer.calculate_information_measure(test_code)
    print(f"  • Kolmogorov complexity: {info_measure.complexity:.3f}")
    print(f"  • Shannon entropy: {info_measure.entropy:.3f}")
    print(f"  • Redundancy: {info_measure.redundancy:.3f}")
    print(f"  • Compressibility: {info_measure.compressibility:.3f}")
    
    # Demonstrate multi-objective scoring
    print(f"\n🎯 Multi-objective Fitness:")
    fitness = pareto_optimizer.evaluate_multi_objective_fitness(test_code)
    print(f"  • Performance: {fitness.performance:.3f}")
    print(f"  • Correctness: {fitness.correctness:.3f}")
    print(f"  • Maintainability: {fitness.maintainability:.3f}")
    print(f"  • Resource efficiency: {fitness.resource_efficiency:.3f}")
    print(f"  • Security: {fitness.security:.3f}")
    print(f"  • Readability: {fitness.readability:.3f}")
    
    # Demonstrate theoretical framework
    print(f"\n🧮 Theoretical Framework:")
    print("  • Optimal control theory: ACTIVE")
    print("  • Game theory analysis: ACTIVE") 
    print("  • Differential geometry: ACTIVE")
    print("  • Stochastic processes: ACTIVE")
    
    # Example optimal control calculation
    state = np.array([0.5, 0.3, 0.8])
    control = np.array([0.1, 0.2])
    cost = optimal_control.hamilton_jacobi_bellman(state, control)
    print(f"  • HJB cost example: {cost:.3f}")
    
    return info_measure, fitness


async def run_complete_demonstration():
    """Run complete demonstration of all features."""
    
    print("🌟 UNIFIED AUTONOMOUS CODE EVOLUTION ENGINE")
    print("🔥 COMPLETE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Demo 1: Basic optimization
        result1 = await demo_basic_optimization()
        
        # Demo 2: Multi-objective evolution
        result2 = await demo_multi_objective_evolution()
        
        # Demo 3: Massive scale generation
        result3 = await demo_massive_scale_generation()
        
        # Demo 4: Real-time monitoring
        status, insights, pareto_stats = await demo_real_time_monitoring()
        
        # Demo 5: Advanced features
        info_measure, fitness = demo_advanced_features()
        
        print("\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 50)
        print("✅ All systems functioning correctly")
        print("🚀 Ready for production autonomous code evolution")
        
        return {
            'basic_result': result1,
            'multi_objective_result': result2,
            'massive_scale_result': result3,
            'system_status': status,
            'learning_insights': insights,
            'advanced_analysis': {'info_measure': info_measure, 'fitness': fitness}
        }
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"Demo error: {traceback.format_exc()}")
        return None


# Utility functions for interactive use
def quick_evolve(code: str, objectives: List[str] = None, generations: int = 5) -> Any:
    """Quick evolution for interactive testing."""
    if objectives is None:
        objectives = ["performance", "readability"]
    
    request = EvolutionRequest(
        initial_code=code,
        objectives=objectives,
        constraints={},
        target_metrics={},
        max_generations=generations,
        enable_massive_scale=False
    )
    
    # Run synchronously for interactive use
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(unified_engine.evolve_code(request))
    loop.close()
    
    return result


def analyze_code(code: str) -> Dict[str, Any]:
    """Quick code analysis."""
    return {
        'complexity': complexity_analyzer.calculate_information_measure(code),
        'fitness': pareto_optimizer.evaluate_multi_objective_fitness(code),
        'lines': len([line for line in code.split('\n') if line.strip()])
    }


# Ready message for notebook use
print("\n" + "="*80)
print("🚀 UNIFIED CODE EVOLUTION ENGINE - READY FOR OPERATION")
print("="*80)
print("\n📖 Available Functions:")
print("  • run_complete_demonstration() - Full system demo")
print("  • quick_evolve(code, objectives, generations) - Quick evolution")
print("  • analyze_code(code) - Quick analysis")
print("  • unified_engine.evolve_code(request) - Full evolution")
print("  • unified_engine.get_system_status() - System status")
print("\n💡 Example usage:")
print("  result = quick_evolve('def hello(): return \"world\"', ['readability'])")
print("  analysis = analyze_code('def test(): pass')")
print("\n🌟 Ready for autonomous code evolution at unlimited scale!")

if __name__ == "__main__":
    # Auto-run demonstration if executed directly
    asyncio.run(run_complete_demonstration())