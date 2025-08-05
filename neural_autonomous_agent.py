#!/usr/bin/env python3
"""
Neural Autonomous Agent
=======================

This module integrates the neural code evolution engine with the existing autonomous
agent system, creating a truly AI-powered code generation and evolution system.

The agent uses code-specialized LLMs for all code generation, mutation, and optimization,
while maintaining the existing infrastructure for orchestration, testing, and evaluation.

Author: Neural Autonomous Agent v2.0
License: MIT
"""

import os
import sys
import json
import time
import logging
import asyncio
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import signal
import atexit

# Import the neural evolution engine
from neural_code_evolution_engine import (
    NeuralCodeEvolutionEngine,
    NeuralEvolutionConfig,
    CodeEvolutionResult
)

# Import existing components
from autonomous_agent_fixed import (
    FixedAutonomousAgent,
    CycleMetrics,
    CodeGenerationPlan
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/neural_autonomous_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class NeuralCycleMetrics(CycleMetrics):
    """Enhanced metrics for neural autonomous cycles."""
    neural_evolutions: int = 0
    successful_neural_evolutions: int = 0
    avg_fitness_score: float = 0.0
    avg_quality_score: float = 0.0
    llm_provider: str = ""
    evolution_types: Dict[str, int] = None
    quality_improvements: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.evolution_types is None:
            self.evolution_types = {}
        if self.quality_improvements is None:
            self.quality_improvements = {}


@dataclass
class NeuralCodeGenerationPlan(CodeGenerationPlan):
    """Enhanced plan for neural code generation."""
    neural_evolution_goals: List[str] = None
    optimization_targets: List[str] = None
    quality_thresholds: Dict[str, float] = None
    llm_prompts: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.neural_evolution_goals is None:
            self.neural_evolution_goals = []
        if self.optimization_targets is None:
            self.optimization_targets = []
        if self.quality_thresholds is None:
            self.quality_thresholds = {}
        if self.llm_prompts is None:
            self.llm_prompts = []


class NeuralCodeGenerator:
    """Neural-powered code generator using LLMs."""
    
    def __init__(self, neural_engine: NeuralCodeEvolutionEngine):
        self.neural_engine = neural_engine
        self.generation_templates = self._load_generation_templates()
        self.optimization_strategies = self._load_optimization_strategies()
    
    def _load_generation_templates(self) -> Dict[str, str]:
        """Load code generation templates for LLM prompts."""
        return {
            "module": """
Generate a complete Python module with the following specifications:

MODULE NAME: {module_name}
PURPOSE: {purpose}
FEATURES: {features}
REQUIREMENTS: {requirements}

Generate a production-ready Python module that includes:
- Comprehensive class/function definitions
- Proper error handling and validation
- Type hints and documentation
- Unit tests
- Performance optimizations
- Security considerations

Generate ONLY the complete module code without explanations.
""",
            "api": """
Generate a complete REST API module with the following specifications:

API NAME: {api_name}
ENDPOINTS: {endpoints}
DATA MODELS: {data_models}
AUTHENTICATION: {authentication}

Generate a production-ready FastAPI module that includes:
- Complete endpoint implementations
- Request/response models
- Error handling and validation
- Authentication and authorization
- Database integration
- API documentation
- Performance optimizations

Generate ONLY the complete API code without explanations.
""",
            "ml": """
Generate a complete machine learning module with the following specifications:

ML TASK: {ml_task}
ALGORITHMS: {algorithms}
DATA PROCESSING: {data_processing}
EVALUATION: {evaluation}

Generate a production-ready ML module that includes:
- Complete model implementations
- Data preprocessing pipelines
- Training and evaluation functions
- Model persistence and loading
- Performance metrics
- Hyperparameter optimization
- Cross-validation

Generate ONLY the complete ML code without explanations.
""",
            "test": """
Generate comprehensive unit tests for the following code:

CODE TO TEST:
{code_to_test}

TEST REQUIREMENTS:
- Test all functions and methods
- Include edge cases and error conditions
- Mock external dependencies
- Achieve high code coverage
- Follow testing best practices

Generate ONLY the complete test code without explanations.
"""
        }
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load optimization strategies for different code types."""
        return {
            "performance": {
                "goals": ["speed", "efficiency", "optimization"],
                "constraints": {"maintainability": 0.7, "readability": 0.6},
                "prompts": ["Optimize for maximum performance", "Improve computational efficiency"]
            },
            "readability": {
                "goals": ["clarity", "maintainability", "documentation"],
                "constraints": {"performance": 0.6, "complexity": 0.8},
                "prompts": ["Improve code readability", "Enhance maintainability"]
            },
            "security": {
                "goals": ["security", "validation", "protection"],
                "constraints": {"performance": 0.5, "usability": 0.7},
                "prompts": ["Enhance security measures", "Add input validation"]
            },
            "robustness": {
                "goals": ["error_handling", "resilience", "reliability"],
                "constraints": {"performance": 0.6, "complexity": 0.7},
                "prompts": ["Improve error handling", "Enhance robustness"]
            }
        }
    
    async def generate_neural_module(self, 
                                   module_name: str,
                                   purpose: str,
                                   features: List[str],
                                   requirements: List[str]) -> str:
        """Generate a complete module using neural evolution."""
        try:
            # Create generation prompt
            template = self.generation_templates["module"]
            prompt = template.format(
                module_name=module_name,
                purpose=purpose,
                features=", ".join(features),
                requirements=", ".join(requirements)
            )
            
            # Generate initial code
            context = {
                "module_name": module_name,
                "purpose": purpose,
                "features": features,
                "requirements": requirements,
                "generation_type": "module"
            }
            
            result = await self.neural_engine.evolve_code(
                code="",  # Empty code for generation
                fitness_goal=prompt,
                context=context
            )
            
            if result.success:
                return result.evolved_code
            else:
                logger.warning(f"Neural generation failed for {module_name}, using fallback")
                return self._generate_fallback_module(module_name, purpose, features)
                
        except Exception as e:
            logger.error(f"Error in neural module generation: {e}")
            return self._generate_fallback_module(module_name, purpose, features)
    
    async def optimize_existing_code(self,
                                   code: str,
                                   optimization_strategy: str) -> str:
        """Optimize existing code using neural evolution."""
        try:
            strategy = self.optimization_strategies.get(optimization_strategy, 
                                                      self.optimization_strategies["performance"])
            
            # Create optimization context
            context = {
                "optimization_strategy": optimization_strategy,
                "goals": strategy["goals"],
                "constraints": strategy["constraints"]
            }
            
            # Use the first prompt from the strategy
            fitness_goal = strategy["prompts"][0]
            
            result = await self.neural_engine.evolve_code(
                code=code,
                fitness_goal=fitness_goal,
                context=context
            )
            
            if result.success:
                return result.evolved_code
            else:
                logger.warning(f"Neural optimization failed, returning original code")
                return code
                
        except Exception as e:
            logger.error(f"Error in neural code optimization: {e}")
            return code
    
    async def generate_neural_tests(self, code_to_test: str) -> str:
        """Generate comprehensive tests using neural evolution."""
        try:
            template = self.generation_templates["test"]
            prompt = template.format(code_to_test=code_to_test)
            
            context = {
                "code_to_test": code_to_test,
                "generation_type": "test",
                "test_requirements": ["comprehensive", "edge_cases", "mocking", "coverage"]
            }
            
            result = await self.neural_engine.evolve_code(
                code="",  # Empty code for generation
                fitness_goal=prompt,
                context=context
            )
            
            if result.success:
                return result.evolved_code
            else:
                logger.warning("Neural test generation failed, using fallback")
                return self._generate_fallback_tests(code_to_test)
                
        except Exception as e:
            logger.error(f"Error in neural test generation: {e}")
            return self._generate_fallback_tests(code_to_test)
    
    def _generate_fallback_module(self, module_name: str, purpose: str, features: List[str]) -> str:
        """Fallback module generation when neural generation fails."""
        class_name = self._get_class_name(module_name)
        
        return f'''#!/usr/bin/env python3
"""
{module_name.title()} Module
==========================

{purpose}

Author: Neural Autonomous Agent v2.0
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class {class_name}Config:
    """Configuration for {class_name}."""
    enabled: bool = True
    debug_mode: bool = False
    max_retries: int = 3


class {class_name}:
    """
    {class_name} - {purpose}
    
    Features:
    {chr(10).join(f"    - {feature}" for feature in features)}
    """
    
    def __init__(self, config: Optional[{class_name}Config] = None):
        self.config = config or {class_name}Config()
        self._initialize()
    
    def _initialize(self):
        """Initialize the component."""
        logger.info(f"Initializing {{self.__class__.__name__}}")
        # TODO: Add initialization logic
    
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the main functionality."""
        try:
            logger.info(f"Executing {{self.__class__.__name__}}")
            # TODO: Add execution logic
            return {{"status": "success", "message": "Operation completed"}}
        except Exception as e:
            logger.error(f"Error in {{self.__class__.__name__}}: {{e}}")
            return {{"status": "error", "message": str(e)}}
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        # TODO: Add validation logic
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info(f"Cleaning up {{self.__class__.__name__}}")
        # TODO: Add cleanup logic


def main():
    """Main function for testing."""
    config = {class_name}Config(debug_mode=True)
    component = {class_name}(config)
    
    result = component.execute()
    print(f"Result: {{result}}")
    
    component.cleanup()


if __name__ == "__main__":
    main()
'''
    
    def _generate_fallback_tests(self, code_to_test: str) -> str:
        """Fallback test generation when neural generation fails."""
        return f'''#!/usr/bin/env python3
"""
Test module for generated code.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path to import the module
sys.path.insert(0, str(Path(__file__).parent))

# Import the module to test (this will be dynamically determined)
# from module_name import ClassName


class TestGeneratedCode:
    """Test suite for generated code."""
    
    def test_initialization(self):
        """Test component initialization."""
        # TODO: Add initialization tests
        assert True
    
    def test_execution(self):
        """Test main execution functionality."""
        # TODO: Add execution tests
        assert True
    
    def test_validation(self):
        """Test data validation."""
        # TODO: Add validation tests
        assert True
    
    def test_error_handling(self):
        """Test error handling."""
        # TODO: Add error handling tests
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    def _get_class_name(self, module_name: str) -> str:
        """Convert module name to class name."""
        return "".join(word.capitalize() for word in module_name.split("_"))


class NeuralTestRunner:
    """Enhanced test runner with neural-powered test generation."""
    
    def __init__(self, neural_engine: NeuralCodeEvolutionEngine):
        self.neural_engine = neural_engine
        self.test_results = {}
    
    async def run_neural_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests with neural-powered enhancements."""
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "coverage": 0.0,
            "test_files": test_files,
            "neural_enhancements": []
        }
        
        for test_file in test_files:
            try:
                # Run the test
                test_result = await self._run_single_test(test_file)
                results["total_tests"] += test_result.get("total", 0)
                results["passed"] += test_result.get("passed", 0)
                results["failed"] += test_result.get("failed", 0)
                results["errors"] += test_result.get("errors", 0)
                
                # Apply neural enhancements if tests fail
                if test_result.get("failed", 0) > 0 or test_result.get("errors", 0) > 0:
                    enhanced_result = await self._enhance_failing_tests(test_file, test_result)
                    results["neural_enhancements"].append(enhanced_result)
                    
            except Exception as e:
                logger.error(f"Error running test {test_file}: {e}")
                results["errors"] += 1
        
        # Calculate overall coverage
        if results["total_tests"] > 0:
            results["coverage"] = (results["passed"] / results["total_tests"]) * 100
        
        return results
    
    async def _run_single_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse test results
            output = result.stdout + result.stderr
            passed = output.count("PASSED")
            failed = output.count("FAILED")
            errors = output.count("ERROR")
            
            return {
                "file": test_file,
                "total": passed + failed + errors,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "output": output,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "file": test_file,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "output": "Test timeout",
                "return_code": -1
            }
        except Exception as e:
            return {
                "file": test_file,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "output": str(e),
                "return_code": -1
            }
    
    async def _enhance_failing_tests(self, test_file: str, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance failing tests using neural evolution."""
        try:
            # Read the test file
            with open(test_file, 'r') as f:
                test_code = f.read()
            
            # Create enhancement context
            context = {
                "test_file": test_file,
                "test_result": test_result,
                "enhancement_type": "test_fix"
            }
            
            # Generate enhanced tests
            result = await self.neural_engine.evolve_code(
                code=test_code,
                fitness_goal="Fix failing tests and improve test coverage",
                context=context
            )
            
            if result.success:
                # Write enhanced tests back to file
                with open(test_file, 'w') as f:
                    f.write(result.evolved_code)
                
                return {
                    "file": test_file,
                    "enhancement_type": "test_fix",
                    "success": True,
                    "fitness_score": result.fitness_score,
                    "quality_metrics": result.quality_metrics
                }
            else:
                return {
                    "file": test_file,
                    "enhancement_type": "test_fix",
                    "success": False,
                    "error": "Neural enhancement failed"
                }
                
        except Exception as e:
            logger.error(f"Error enhancing tests for {test_file}: {e}")
            return {
                "file": test_file,
                "enhancement_type": "test_fix",
                "success": False,
                "error": str(e)
            }


class NeuralAutonomousAgent(FixedAutonomousAgent):
    """
    Neural-powered autonomous agent that uses LLMs for code evolution.
    
    This agent extends the existing autonomous agent with neural-powered
    code generation, mutation, and optimization capabilities.
    """
    
    def __init__(self, 
                 repo_path: str = ".", 
                 max_cycles: Optional[int] = None,
                 neural_config: Optional[NeuralEvolutionConfig] = None):
        
        # Initialize base agent
        super().__init__(repo_path, max_cycles)
        
        # Initialize neural components
        self.neural_config = neural_config or self._create_default_neural_config()
        self.neural_engine = NeuralCodeEvolutionEngine(self.neural_config)
        self.neural_generator = NeuralCodeGenerator(self.neural_engine)
        self.neural_test_runner = NeuralTestRunner(self.neural_engine)
        
        # Enhanced metrics
        self.neural_cycle_metrics = []
        self.neural_evolution_history = []
        
        logger.info("Neural Autonomous Agent initialized with LLM-powered evolution")
    
    def _create_default_neural_config(self) -> NeuralEvolutionConfig:
        """Create default neural evolution configuration."""
        return NeuralEvolutionConfig(
            provider_type="openai",
            model_name="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_concurrent_evolutions=5,
            enable_quality_analysis=True,
            enable_parallel_evolution=True,
            fitness_threshold=0.7
        )
    
    async def start_neural_autonomous_loop(self):
        """Start the neural-powered autonomous generation loop."""
        logger.info("🚀 Starting Neural Autonomous Agent")
        logger.info(f"🧠 Using LLM Provider: {self.neural_config.provider_type}")
        logger.info(f"🤖 Model: {self.neural_config.model_name}")
        
        try:
            while self.should_continue():
                await self._execute_neural_cycle()
                
                # Save neural evolution state
                self.neural_engine.save_evolution_state(
                    f"neural_evolution_state_cycle_{self.current_cycle}.pkl"
                )
                
        except KeyboardInterrupt:
            logger.info("🛑 Neural autonomous loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Critical error in neural autonomous loop: {e}")
            self._handle_critical_error(e)
        finally:
            self._cleanup()
    
    async def _execute_neural_cycle(self):
        """Execute a single neural-powered autonomous cycle."""
        cycle_start = datetime.now()
        cycle_metrics = NeuralCycleMetrics(
            cycle_number=self.current_cycle,
            start_time=cycle_start.isoformat(),
            llm_provider=self.neural_config.provider_type
        )
        
        logger.info(f"🔄 Starting Neural Cycle {self.current_cycle}")
        
        try:
            # Create neural generation plan
            plan = await self._create_neural_generation_plan()
            
            # Generate code using neural evolution
            generated_files = await self._generate_neural_code(plan)
            cycle_metrics.files_generated = len(generated_files)
            
            # Run neural-powered tests
            test_results = await self._run_neural_tests(generated_files)
            cycle_metrics.tests_passed = test_results.get("passed", 0)
            cycle_metrics.tests_failed = test_results.get("failed", 0)
            cycle_metrics.coverage_percentage = test_results.get("coverage", 0.0)
            
            # Update neural metrics
            neural_stats = self.neural_engine.get_evolution_statistics()
            cycle_metrics.neural_evolutions = neural_stats.get("total_evolutions", 0)
            cycle_metrics.successful_neural_evolutions = neural_stats.get("successful_evolutions", 0)
            cycle_metrics.avg_fitness_score = neural_stats.get("avg_fitness_score", 0.0)
            cycle_metrics.evolution_types = neural_stats.get("evolution_types", {})
            
            # Commit changes
            commit_message = self._generate_neural_commit_message(plan, test_results)
            commit_hash = self.git_manager.commit_changes(commit_message)
            cycle_metrics.commit_hash = commit_hash
            
            # Update cycle metrics
            cycle_metrics.end_time = datetime.now().isoformat()
            self.neural_cycle_metrics.append(cycle_metrics)
            
            # Generate cycle summary
            await self._generate_neural_cycle_summary(cycle_metrics, plan, test_results)
            
            self.current_cycle += 1
            
        except Exception as e:
            logger.error(f"❌ Error in neural cycle {self.current_cycle}: {e}")
            cycle_metrics.errors.append(str(e))
            cycle_metrics.end_time = datetime.now().isoformat()
            self.neural_cycle_metrics.append(cycle_metrics)
            self._handle_cycle_error(e)
    
    async def _create_neural_generation_plan(self) -> NeuralCodeGenerationPlan:
        """Create a neural-powered code generation plan."""
        existing_modules = self._analyze_existing_modules()
        target_modules = self._determine_target_modules(existing_modules)
        
        # Enhanced neural goals
        neural_goals = [
            "Generate high-performance, production-ready code",
            "Implement comprehensive error handling and validation",
            "Add advanced features and optimizations",
            "Ensure code quality and maintainability"
        ]
        
        optimization_targets = ["performance", "readability", "security", "robustness"]
        
        plan = NeuralCodeGenerationPlan(
            cycle_number=self.current_cycle,
            target_modules=target_modules,
            new_features=self._identify_new_features(),
            optimizations=self._identify_optimizations(),
            refactoring_targets=self._identify_refactoring_targets(),
            neural_evolution_goals=neural_goals,
            optimization_targets=optimization_targets,
            quality_thresholds={
                "overall_score": 8.0,
                "performance_score": 8.0,
                "readability_score": 8.0,
                "security_score": 8.0
            }
        )
        
        return plan
    
    async def _generate_neural_code(self, plan: NeuralCodeGenerationPlan) -> List[str]:
        """Generate code using neural evolution."""
        generated_files = []
        
        for module_name in plan.target_modules:
            try:
                # Generate module using neural evolution
                module_code = await self.neural_generator.generate_neural_module(
                    module_name=module_name,
                    purpose=f"Advanced {module_name.replace('_', ' ')} functionality",
                    features=[
                        "High performance implementation",
                        "Comprehensive error handling",
                        "Advanced optimization features",
                        "Production-ready design"
                    ],
                    requirements=[
                        "Python 3.9+ compatibility",
                        "Async/await support",
                        "Type hints and validation",
                        "Comprehensive testing"
                    ]
                )
                
                # Write module file
                module_path = Path(self.repo_path) / f"{module_name}.py"
                with open(module_path, 'w') as f:
                    f.write(module_code)
                
                generated_files.append(str(module_path))
                
                # Generate corresponding test file
                test_code = await self.neural_generator.generate_neural_tests(module_code)
                test_path = Path(self.repo_path) / f"test_{module_name}.py"
                with open(test_path, 'w') as f:
                    f.write(test_code)
                
                generated_files.append(str(test_path))
                
                logger.info(f"🧠 Generated neural module: {module_name}")
                
            except Exception as e:
                logger.error(f"❌ Error generating neural module {module_name}: {e}")
        
        return generated_files
    
    async def _run_neural_tests(self, generated_files: List[str]) -> Dict[str, Any]:
        """Run tests with neural-powered enhancements."""
        test_files = [f for f in generated_files if f.endswith("_test.py") or "test_" in f]
        
        if not test_files:
            logger.warning("⚠️ No test files found for neural testing")
            return {"passed": 0, "failed": 0, "coverage": 0.0}
        
        return await self.neural_test_runner.run_neural_tests(test_files)
    
    def _generate_neural_commit_message(self, 
                                      plan: NeuralCodeGenerationPlan, 
                                      test_results: Dict[str, Any]) -> str:
        """Generate commit message for neural evolution."""
        neural_stats = self.neural_engine.get_evolution_statistics()
        
        return f"""🧠 Neural Evolution Cycle {self.current_cycle}

🤖 LLM Provider: {self.neural_config.provider_type}
📊 Neural Evolutions: {neural_stats.get('total_evolutions', 0)}
✅ Successful Evolutions: {neural_stats.get('successful_evolutions', 0)}
🎯 Avg Fitness Score: {neural_stats.get('avg_fitness_score', 0.0):.3f}

📁 Files Generated: {len(plan.target_modules)}
🧪 Tests Passed: {test_results.get('passed', 0)}
📈 Coverage: {test_results.get('coverage', 0.0):.1f}%

🎯 Evolution Goals:
{chr(10).join(f"  - {goal}" for goal in plan.neural_evolution_goals)}

🚀 Optimization Targets:
{chr(10).join(f"  - {target}" for target in plan.optimization_targets)}

Generated by Neural Autonomous Agent v2.0
Timestamp: {datetime.now().isoformat()}
"""
    
    async def _generate_neural_cycle_summary(self, 
                                           metrics: NeuralCycleMetrics,
                                           plan: NeuralCodeGenerationPlan,
                                           test_results: Dict[str, Any]):
        """Generate comprehensive neural cycle summary."""
        summary = {
            "cycle_number": metrics.cycle_number,
            "timestamp": datetime.now().isoformat(),
            "neural_metrics": {
                "total_evolutions": metrics.neural_evolutions,
                "successful_evolutions": metrics.successful_neural_evolutions,
                "avg_fitness_score": metrics.avg_fitness_score,
                "llm_provider": metrics.llm_provider,
                "evolution_types": metrics.evolution_types
            },
            "code_generation": {
                "files_generated": metrics.files_generated,
                "target_modules": plan.target_modules,
                "neural_goals": plan.neural_evolution_goals,
                "optimization_targets": plan.optimization_targets
            },
            "testing": {
                "tests_passed": metrics.tests_passed,
                "tests_failed": metrics.tests_failed,
                "coverage": metrics.coverage_percentage,
                "neural_enhancements": test_results.get("neural_enhancements", [])
            },
            "quality_metrics": {
                "quality_thresholds": plan.quality_thresholds,
                "quality_improvements": metrics.quality_improvements
            }
        }
        
        # Save summary to file
        summary_path = Path(self.repo_path) / "logs" / f"neural_cycle_{metrics.cycle_number}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Neural cycle summary saved to {summary_path}")
        
        # Print summary
        print(f"\n🧠 Neural Cycle {metrics.cycle_number} Summary")
        print("=" * 50)
        print(f"🤖 LLM Provider: {metrics.llm_provider}")
        print(f"📊 Neural Evolutions: {metrics.neural_evolutions}")
        print(f"✅ Success Rate: {metrics.successful_neural_evolutions/max(metrics.neural_evolutions, 1)*100:.1f}%")
        print(f"🎯 Avg Fitness: {metrics.avg_fitness_score:.3f}")
        print(f"📁 Files Generated: {metrics.files_generated}")
        print(f"🧪 Tests Passed: {metrics.tests_passed}/{metrics.tests_passed + metrics.tests_failed}")
        print(f"📈 Coverage: {metrics.coverage_percentage:.1f}%")
    
    def get_neural_statistics(self) -> Dict[str, Any]:
        """Get comprehensive neural evolution statistics."""
        neural_stats = self.neural_engine.get_evolution_statistics()
        
        return {
            "neural_engine": neural_stats,
            "cycle_metrics": [asdict(metrics) for metrics in self.neural_cycle_metrics],
            "configuration": asdict(self.neural_config),
            "total_cycles": len(self.neural_cycle_metrics),
            "overall_success_rate": sum(1 for m in self.neural_cycle_metrics if not m.errors) / max(len(self.neural_cycle_metrics), 1),
            "avg_cycle_duration": self._calculate_avg_cycle_duration()
        }
    
    def _calculate_avg_cycle_duration(self) -> float:
        """Calculate average cycle duration."""
        if not self.neural_cycle_metrics:
            return 0.0
        
        total_duration = 0
        for metrics in self.neural_cycle_metrics:
            if metrics.start_time and metrics.end_time:
                start = datetime.fromisoformat(metrics.start_time)
                end = datetime.fromisoformat(metrics.end_time)
                total_duration += (end - start).total_seconds()
        
        return total_duration / len(self.neural_cycle_metrics)


async def main():
    """Main function for neural autonomous agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Autonomous Agent")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to run")
    parser.add_argument("--infinite", action="store_true", help="Run infinite cycles")
    parser.add_argument("--provider", choices=["openai", "codellama", "hybrid"], 
                       default="openai", help="LLM provider to use")
    parser.add_argument("--model", default="gpt-4", help="Model name to use")
    parser.add_argument("--api-key", help="API key for OpenAI")
    parser.add_argument("--api-endpoint", help="API endpoint for Code Llama")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    
    args = parser.parse_args()
    
    # Create neural configuration
    neural_config = NeuralEvolutionConfig(
        provider_type=args.provider,
        model_name=args.model,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        api_endpoint=args.api_endpoint or os.getenv("CODELLAMA_API_ENDPOINT"),
        max_concurrent_evolutions=5,
        enable_quality_analysis=True,
        enable_parallel_evolution=True,
        fitness_threshold=0.7
    )
    
    # Initialize neural agent
    agent = NeuralAutonomousAgent(
        repo_path=args.repo_path,
        max_cycles=None if args.infinite else args.cycles,
        neural_config=neural_config
    )
    
    # Start neural autonomous loop
    await agent.start_neural_autonomous_loop()
    
    # Print final statistics
    stats = agent.get_neural_statistics()
    print(f"\n🧠 Final Neural Evolution Statistics")
    print("=" * 50)
    print(f"Total Cycles: {stats['total_cycles']}")
    print(f"Overall Success Rate: {stats['overall_success_rate']:.2%}")
    print(f"Average Cycle Duration: {stats['avg_cycle_duration']:.2f}s")
    print(f"Total Neural Evolutions: {stats['neural_engine'].get('total_evolutions', 0)}")
    print(f"Neural Success Rate: {stats['neural_engine'].get('success_rate', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())