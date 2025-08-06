"""
Cell 4: Neural LLM Integration for AI-Powered Code Evolution
=============================================================

This cell integrates large language models (LLMs) specialized for code
to power intelligent mutations, optimizations, and generation.

Features:
- OpenAI Codex/GPT-4 integration
- Local code model support (Code Llama, StarCoder, etc.)
- Semantic-aware code mutations
- AI-driven optimization strategies
- Context-aware code generation
"""

class CodeLLMProvider(ABC):
    """Abstract base class for code LLM providers."""
    
    @abstractmethod
    async def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code based on prompt and context."""
        pass
    
    @abstractmethod
    async def mutate_code(self, code: str, mutation_type: str) -> str:
        """Mutate existing code."""
        pass
    
    @abstractmethod
    async def optimize_code(self, code: str, objectives: List[str]) -> str:
        """Optimize code for specific objectives."""
        pass


@dataclass
class CodeEvolutionResult:
    """Result of AI-powered code evolution."""
    original_code: str
    evolved_code: str
    fitness_improvement: float
    mutation_type: str
    optimization_objectives: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenAICodeProvider(CodeLLMProvider):
    """OpenAI-based code evolution provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if HAS_ML_LIBS and self.api_key:
            openai.api_key = self.api_key
            self.available = True
        else:
            self.available = False
            logger.warning("OpenAI API not available - missing key or libraries")
    
    async def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using GPT-4/Codex."""
        if not self.available:
            return self._fallback_generation(prompt)
        
        try:
            full_prompt = f"""
            Context: {context}
            
            Task: {prompt}
            
            Generate high-quality, optimized Python code that fulfills the requirements.
            Include proper error handling, documentation, and follow best practices.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer focused on writing efficient, clean, and well-documented code."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI code generation failed: {e}")
            return self._fallback_generation(prompt)
    
    async def mutate_code(self, code: str, mutation_type: str) -> str:
        """Mutate code using AI understanding."""
        if not self.available:
            return self._fallback_mutation(code, mutation_type)
        
        mutation_prompts = {
            "optimization": "Optimize this code for better performance while maintaining functionality",
            "refactoring": "Refactor this code to improve readability and maintainability",
            "error_handling": "Add comprehensive error handling to this code",
            "documentation": "Add detailed documentation and type hints to this code",
            "algorithmic": "Improve the algorithm used in this code for better efficiency",
            "parallelization": "Add parallelization to this code where appropriate",
            "memory_optimization": "Optimize this code for better memory usage",
            "security": "Improve the security of this code by addressing potential vulnerabilities"
        }
        
        prompt = mutation_prompts.get(mutation_type, "Improve this code")
        
        try:
            full_prompt = f"""
            {prompt}:
            
            ```python
            {code}
            ```
            
            Return only the improved code, maintaining the original functionality.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert code optimizer. Improve the given code while maintaining its original functionality."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return self._extract_code_from_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"OpenAI code mutation failed: {e}")
            return self._fallback_mutation(code, mutation_type)
    
    async def optimize_code(self, code: str, objectives: List[str]) -> str:
        """Optimize code for specific objectives."""
        if not self.available:
            return code
        
        objectives_str = ", ".join(objectives)
        
        try:
            prompt = f"""
            Optimize the following Python code for these objectives: {objectives_str}
            
            ```python
            {code}
            ```
            
            Focus on maintaining correctness while improving the specified aspects.
            Return only the optimized code.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in code optimization. Improve code quality while maintaining functionality."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )
            
            return self._extract_code_from_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"OpenAI code optimization failed: {e}")
            return code
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from LLM response."""
        # Look for code blocks
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the whole response (filtered)
        lines = response.split('\n')
        code_lines = [line for line in lines if not line.strip().startswith('#') or len(line.strip()) > 1]
        return '\n'.join(code_lines).strip()
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback code generation when API is unavailable."""
        return f'''
def generated_function():
    """
    Generated function for: {prompt}
    This is a fallback implementation when AI is unavailable.
    """
    pass
    return None
'''
    
    def _fallback_mutation(self, code: str, mutation_type: str) -> str:
        """Fallback mutation when API is unavailable."""
        # Simple rule-based mutations
        if mutation_type == "optimization":
            return f"# Optimized version\n{code}"
        elif mutation_type == "documentation":
            return f'"""\nDocumented version of the code.\n"""\n{code}'
        else:
            return f"# Mutated: {mutation_type}\n{code}"


class LocalCodeProvider(CodeLLMProvider):
    """Local code model provider (for self-hosted models)."""
    
    def __init__(self, model_endpoint: str = "http://localhost:8000"):
        self.model_endpoint = model_endpoint
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if local model is available."""
        try:
            if HAS_ML_LIBS:
                response = requests.get(f"{self.model_endpoint}/health", timeout=5)
                return response.status_code == 200
        except:
            pass
        return False
    
    async def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using local model."""
        if not self.available:
            return self._fallback_generation(prompt)
        
        try:
            payload = {
                "prompt": f"Context: {context}\nTask: {prompt}\nCode:",
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.model_endpoint}/generate", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get("generated_text", "")
            
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
        
        return self._fallback_generation(prompt)
    
    async def mutate_code(self, code: str, mutation_type: str) -> str:
        """Mutate code using local model."""
        # Similar implementation to OpenAI but with local endpoint
        return code  # Simplified for this example
    
    async def optimize_code(self, code: str, objectives: List[str]) -> str:
        """Optimize code using local model."""
        return code  # Simplified for this example
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback when local model unavailable."""
        return f"# Generated locally for: {prompt}\npass"


class NeuralCodeEvolutionEngine:
    """AI-powered code evolution engine."""
    
    def __init__(self, providers: List[CodeLLMProvider]):
        self.providers = providers
        self.available_providers = [p for p in providers if hasattr(p, 'available') and p.available]
        self.evolution_history = []
        self.fitness_cache = {}
        
    async def evolve_code(self, 
                         initial_code: str, 
                         mutation_types: List[str],
                         objectives: List[str],
                         generations: int = 10) -> CodeEvolutionResult:
        """Evolve code through multiple AI-powered mutations."""
        
        current_code = initial_code
        best_fitness = await self._evaluate_fitness(current_code, objectives)
        best_code = current_code
        
        evolution_log = []
        
        for generation in range(generations):
            # Try different mutation types
            candidates = []
            
            for mutation_type in mutation_types:
                for provider in self.available_providers:
                    try:
                        mutated_code = await provider.mutate_code(current_code, mutation_type)
                        fitness = await self._evaluate_fitness(mutated_code, objectives)
                        
                        candidates.append({
                            'code': mutated_code,
                            'fitness': fitness,
                            'mutation_type': mutation_type,
                            'provider': provider.__class__.__name__
                        })
                        
                    except Exception as e:
                        logger.error(f"Mutation failed: {e}")
                        continue
            
            # Select best candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['fitness'])
                
                if best_candidate['fitness'] > best_fitness:
                    best_fitness = best_candidate['fitness']
                    best_code = best_candidate['code']
                    current_code = best_code
                    
                    evolution_log.append({
                        'generation': generation,
                        'fitness': best_fitness,
                        'mutation_type': best_candidate['mutation_type'],
                        'provider': best_candidate['provider']
                    })
        
        fitness_improvement = best_fitness - await self._evaluate_fitness(initial_code, objectives)
        
        return CodeEvolutionResult(
            original_code=initial_code,
            evolved_code=best_code,
            fitness_improvement=fitness_improvement,
            mutation_type="multi-stage",
            optimization_objectives=objectives,
            execution_time=0.0,  # Would be measured in real implementation
            success=fitness_improvement > 0,
            metadata={'evolution_log': evolution_log}
        )
    
    async def _evaluate_fitness(self, code: str, objectives: List[str]) -> float:
        """Evaluate code fitness based on objectives."""
        cache_key = hashlib.md5((code + str(objectives)).encode()).hexdigest()
        
        if cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]
        
        fitness = 0.0
        
        # Basic fitness evaluation
        try:
            # Syntax check
            ast.parse(code)
            fitness += 1.0
            
            # Complexity analysis
            complexity = complexity_analyzer.calculate_kolmogorov_complexity(code)
            fitness += (1.0 - complexity)  # Lower complexity is better
            
            # Entropy analysis
            entropy = complexity_analyzer.shannon_entropy(code)
            fitness += entropy / 10.0  # Normalized entropy contribution
            
            # Objective-specific fitness
            for objective in objectives:
                if "performance" in objective.lower():
                    fitness += self._estimate_performance_score(code)
                elif "readability" in objective.lower():
                    fitness += self._estimate_readability_score(code)
                elif "security" in objective.lower():
                    fitness += self._estimate_security_score(code)
            
        except SyntaxError:
            fitness = 0.0
        except Exception as e:
            logger.error(f"Fitness evaluation error: {e}")
            fitness = 0.0
        
        self.fitness_cache[cache_key] = fitness
        return fitness
    
    def _estimate_performance_score(self, code: str) -> float:
        """Estimate performance score based on code patterns."""
        score = 0.5  # Base score
        
        # Look for performance patterns
        if "numpy" in code or "np." in code:
            score += 0.2
        if "vectorized" in code.lower():
            score += 0.1
        if "for" in code and "range" in code:
            score -= 0.1  # Loops can be slow
        if "list comprehension" in code.lower() or "[" in code and "for" in code:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _estimate_readability_score(self, code: str) -> float:
        """Estimate readability score."""
        score = 0.5
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Docstring bonus
        if '"""' in code or "'''" in code:
            score += 0.2
        
        # Comments bonus
        comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / max(1, len(non_empty_lines))
        score += min(0.2, comment_ratio)
        
        # Function length penalty
        if len(non_empty_lines) > 50:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _estimate_security_score(self, code: str) -> float:
        """Estimate security score."""
        score = 0.8  # Start with high security assumption
        
        # Security red flags
        dangerous_patterns = ['eval(', 'exec(', 'os.system(', 'subprocess.call(', '__import__']
        for pattern in dangerous_patterns:
            if pattern in code:
                score -= 0.2
        
        # Input validation bonus
        if 'validate' in code.lower() or 'sanitize' in code.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))


# Initialize neural code evolution system
openai_provider = OpenAICodeProvider()
local_provider = LocalCodeProvider()

neural_evolution_engine = NeuralCodeEvolutionEngine([
    openai_provider,
    local_provider
])

logger.info("🤖 Neural LLM integration initialized")
logger.info(f"🔗 Available providers: {len(neural_evolution_engine.available_providers)}")
logger.info("🧠 AI-powered code evolution ready")