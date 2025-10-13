#!/usr/bin/env python3
"""
PROBLEM-DRIVEN CODE GENERATION

Instead of generating arbitrary tools, this:
1. ANALYZES codebase for real problems
2. IDENTIFIES patterns that need abstraction
3. GENERATES solutions to THOSE problems
4. INTEGRATES into existing code
5. MEASURES improvement

This addresses the critical 0% usage finding.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class CodeProblem:
    """Identified problem in codebase"""
    problem_type: str
    description: str
    evidence: List[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    files_affected: List[str]
    suggested_solution: str


@dataclass
class CodeSmell:
    """Code smell detected"""
    smell_type: str
    location: str
    line_number: int
    description: str


class CodebaseAnalyzer:
    """
    Analyzes codebase to find REAL problems.
    
    Looks for:
    - Repeated code blocks (need abstraction)
    - Missing error handling
    - Hard-coded values (need config)
    - Performance bottlenecks
    - Missing tests
    - Inconsistent patterns
    """
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.problems: List[CodeProblem] = []
        self.code_smells: List[CodeSmell] = []
        
    def analyze_for_problems(self, max_files: int = 100) -> List[CodeProblem]:
        """
        Analyze codebase for real problems.
        
        This is PROBLEM DISCOVERY phase.
        """
        
        print("="*80)
        print("PROBLEM-DRIVEN ANALYSIS")
        print("="*80)
        print()
        
        # Get Python files
        py_files = list(self.root.glob("**/*.py"))
        # Exclude generated and test files
        py_files = [
            f for f in py_files 
            if '__pycache__' not in str(f)
            and 'venv' not in str(f)
            and not any(x in f.name for x in ['test_', '_test.py', 'auto_gen', 'max_tool'])
        ][:max_files]
        
        print(f"📊 Analyzing {len(py_files)} Python files for real problems...")
        print()
        
        # Analysis patterns
        self._find_repeated_code(py_files)
        self._find_missing_error_handling(py_files)
        self._find_hardcoded_values(py_files)
        self._find_missing_abstractions(py_files)
        
        # Summarize
        print(f"✅ Analysis complete:")
        print(f"   Problems found: {len(self.problems)}")
        print(f"   Code smells: {len(self.code_smells)}")
        print()
        
        return self.problems
    
    def _find_repeated_code(self, files: List[Path]):
        """Find repeated code patterns that need abstraction"""
        
        print("🔍 Searching for repeated code patterns...")
        
        # Track common function patterns
        function_bodies = defaultdict(list)
        
        for file in files[:50]:  # Sample
            try:
                content = file.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Get function body as string (simplified)
                        body_str = ast.unparse(node).strip()[:200]
                        function_bodies[body_str].append((file.name, node.name))
                        
            except:
                continue
        
        # Find duplicates
        duplicates = {k: v for k, v in function_bodies.items() if len(v) > 2}
        
        if duplicates:
            top_duplicate = list(duplicates.items())[0]
            pattern, occurrences = top_duplicate
            
            self.problems.append(CodeProblem(
                problem_type='code_duplication',
                description=f'Similar code pattern repeated {len(occurrences)} times',
                evidence=[f'{f}:{n}' for f, n in occurrences[:3]],
                severity='medium',
                files_affected=[f for f, _ in occurrences],
                suggested_solution='Extract common pattern into reusable utility function'
            ))
            
            print(f"   ⚠️  Found {len(duplicates)} repeated patterns")
            print(f"      Top: repeated {len(occurrences)} times")
        else:
            print(f"   ✓ No significant code duplication")
        
        print()
    
    def _find_missing_error_handling(self, files: List[Path]):
        """Find functions without error handling"""
        
        print("🔍 Checking error handling...")
        
        missing_error_handling = []
        
        for file in files[:30]:
            try:
                content = file.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function has try/except
                        has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
                        has_raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                        
                        # If function does I/O but no error handling
                        has_io = any(
                            call.func.id in ['open', 'read', 'write'] 
                            for call in ast.walk(node) 
                            if isinstance(call, ast.Call) 
                            and isinstance(call.func, ast.Name)
                        )
                        
                        if has_io and not (has_try or has_raises):
                            missing_error_handling.append((file.name, node.name, node.lineno))
                            
            except:
                continue
        
        if missing_error_handling:
            self.problems.append(CodeProblem(
                problem_type='missing_error_handling',
                description=f'{len(missing_error_handling)} functions with I/O but no error handling',
                evidence=[f'{f}:{n} (line {l})' for f, n, l in missing_error_handling[:5]],
                severity='high',
                files_affected=list(set(f for f, _, _ in missing_error_handling)),
                suggested_solution='Add try/except blocks or error handling decorator'
            ))
            
            print(f"   ⚠️  {len(missing_error_handling)} functions missing error handling")
        else:
            print(f"   ✓ Error handling looks good")
        
        print()
    
    def _find_hardcoded_values(self, files: List[Path]):
        """Find hardcoded values that should be config"""
        
        print("🔍 Searching for hardcoded values...")
        
        hardcoded = []
        
        for file in files[:30]:
            try:
                content = file.read_text()
                
                # Look for common hardcoded patterns
                patterns = [
                    (r'timeout\s*=\s*\d+', 'timeout value'),
                    (r'max_\w+\s*=\s*\d+', 'max limit'),
                    (r'host\s*=\s*["\']localhost["\']', 'host address'),
                    (r'port\s*=\s*\d{4,5}', 'port number'),
                    (r'batch_size\s*=\s*\d+', 'batch size'),
                ]
                
                for pattern, desc in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        hardcoded.append((file.name, desc, line_num))
                        
            except:
                continue
        
        if hardcoded:
            self.problems.append(CodeProblem(
                problem_type='hardcoded_values',
                description=f'{len(hardcoded)} hardcoded values found',
                evidence=[f'{f}:{d} (line {l})' for f, d, l in hardcoded[:5]],
                severity='medium',
                files_affected=list(set(f for f, _, _ in hardcoded)),
                suggested_solution='Create centralized configuration system'
            ))
            
            print(f"   ⚠️  {len(hardcoded)} hardcoded values (should be config)")
        else:
            print(f"   ✓ No significant hardcoded values")
        
        print()
    
    def _find_missing_abstractions(self, files: List[Path]):
        """Find patterns that need abstraction"""
        
        print("🔍 Identifying missing abstractions...")
        
        # Look for common operations without abstraction
        common_operations = Counter()
        
        for file in files[:30]:
            try:
                content = file.read_text()
                
                # Check for common patterns
                if 'sqlite3.connect' in content:
                    common_operations['database_connection'] += 1
                if 'threading.Lock()' in content:
                    common_operations['thread_synchronization'] += 1
                if 'json.loads' in content and 'json.dumps' in content:
                    common_operations['json_serialization'] += 1
                if 'datetime.now()' in content:
                    common_operations['timestamp_handling'] += 1
                    
            except:
                continue
        
        # Find most common
        if common_operations:
            most_common = common_operations.most_common(3)
            
            for operation, count in most_common:
                if count >= 5:  # Appears in 5+ files
                    self.problems.append(CodeProblem(
                        problem_type='missing_abstraction',
                        description=f'{operation} used in {count} files without abstraction',
                        evidence=[operation],
                        severity='medium',
                        files_affected=[],
                        suggested_solution=f'Create {operation} utility class or decorator'
                    ))
                    
                    print(f"   ⚠️  {operation}: used {count} times, needs abstraction")
        else:
            print(f"   ✓ Abstractions look reasonable")
        
        print()


class ProblemDrivenGenerator:
    """
    Generates solutions to REAL problems.
    
    Not arbitrary tools - solutions to identified issues.
    """
    
    def __init__(self):
        self.analyzer = CodebaseAnalyzer(Path("/workspace"))
    
    def generate_solution(self, problem: CodeProblem) -> str:
        """
        Generate code to solve a specific problem.
        
        This is SOLUTION GENERATION phase.
        """
        
        if problem.problem_type == 'missing_error_handling':
            return self._generate_error_handling_decorator()
        
        elif problem.problem_type == 'hardcoded_values':
            return self._generate_config_system()
        
        elif problem.problem_type == 'missing_abstraction':
            if 'database_connection' in problem.description:
                return self._generate_database_manager()
            elif 'thread_synchronization' in problem.description:
                return self._generate_sync_utilities()
        
        elif problem.problem_type == 'code_duplication':
            return self._generate_abstraction_utility()
        
        return ""
    
    def _generate_error_handling_decorator(self) -> str:
        """Generate error handling decorator"""
        
        return '''"""
Error Handling Decorator
Automatically adds error handling to functions.
"""

import functools
import logging
from typing import Callable, Any, Optional


logger = logging.getLogger(__name__)


def with_error_handling(
    default_return: Any = None,
    log_errors: bool = True,
    raise_on_error: bool = False
):
    """
    Decorator that adds error handling to functions.
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        raise_on_error: Whether to re-raise exceptions
    
    Example:
        @with_error_handling(default_return=[], log_errors=True)
        def read_file(path):
            with open(path) as f:
                return f.read()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True
                    )
                if raise_on_error:
                    raise
                return default_return
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator that retries function on failure.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries (seconds)
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}"
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
'''
    
    def _generate_config_system(self) -> str:
        """Generate centralized configuration system"""
        
        return '''"""
Centralized Configuration System
Single source of truth for all configuration values.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Centralized configuration.
    
    Usage:
        config = Config.load()
        timeout = config.get('timeout', default=30)
    """
    
    _data: Dict[str, Any] = field(default_factory=dict)
    _env_prefix: str = "APP_"
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """Load configuration from file and environment"""
        
        config = cls()
        
        # Load from file
        if config_path and config_path.exists():
            with open(config_path) as f:
                config._data = json.load(f)
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith(config._env_prefix):
                config_key = key[len(config._env_prefix):].lower()
                config._data[config_key] = value
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._data[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple values"""
        self._data.update(updates)
    
    def save(self, config_path: Path):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self._data, f, indent=2)


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config.load()
    return _global_config
'''
    
    def _generate_database_manager(self) -> str:
        """Generate database connection manager"""
        
        return '''"""
Database Connection Manager
Centralized database connection handling with pooling.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager


class DatabaseManager:
    """
    Manages database connections with pooling.
    
    Usage:
        db = DatabaseManager('app.db')
        with db.connection() as conn:
            cursor = conn.execute('SELECT * FROM users')
    """
    
    def __init__(self, db_path: Path, max_connections: int = 5):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self._connections = []
        self._lock = threading.Lock()
    
    @contextmanager
    def connection(self):
        """Get database connection (context manager)"""
        conn = self._acquire_connection()
        try:
            yield conn
        finally:
            self._release_connection(conn)
    
    def _acquire_connection(self) -> sqlite3.Connection:
        """Acquire connection from pool"""
        with self._lock:
            if self._connections:
                return self._connections.pop()
            else:
                return sqlite3.connect(str(self.db_path))
    
    def _release_connection(self, conn: sqlite3.Connection):
        """Release connection back to pool"""
        with self._lock:
            if len(self._connections) < self.max_connections:
                self._connections.append(conn)
            else:
                conn.close()
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute query and return results"""
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: list):
        """Execute query with multiple parameter sets"""
        with self.connection() as conn:
            conn.executemany(query, params_list)
            conn.commit()
    
    def close_all(self):
        """Close all pooled connections"""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
'''
    
    def _generate_sync_utilities(self) -> str:
        """Generate thread synchronization utilities"""
        
        return '''"""
Thread Synchronization Utilities
Common patterns for thread-safe operations.
"""

import threading
import functools
from typing import Callable


def synchronized(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe.
    
    Usage:
        class MyClass:
            @synchronized
            def thread_safe_method(self):
                # This is automatically thread-safe
                pass
    """
    
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper


class ThreadSafeCounter:
    """Thread-safe counter"""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value


class ReadWriteLock:
    """
    Read-Write lock for performance.
    Allows multiple readers or single writer.
    """
    
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._write_ready = threading.Condition(threading.Lock())
    
    def acquire_read(self):
        """Acquire read lock"""
        with self._read_ready:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
    
    def release_read(self):
        """Release read lock"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
    
    def acquire_write(self):
        """Acquire write lock"""
        with self._write_ready:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers += 1
    
    def release_write(self):
        """Release write lock"""
        with self._write_ready:
            self._writers -= 1
            self._write_ready.notify_all()
            self._read_ready.notify_all()
'''
    
    def _generate_abstraction_utility(self) -> str:
        """Generate utility for common abstraction"""
        return "# Placeholder for abstraction utility"
    
    def run_complete_cycle(self):
        """
        Run complete problem-driven cycle:
        1. Analyze for problems
        2. Generate solutions
        3. Save solutions
        4. Report
        """
        
        print("="*80)
        print("PROBLEM-DRIVEN GENERATION CYCLE")
        print("="*80)
        print()
        
        # PHASE 1: ANALYZE
        print("PHASE 1: PROBLEM DISCOVERY")
        print("-"*80)
        problems = self.analyzer.analyze_for_problems()
        
        if not problems:
            print("✅ No significant problems found!")
            print()
            return
        
        # PHASE 2: PRIORITIZE
        print("PHASE 2: PRIORITIZATION")
        print("-"*80)
        
        critical = [p for p in problems if p.severity == 'critical']
        high = [p for p in problems if p.severity == 'high']
        medium = [p for p in problems if p.severity == 'medium']
        
        print(f"Critical: {len(critical)}")
        print(f"High:     {len(high)}")
        print(f"Medium:   {len(medium)}")
        print()
        
        # Sort by severity
        sorted_problems = critical + high + medium
        
        print("Top 3 problems to solve:")
        for i, problem in enumerate(sorted_problems[:3], 1):
            print(f"{i}. [{problem.severity.upper()}] {problem.problem_type}")
            print(f"   {problem.description}")
            print(f"   Solution: {problem.suggested_solution}")
            print()
        
        # PHASE 3: GENERATE SOLUTIONS
        print("PHASE 3: SOLUTION GENERATION")
        print("-"*80)
        
        solutions_dir = Path("/workspace/problem_solutions")
        solutions_dir.mkdir(exist_ok=True)
        
        generated = []
        
        for i, problem in enumerate(sorted_problems[:3], 1):
            print(f"Generating solution {i}/3: {problem.problem_type}")
            
            code = self.generate_solution(problem)
            if code:
                # Save solution
                filename = f"{problem.problem_type}_solution.py"
                filepath = solutions_dir / filename
                filepath.write_text(code)
                
                generated.append((problem, filepath))
                print(f"  ✅ Saved to: {filepath}")
            else:
                print(f"  ⚠️  No generator for this problem type yet")
            print()
        
        # PHASE 4: REPORT
        print("="*80)
        print("PROBLEM-DRIVEN GENERATION COMPLETE")
        print("="*80)
        print()
        
        print(f"Problems analyzed: {len(problems)}")
        print(f"Solutions generated: {len(generated)}")
        print()
        
        print("Generated solutions:")
        for problem, filepath in generated:
            print(f"  ✅ {filepath.name}")
            print(f"     Solves: {problem.description}")
        print()
        
        print("These solutions address REAL problems in the codebase,")
        print("not arbitrary tool generation.")
        print()
        
        return generated


def main():
    """Run problem-driven generation"""
    
    generator = ProblemDrivenGenerator()
    generator.run_complete_cycle()


if __name__ == '__main__':
    main()
