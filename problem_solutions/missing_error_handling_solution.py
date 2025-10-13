"""
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
