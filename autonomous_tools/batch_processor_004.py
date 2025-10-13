#!/usr/bin/env python3
"""Parallel batch processor with progress tracking."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int
    succeeded: int
    failed: int
    elapsed: float
    results: List[Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.succeeded / self.total if self.total > 0 else 0.0


class BatchProcessor:
    """Process items in parallel with configurable workers."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
    
    def process(
        self,
        items: List[Any],
        func: Callable,
        show_progress: bool = False
    ) -> BatchResult:
        """Process items in parallel."""
        start = time.time()
        succeeded = 0
        failed = 0
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(func, item): item for item in items}
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    succeeded += 1
                except Exception as e:
                    results.append(None)
                    failed += 1
                
                if show_progress:
                    print(f"Progress: {i}/{len(items)} ({succeeded} OK, {failed} failed)")
        
        elapsed = time.time() - start
        return BatchResult(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            elapsed=elapsed,
            results=results
        )


if __name__ == '__main__':
    # Example usage
    def process_item(x: int) -> int:
        time.sleep(0.1)
        return x * 2
    
    processor = BatchProcessor(max_workers=5)
    items = list(range(20))
    
    result = processor.process(items, process_item, show_progress=True)
    print(f"\nProcessed {result.total} items in {result.elapsed:.2f}s")
    print(f"Success rate: {result.success_rate:.1%}")
