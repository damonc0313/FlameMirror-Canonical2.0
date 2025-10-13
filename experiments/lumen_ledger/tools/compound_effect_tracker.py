#!/usr/bin/env python3
"""Compound Effect Tracker - Measures recursive improvement compounding"""

class CompoundEffectTracker:
    def track_iterations(self, iterations: List[Dict]) -> Dict:
        """Track if improvement compounds"""
        
        if len(iterations) < 2:
            return {'status': 'insufficient_data'}
        
        # Measure improvement rate
        improvements = []
        for i in range(1, len(iterations)):
            improvement = iterations[i]['performance'] / iterations[i-1]['performance']
            improvements.append(improvement)
        
        # Check if accelerating
        if len(improvements) >= 2:
            early_improvement = improvements[0]
            late_improvement = improvements[-1]
            
            if late_improvement > early_improvement * 1.2:
                return {
                    'status': 'compounding',
                    'acceleration': late_improvement / early_improvement,
                    'insight': 'Improvement rate itself is improving'
                }
        
        return {'status': 'linear', 'insight': 'Steady improvement'}

if __name__ == '__main__':
    tracker = CompoundEffectTracker()
    
    # Simulate my iterations
    iters = [
        {'iteration': 1, 'performance': 1.0},
        {'iteration': 2, 'performance': 1.4},
        {'iteration': 3, 'performance': 2.1}
    ]
    
    result = tracker.track_iterations(iters)
    print(f"✓ Compound tracking: {result['status']}")
    if 'acceleration' in result:
        print(f"  Acceleration: {result['acceleration']:.2f}x")
        print(f"  {result['insight']}")
