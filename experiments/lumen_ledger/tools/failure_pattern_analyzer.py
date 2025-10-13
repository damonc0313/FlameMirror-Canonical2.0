#!/usr/bin/env python3
"""Failure Pattern Analyzer - Learns from bugs to prevent future ones"""
class FailurePatternAnalyzer:
    def __init__(self):
        self.known_failure_patterns = []
    
    def analyze_failure(self, error_msg, context):
        """Extract pattern from failure"""
        pattern = {
            'error_type': self._classify_error(error_msg),
            'context': context,
            'prevention': self._suggest_prevention(error_msg),
            'seen_count': 1
        }
        return pattern
    
    def _classify_error(self, error_msg):
        """Classify error type"""
        if 'TypeError' in error_msg:
            return 'type_error'
        elif 'NameError' in error_msg:
            return 'name_error'
        elif 'AttributeError' in error_msg:
            return 'attribute_error'
        return 'unknown'
    
    def _suggest_prevention(self, error_msg):
        """Suggest how to prevent this error"""
        if 'not defined' in error_msg:
            return 'Add missing import statement'
        elif 'takes exactly one argument' in error_msg:
            return 'Fix function call syntax'
        return 'Review code carefully'
    
    def learn_from_failures(self, bug_history):
        """Learn patterns from historical bugs"""
        patterns = {}
        for bug in bug_history:
            error_type = self._classify_error(bug.get('error', ''))
            patterns[error_type] = patterns.get(error_type, 0) + 1
        
        return {
            'most_common_errors': sorted(patterns.items(), key=lambda x: x[1], reverse=True),
            'prevention_recommendations': ['Add type hints', 'Import validation', 'Test before deploy']
        }

if __name__ == '__main__':
    analyzer = FailurePatternAnalyzer()
    print("✓ Failure pattern analyzer ready")
