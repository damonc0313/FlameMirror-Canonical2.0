#!/usr/bin/env python3
"""Breakthrough Detector - Automatically identifies breakthrough moments in research"""
class BreakthroughDetector:
    def __init__(self):
        self.breakthrough_criteria = {
            "high_confidence": 0.92,
            "novelty_threshold": 0.8,
            "impact_threshold": 3  # affects 3+ other entries
        }
    
    def is_breakthrough(self, entry):
        """Determine if entry represents a breakthrough"""
        if entry.get('confidence', 0) >= self.breakthrough_criteria['high_confidence']:
            if any(keyword in entry.get('observation', '').lower() 
                   for keyword in ['first', 'breakthrough', 'validated', 'discovered']):
                return True
        return False
    
    def scan_ledger(self, entries):
        """Scan all entries for breakthroughs"""
        breakthroughs = [e for e in entries if self.is_breakthrough(e)]
        return breakthroughs

if __name__ == '__main__':
    detector = BreakthroughDetector()
    print("✓ Breakthrough detector ready")
    print(f"  Criteria: conf >= {detector.breakthrough_criteria['high_confidence']}")
