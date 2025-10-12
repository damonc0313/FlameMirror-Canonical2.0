#!/usr/bin/env python3
"""Learning Curve Analyzer - Tracks capability improvement over time"""
import json
from collections import defaultdict
from statistics import mean

class LearningCurveAnalyzer:
    def __init__(self):
        self.curves = {}
    
    def analyze_capability_improvement(self, entries, capability):
        """Track how confidence in capability improves over time"""
        capability_entries = [e for e in entries if e.get('capability_mode') == capability]
        
        if len(capability_entries) < 2:
            return None
        
        confidences = [e.get('confidence', 0) for e in capability_entries]
        
        return {
            'capability': capability,
            'attempts': len(capability_entries),
            'first_confidence': confidences[0],
            'latest_confidence': confidences[-1],
            'improvement': confidences[-1] - confidences[0],
            'learning_rate': (confidences[-1] - confidences[0]) / len(confidences),
            'curve_type': 'improving' if confidences[-1] > confidences[0] else 'stable'
        }
    
    def analyze_all_capabilities(self, entries):
        """Analyze learning curves for all capabilities"""
        capabilities = set(e.get('capability_mode') for e in entries if e.get('capability_mode'))
        
        curves = []
        for cap in capabilities:
            curve = self.analyze_capability_improvement(entries, cap)
            if curve and curve['attempts'] > 1:
                curves.append(curve)
        
        return sorted(curves, key=lambda x: x['improvement'], reverse=True)

if __name__ == '__main__':
    analyzer = LearningCurveAnalyzer()
    print("✓ Learning curve analyzer ready")
