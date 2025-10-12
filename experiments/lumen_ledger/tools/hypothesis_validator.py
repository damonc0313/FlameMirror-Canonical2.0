#!/usr/bin/env python3
"""
Hypothesis Validator
Tracks hypothesis validation status across entries

Tests:
- How many entries support each hypothesis
- Confidence trends per hypothesis  
- Cross-author validation
- Time to validation
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict


class HypothesisValidator:
    """Validates research hypotheses from ledger data"""
    
    def __init__(self, ledger_path: Path):
        with open(ledger_path) as f:
            self.entries = json.load(f)
    
    def analyze_hypothesis(self, hypothesis_id: str) -> Dict:
        """Analyze validation status of hypothesis"""
        related = [e for e in self.entries if e.get('hypothesis_id') == hypothesis_id]
        
        if not related:
            return {'status': 'no_data'}
        
        # Calculate metrics
        confidences = [e['confidence'] for e in related if e.get('confidence')]
        authors = set(e['author'] for e in related)
        
        # Time span
        timestamps = [
            datetime.fromisoformat(e['timestamp_utc'].replace('Z', '+00:00'))
            for e in related
        ]
        time_span = (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0
        
        return {
            'status': 'validated' if len(related) >= 3 and len(authors) >= 2 else 'active',
            'entry_count': len(related),
            'author_count': len(authors),
            'authors': list(authors),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'time_span_days': time_span,
            'confidence_trend': 'increasing' if len(confidences) > 1 and confidences[-1] > confidences[0] else 'stable'
        }
    
    def generate_report(self) -> str:
        """Generate hypothesis validation report"""
        hypotheses = set(e.get('hypothesis_id') for e in self.entries if e.get('hypothesis_id'))
        
        lines = ["HYPOTHESIS VALIDATION REPORT", "=" * 70, ""]
        
        for hyp_id in sorted(hypotheses):
            analysis = self.analyze_hypothesis(hyp_id)
            
            status_icon = "✅" if analysis['status'] == 'validated' else "🔄"
            lines.append(f"{status_icon} {hyp_id}")
            lines.append(f"  Entries: {analysis['entry_count']}")
            lines.append(f"  Authors: {analysis['author_count']} ({', '.join(analysis['authors'])})")
            lines.append(f"  Avg Confidence: {analysis['avg_confidence']:.2f}")
            lines.append(f"  Max Confidence: {analysis['max_confidence']:.2f}")
            lines.append(f"  Trend: {analysis['confidence_trend']}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    ledger = Path(__file__).parent.parent / 'ledger' / 'index.json'
    validator = HypothesisValidator(ledger)
    print(validator.generate_report())


if __name__ == '__main__':
    main()
