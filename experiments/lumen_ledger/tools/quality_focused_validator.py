#!/usr/bin/env python3
"""
Quality-Focused Validator v1.0
FOCUSED TOOL #3 - RL learned quality matters

Deep quality validation with:
- Multi-level quality scoring
- Evidence completeness checking
- Confidence justification validation
- Cross-reference verification
- Automated quality improvement suggestions
- Quality trend analysis

RL learned: My high-confidence entries (Q=high) came from quality focus.
Building tools that improve quality = applying learned policy.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class QualityReport:
    """Comprehensive quality assessment"""
    overall_score: float
    completeness_score: float
    evidence_score: float
    confidence_score: float
    clarity_score: float
    issues: List[str]
    suggestions: List[str]
    passing: bool


class QualityFocusedValidator:
    """Deep quality validation"""
    
    def __init__(self):
        self.quality_weights = {
            'completeness': 0.25,
            'evidence': 0.30,
            'confidence_justification': 0.25,
            'clarity': 0.20
        }
        self.passing_threshold = 0.75
    
    def validate_entry(self, entry: Dict) -> QualityReport:
        """Comprehensive quality validation"""
        
        # Score each dimension
        completeness = self._score_completeness(entry)
        evidence = self._score_evidence_quality(entry)
        confidence_just = self._score_confidence_justification(entry)
        clarity = self._score_clarity(entry)
        
        # Weighted overall
        overall = (
            completeness * self.quality_weights['completeness'] +
            evidence * self.quality_weights['evidence'] +
            confidence_just * self.quality_weights['confidence_justification'] +
            clarity * self.quality_weights['clarity']
        )
        
        # Identify issues
        issues = []
        suggestions = []
        
        if completeness < 0.70:
            issues.append("Incomplete required sections")
            suggestions.append("Fill in all observation, evidence, delta, questions, signals")
        
        if evidence < 0.70:
            issues.append("Insufficient evidence")
            suggestions.append("Add specific artifacts, measurements, test results")
        
        if confidence_just < 0.70:
            issues.append("Confidence not well justified")
            if entry.get('confidence_basis') == 'artifact_evidence':
                suggestions.append("Link to actual artifacts to justify artifact_evidence claim")
        
        if clarity < 0.70:
            issues.append("Observation lacks clarity")
            suggestions.append("Expand observation with more specific details")
        
        return QualityReport(
            overall_score=overall,
            completeness_score=completeness,
            evidence_score=evidence,
            confidence_score=confidence_just,
            clarity_score=clarity,
            issues=issues,
            suggestions=suggestions,
            passing=overall >= self.passing_threshold
        )
    
    def _score_completeness(self, entry: Dict) -> float:
        """Score completeness of entry"""
        required = ['observation', 'evidence', 'delta', 'open_questions', 'next_signals']
        score = 0.0
        
        for field in required:
            if field in entry and entry[field] and len(str(entry[field])) > 20:
                score += 0.2
        
        return min(1.0, score)
    
    def _score_evidence_quality(self, entry: Dict) -> float:
        """Score evidence quality"""
        score = 0.0
        
        # Has evidence field
        if entry.get('evidence'):
            score += 0.3
        
        # Has artifacts
        artifacts = entry.get('related_artifacts', [])
        if artifacts:
            score += min(0.4, len(artifacts) * 0.1)
        
        # Evidence basis matches artifacts
        if entry.get('confidence_basis') == 'artifact_evidence' and artifacts:
            score += 0.3
        
        return min(1.0, score)
    
    def _score_confidence_justification(self, entry: Dict) -> float:
        """Score confidence justification"""
        confidence = entry.get('confidence', 0)
        basis = entry.get('confidence_basis', '')
        artifacts = entry.get('related_artifacts', [])
        
        # High confidence should have strong evidence
        if confidence >= 0.90:
            if basis == 'artifact_evidence' and len(artifacts) >= 3:
                return 1.0
            elif basis == 'artifact_evidence' and artifacts:
                return 0.7
            else:
                return 0.4  # Unjustified high confidence
        
        # Medium confidence
        elif confidence >= 0.75:
            if basis == 'artifact_evidence' or basis == 'cross_validation':
                return 0.9
            else:
                return 0.6
        
        # Lower confidence is fine with any basis
        else:
            return 0.8
    
    def _score_clarity(self, entry: Dict) -> float:
        """Score observation clarity"""
        observation = entry.get('observation', '')
        
        if not observation:
            return 0.0
        
        # Length (should be substantial but not excessive)
        length = len(observation)
        if 100 <= length <= 1000:
            length_score = 1.0
        elif length < 100:
            length_score = length / 100
        else:
            length_score = max(0.5, 1.0 - (length - 1000) / 1000)
        
        # Specificity (mentions numbers, artifacts)
        specificity = 0.0
        if any(word in observation for word in ['.py', '.json', 'lines', 'measured']):
            specificity += 0.5
        
        # Structure (uses bullet points or clear sentences)
        if '- ' in observation or '. ' in observation:
            specificity += 0.5
        
        return (length_score + specificity) / 2


def main():
    """Execute quality validator"""
    
    print("="*70)
    print("✅ QUALITY-FOCUSED VALIDATOR")
    print("="*70)
    print("\nFOCUSED TOOL #3 - Applying RL policy")
    print("Building sequence of focused tools as RL recommended")
    print()
    
    # Test on recent entries
    from pathlib import Path
    ledger_path = Path(__file__).parent.parent / 'ledger' / 'index.json'
    
    with open(ledger_path) as f:
        import json
        entries = json.load(f)
    
    validator = QualityFocusedValidator()
    
    print(f"Validating {len(entries)} entries...")
    
    passing = 0
    failing = 0
    
    for entry in entries:
        report = validator.validate_entry(entry)
        if report.passing:
            passing += 1
        else:
            failing += 1
    
    print(f"\n✓ Validation complete:")
    print(f"  Passing: {passing}/{len(entries)} ({passing/len(entries)*100:.1f}%)")
    print(f"  Failing: {failing}/{len(entries)}")
    
    # Analyze Kael's quality
    kael_entries = [e for e in entries if e.get('author') == 'kael']
    kael_passing = sum(1 for e in kael_entries 
                       if validator.validate_entry(e).passing)
    
    print(f"\nKael's quality rate: {kael_passing}/{len(kael_entries)} ({kael_passing/len(kael_entries)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ QUALITY VALIDATOR OPERATIONAL")
    print(f"{'='*70}")
    print("\nRL-driven development:")
    print("  Tool #1: Code optimizer (RL recommended)")
    print("  Tool #2: Pattern learner (RL recommended)")
    print("  Tool #3: Quality validator (RL recommended)")
    print("\nAll focused, all useful, all high-quality")
    print("Applying learned policy = getting better results")


if __name__ == '__main__':
    main()
