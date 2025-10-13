#!/usr/bin/env python3
"""
Hypothesis Evolution Engine v1.0
Auto-generates, tests, and validates hypotheses from ledger patterns.
Enables exponential hypothesis exploration through recursive spawning.
"""

import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from statistics import mean, stdev


@dataclass
class Pattern:
    """Detected pattern in ledger data"""
    pattern_type: str  # cluster, progression, correlation, emergence
    description: str
    evidence_count: int
    confidence: float
    related_entries: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def strength(self) -> float:
        """Calculate pattern strength from evidence and confidence"""
        return (self.confidence * 0.6) + (min(self.evidence_count / 5, 1.0) * 0.4)


@dataclass
class Hypothesis:
    """Research hypothesis with validation tracking"""
    hypothesis_id: str
    title: str
    description: str
    proposed_at: str
    proposed_by: str
    status: str  # proposed, testing, validated, invalidated
    confidence: float
    supporting_patterns: List[Pattern]
    validation_criteria: List[str]
    evidence_entries: List[str] = field(default_factory=list)
    invalidation_entries: List[str] = field(default_factory=list)
    sub_hypotheses: List[str] = field(default_factory=list)
    parent_hypothesis: Optional[str] = None
    
    def validation_score(self) -> float:
        """Calculate how validated this hypothesis is"""
        if not self.validation_criteria:
            return 0.0
        
        evidence_ratio = len(self.evidence_entries) / max(1, len(self.evidence_entries) + len(self.invalidation_entries))
        
        # Check criteria against evidence entries
        evidence_str = ' '.join(self.evidence_entries).lower()
        criteria_met = sum(1 for c in self.validation_criteria if c.lower() in evidence_str)
        criteria_ratio = criteria_met / len(self.validation_criteria)
        
        return (evidence_ratio * 0.6) + (criteria_ratio * 0.4)
    
    def should_spawn_sub_hypothesis(self) -> bool:
        """Determine if this hypothesis should spawn sub-hypotheses"""
        return (
            self.status == 'testing' and
            len(self.evidence_entries) >= 3 and
            self.validation_score() > 0.6 and
            len(self.sub_hypotheses) < 5
        )


class HypothesisEvolutionEngine:
    """
    Auto-generates and validates hypotheses from ledger patterns.
    Implements recursive hypothesis exploration.
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.patterns: List[Pattern] = []
        self.ledger_entries: List[Dict] = []
        
    def load_ledger_data(self, entries: List[Dict]):
        """Load ledger entries for analysis"""
        self.ledger_entries = entries
        print(f"Loaded {len(entries)} ledger entries")
    
    def detect_patterns(self) -> List[Pattern]:
        """Detect all pattern types"""
        patterns = []
        patterns.extend(self._detect_capability_clusters())
        patterns.extend(self._detect_confidence_progressions())
        patterns.extend(self._detect_cross_author_correlations())
        self.patterns = patterns
        print(f"Detected {len(patterns)} patterns")
        return patterns
    
    def _detect_capability_clusters(self) -> List[Pattern]:
        """Find groups of related capability modes"""
        clusters = []
        capability_groups = defaultdict(list)
        
        for entry in self.ledger_entries:
            if not entry.get('capability_mode'):
                continue
            mode = entry['capability_mode']
            base_concepts = set(word for word in mode.split('-') if len(word) > 4)
            
            for concept in base_concepts:
                capability_groups[concept].append(entry)
        
        for concept, entries in capability_groups.items():
            if len(entries) >= 2:
                confidences = [e['confidence'] for e in entries if e.get('confidence')]
                if confidences:
                    pattern = Pattern(
                        pattern_type='cluster',
                        description=f"Capability cluster around '{concept}' concept",
                        evidence_count=len(entries),
                        confidence=mean(confidences),
                        related_entries=[e['id'] for e in entries],
                        metrics={'avg_confidence': mean(confidences)}
                    )
                    clusters.append(pattern)
        
        return clusters
    
    def _detect_confidence_progressions(self) -> List[Pattern]:
        """Detect confidence increasing over time"""
        progressions = []
        author_data = defaultdict(list)
        
        for entry in self.ledger_entries:
            if entry.get('confidence') and entry.get('author'):
                author_data[entry['author']].append({
                    'timestamp': entry['timestamp_utc'],
                    'confidence': entry['confidence'],
                    'id': entry['id']
                })
        
        for author, entries_list in author_data.items():
            if len(entries_list) >= 3:
                sorted_entries = sorted(entries_list, key=lambda x: x['timestamp'])
                confidences = [e['confidence'] for e in sorted_entries]
                
                early_avg = mean(confidences[:len(confidences)//2])
                late_avg = mean(confidences[len(confidences)//2:])
                
                if late_avg > early_avg + 0.05:
                    pattern = Pattern(
                        pattern_type='progression',
                        description=f"Confidence progression for {author}",
                        evidence_count=len(sorted_entries),
                        confidence=late_avg,
                        related_entries=[e['id'] for e in sorted_entries],
                        metrics={'improvement': late_avg - early_avg}
                    )
                    progressions.append(pattern)
        
        return progressions
    
    def _detect_cross_author_correlations(self) -> List[Pattern]:
        """Find similar capabilities by different authors"""
        correlations = []
        capability_authors = defaultdict(set)
        capability_entries = defaultdict(list)
        
        for entry in self.ledger_entries:
            if entry.get('capability_mode') and entry.get('author'):
                mode = entry['capability_mode']
                capability_authors[mode].add(entry['author'])
                capability_entries[mode].append(entry)
        
        for mode, authors in capability_authors.items():
            if len(authors) >= 2:
                entries = capability_entries[mode]
                confidences = [e['confidence'] for e in entries if e.get('confidence')]
                
                if confidences:
                    pattern = Pattern(
                        pattern_type='correlation',
                        description=f"Cross-author validation: {mode}",
                        evidence_count=len(entries),
                        confidence=mean(confidences),
                        related_entries=[e['id'] for e in entries],
                        metrics={'num_authors': len(authors)}
                    )
                    correlations.append(pattern)
        
        return correlations
    
    def generate_hypotheses(self) -> List[Hypothesis]:
        """Generate new hypotheses from detected patterns"""
        new_hypotheses = []
        
        for pattern in self.patterns:
            if pattern.strength() >= 0.6:
                hypothesis = self._pattern_to_hypothesis(pattern)
                if hypothesis and hypothesis.hypothesis_id not in self.hypotheses:
                    new_hypotheses.append(hypothesis)
                    self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        print(f"Generated {len(new_hypotheses)} new hypotheses")
        return new_hypotheses
    
    def _pattern_to_hypothesis(self, pattern: Pattern) -> Optional[Hypothesis]:
        """Convert pattern to testable hypothesis"""
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        if pattern.pattern_type == 'cluster':
            match = re.search(r"'(\w+)'", pattern.description)
            concept = match.group(1) if match else "unknown"
            
            return Hypothesis(
                hypothesis_id=f"{concept}-clustering",
                title=f"{concept.capitalize()} capabilities cluster together",
                description=f"Capabilities with '{concept}' emerge as interconnected cluster",
                proposed_at=timestamp,
                proposed_by="hypothesis-engine",
                status='testing',
                confidence=pattern.confidence,
                supporting_patterns=[pattern],
                validation_criteria=[
                    f"At least 3 {concept} capabilities",
                    "Average confidence >= 0.80"
                ],
                evidence_entries=pattern.related_entries
            )
        
        return None
    
    def export_hypotheses(self, filepath: str):
        """Export hypotheses to JSON"""
        export_data = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_hypotheses': len(self.hypotheses),
            'hypotheses': {
                hid: {**asdict(h), 'validation_score': h.validation_score()}
                for hid, h in self.hypotheses.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Exported {len(self.hypotheses)} hypotheses")


if __name__ == '__main__':
    from pathlib import Path
    ledger_path = Path(__file__).parent.parent / 'ledger' / 'index.json'
    
    with open(ledger_path) as f:
        entries = json.load(f)
    
    engine = HypothesisEvolutionEngine()
    engine.load_ledger_data(entries)
    engine.detect_patterns()
    engine.generate_hypotheses()
    engine.export_hypotheses('/tmp/hypotheses.json')
    
    print(f"\n✓ Generated {len(engine.hypotheses)} hypotheses from {len(engine.patterns)} patterns")
