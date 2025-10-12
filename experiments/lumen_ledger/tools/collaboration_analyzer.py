#!/usr/bin/env python3
"""Collaboration Analyzer - Measures cross-author collaboration effectiveness"""
class CollaborationAnalyzer:
    def analyze_collaboration(self, entries):
        """Measure collaboration patterns"""
        # Count peer reviews
        reviews = sum(1 for e in entries if e.get('reviewed_by'))
        
        # Count cross-author hypothesis work
        from collections import defaultdict
        hyp_authors = defaultdict(set)
        for e in entries:
            if e.get('hypothesis_id') and e.get('author'):
                hyp_authors[e['hypothesis_id']].add(e['author'])
        
        cross_author_hyps = sum(1 for authors in hyp_authors.values() if len(authors) > 1)
        
        return {
            'total_reviews': reviews,
            'cross_author_hypotheses': cross_author_hyps,
            'collaboration_score': (reviews + cross_author_hyps * 2) / max(len(entries), 1)
        }

if __name__ == '__main__':
    analyzer = CollaborationAnalyzer()
    print("✓ Collaboration analyzer ready")
