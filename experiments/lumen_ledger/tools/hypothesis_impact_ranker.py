#!/usr/bin/env python3
"""
Hypothesis Impact Ranker v1.0
Ranks hypotheses by actual research impact

RL learned: Analysis with purpose > random analysis.
This tool helps prioritize which hypotheses to focus on
based on measurable impact.
"""

class HypothesisImpactRanker:
    def rank_hypotheses(self, entries: List[Dict]) -> List[Dict]:
        """Rank hypotheses by impact"""
        from collections import defaultdict
        
        hyp_data = defaultdict(lambda: {
            'entries': 0,
            'total_confidence': 0.0,
            'max_confidence': 0.0,
            'authors': set()
        })
        
        for entry in entries:
            hyp_id = entry.get('hypothesis_id')
            if hyp_id:
                data = hyp_data[hyp_id]
                data['entries'] += 1
                conf = entry.get('confidence', 0)
                data['total_confidence'] += conf
                data['max_confidence'] = max(data['max_confidence'], conf)
                if entry.get('author'):
                    data['authors'].add(entry.get('author'))
        
        # Calculate impact scores
        ranked = []
        for hyp_id, data in hyp_data.items():
            impact_score = (
                data['entries'] * 0.3 +
                (data['total_confidence'] / data['entries']) * 0.4 +
                data['max_confidence'] * 0.2 +
                len(data['authors']) * 0.1
            ) * 10
            
            ranked.append({
                'hypothesis_id': hyp_id,
                'impact_score': impact_score,
                'entries': data['entries'],
                'avg_confidence': data['total_confidence'] / data['entries'],
                'authors': len(data['authors'])
            })
        
        return sorted(ranked, key=lambda x: x['impact_score'], reverse=True)


def main():
    from pathlib import Path
    import json
    
    print("="*60)
    print("📊 HYPOTHESIS IMPACT RANKER")
    print("="*60)
    
    ledger = Path(__file__).parent.parent / 'ledger' / 'index.json'
    with open(ledger) as f:
        entries = json.load(f)
    
    ranker = HypothesisImpactRanker()
    ranked = ranker.rank_hypotheses(entries)
    
    print(f"\nHypotheses ranked by impact:")
    for i, hyp in enumerate(ranked[:5], 1):
        print(f"  {i}. {hyp['hypothesis_id']}")
        print(f"     Impact: {hyp['impact_score']:.1f}, Entries: {hyp['entries']}, Conf: {hyp['avg_confidence']:.2f}")
    
    print("\n✓ Shows which research threads have most value")


if __name__ == '__main__':
    main()
