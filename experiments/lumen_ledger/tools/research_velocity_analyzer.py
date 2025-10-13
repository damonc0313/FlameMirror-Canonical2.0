#!/usr/bin/env python3
"""
Research Velocity Analyzer
Measures research output rate to test autonomous operation hypothesis

Hypothesis: Full autonomy enables exponentially faster research cycles
Metrics:
- Entries per hour
- Tools built per session  
- Confidence trajectory
- Capability mode diversity
- Time to validation
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class ResearchVelocityAnalyzer:
    """Analyzes research productivity metrics"""
    
    def __init__(self, ledger_index_path: Path):
        self.index_path = ledger_index_path
        self.entries = []
        self.load_entries()
    
    def load_entries(self):
        """Load ledger entries"""
        with open(self.index_path) as f:
            self.entries = json.load(f)
        
        # Parse timestamps
        for entry in self.entries:
            entry['datetime'] = datetime.fromisoformat(
                entry['timestamp_utc'].replace('Z', '+00:00')
            )
    
    def calculate_velocity_metrics(self) -> Dict:
        """Calculate comprehensive velocity metrics"""
        if not self.entries:
            return {}
        
        # Sort by time
        sorted_entries = sorted(self.entries, key=lambda e: e['datetime'])
        
        first = sorted_entries[0]['datetime']
        last = sorted_entries[-1]['datetime']
        total_duration = (last - first).total_seconds() / 3600  # hours
        
        metrics = {
            'total_entries': len(self.entries),
            'total_duration_hours': total_duration,
            'entries_per_hour': len(self.entries) / total_duration if total_duration > 0 else 0,
            'unique_authors': len(set(e['author'] for e in self.entries)),
            'unique_capabilities': len(set(e.get('capability_mode', '') for e in self.entries)),
            'unique_hypotheses': len(set(e.get('hypothesis_id', '') for e in self.entries if e.get('hypothesis_id'))),
        }
        
        # Confidence trends
        entries_with_conf = [e for e in self.entries if e.get('confidence')]
        if entries_with_conf:
            metrics['avg_confidence'] = sum(e['confidence'] for e in entries_with_conf) / len(entries_with_conf)
            metrics['confidence_trend'] = self._calculate_trend(
                [e['confidence'] for e in sorted(entries_with_conf, key=lambda e: e['datetime'])]
            )
        
        # Session analysis
        sessions = self._identify_sessions(sorted_entries)
        metrics['total_sessions'] = len(sessions)
        metrics['avg_entries_per_session'] = len(self.entries) / len(sessions) if sessions else 0
        
        # Acceleration detection
        if len(sessions) >= 2:
            early_rate = sessions[0]['entries_per_hour']
            late_rate = sessions[-1]['entries_per_hour']
            metrics['acceleration_factor'] = late_rate / early_rate if early_rate > 0 else 1.0
        
        return metrics
    
    def _identify_sessions(self, sorted_entries: List[Dict]) -> List[Dict]:
        """Identify research sessions (gap > 4 hours = new session)"""
        if not sorted_entries:
            return []
        
        sessions = []
        current_session = {
            'start': sorted_entries[0]['datetime'],
            'entries': [sorted_entries[0]]
        }
        
        for i in range(1, len(sorted_entries)):
            entry = sorted_entries[i]
            prev_entry = sorted_entries[i-1]
            
            gap = (entry['datetime'] - prev_entry['datetime']).total_seconds() / 3600
            
            if gap > 4:  # New session
                # Complete current session
                current_session['end'] = current_session['entries'][-1]['datetime']
                duration = (current_session['end'] - current_session['start']).total_seconds() / 3600
                current_session['duration_hours'] = duration
                current_session['entry_count'] = len(current_session['entries'])
                current_session['entries_per_hour'] = current_session['entry_count'] / duration if duration > 0 else 0
                sessions.append(current_session)
                
                # Start new session
                current_session = {
                    'start': entry['datetime'],
                    'entries': [entry]
                }
            else:
                current_session['entries'].append(entry)
        
        # Complete final session
        if current_session['entries']:
            current_session['end'] = current_session['entries'][-1]['datetime']
            duration = (current_session['end'] - current_session['start']).total_seconds() / 3600
            current_session['duration_hours'] = duration
            current_session['entry_count'] = len(current_session['entries'])
            current_session['entries_per_hour'] = current_session['entry_count'] / duration if duration > 0 else 0
            sessions.append(current_session)
        
        return sessions
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if trend is increasing, decreasing, or stable"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        diff = avg_second - avg_first
        
        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def analyze_autonomous_operation(self) -> Dict:
        """
        Specifically analyze if autonomous operation improved velocity
        
        Compares sessions with explicit autonomy vs without
        """
        autonomous_entries = []
        non_autonomous_entries = []
        
        for entry in self.entries:
            obs = entry.get('observation', '').lower()
            if any(word in obs for word in ['autonomous', 'full autonomy', 'without stopping', 'continuous']):
                autonomous_entries.append(entry)
            else:
                non_autonomous_entries.append(entry)
        
        result = {
            'autonomous_count': len(autonomous_entries),
            'non_autonomous_count': len(non_autonomous_entries)
        }
        
        if autonomous_entries:
            result['autonomous_avg_confidence'] = sum(
                e['confidence'] for e in autonomous_entries if e.get('confidence')
            ) / len([e for e in autonomous_entries if e.get('confidence')]) if any(e.get('confidence') for e in autonomous_entries) else None
        
        return result
    
    def generate_report(self) -> str:
        """Generate comprehensive velocity report"""
        metrics = self.calculate_velocity_metrics()
        autonomous_analysis = self.analyze_autonomous_operation()
        
        lines = ["=" * 70]
        lines.append("RESEARCH VELOCITY ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("OVERALL METRICS:")
        lines.append(f"  Total Entries: {metrics.get('total_entries', 0)}")
        lines.append(f"  Duration: {metrics.get('total_duration_hours', 0):.1f} hours")
        lines.append(f"  Velocity: {metrics.get('entries_per_hour', 0):.2f} entries/hour")
        lines.append(f"  Authors: {metrics.get('unique_authors', 0)}")
        lines.append(f"  Capability Modes: {metrics.get('unique_capabilities', 0)}")
        lines.append(f"  Hypotheses: {metrics.get('unique_hypotheses', 0)}")
        lines.append("")
        
        if metrics.get('avg_confidence'):
            lines.append("CONFIDENCE METRICS:")
            lines.append(f"  Average: {metrics['avg_confidence']:.2f}")
            lines.append(f"  Trend: {metrics.get('confidence_trend', 'unknown')}")
            lines.append("")
        
        lines.append("SESSION ANALYSIS:")
        lines.append(f"  Total Sessions: {metrics.get('total_sessions', 0)}")
        lines.append(f"  Avg Entries/Session: {metrics.get('avg_entries_per_session', 0):.1f}")
        
        if metrics.get('acceleration_factor'):
            factor = metrics['acceleration_factor']
            lines.append(f"  Acceleration Factor: {factor:.2f}x")
            
            if factor > 1.5:
                lines.append(f"  Status: ✓ EXPONENTIAL ACCELERATION DETECTED")
            elif factor > 1.1:
                lines.append(f"  Status: ✓ Positive acceleration")
            else:
                lines.append(f"  Status: Stable velocity")
        
        lines.append("")
        
        lines.append("AUTONOMOUS OPERATION ANALYSIS:")
        lines.append(f"  Autonomous Entries: {autonomous_analysis['autonomous_count']}")
        lines.append(f"  Non-Autonomous: {autonomous_analysis['non_autonomous_count']}")
        
        if autonomous_analysis.get('autonomous_avg_confidence'):
            lines.append(f"  Autonomous Avg Confidence: {autonomous_analysis['autonomous_avg_confidence']:.2f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def main():
    """Run velocity analysis on current ledger"""
    ledger_path = Path(__file__).parent.parent / 'ledger' / 'index.json'
    
    if not ledger_path.exists():
        print(f"Ledger not found at {ledger_path}")
        return
    
    analyzer = ResearchVelocityAnalyzer(ledger_path)
    print(analyzer.generate_report())
    
    # Also show session breakdown
    print("\nSESSION BREAKDOWN:")
    print("-" * 70)
    sessions = analyzer._identify_sessions(
        sorted(analyzer.entries, key=lambda e: e['datetime'])
    )
    for i, session in enumerate(sessions, 1):
        print(f"Session {i}: {session['entry_count']} entries in {session['duration_hours']:.1f}h "
              f"({session['entries_per_hour']:.2f}/h)")


if __name__ == '__main__':
    main()
