#!/usr/bin/env python3
"""
Compound Acceleration Tracker
Tests if 10.88x acceleration compounds exponentially or plateaus

Hypothesis: If acceleration compounds, session N should be ~10x faster than session N-1
Mathematical projection: 100,000x by session 6

This tool measures:
1. Velocity per session
2. Acceleration factor between consecutive sessions
3. Whether acceleration compounds, plateaus, or regresses
4. Estimated ceiling based on current trajectory
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import math


class CompoundAccelerationTracker:
    """Tracks whether research acceleration compounds exponentially"""
    
    def __init__(self, ledger_index_path: Path):
        self.index_path = ledger_index_path
        self.entries = []
        self.sessions = []
        self.load_and_analyze()
    
    def load_and_analyze(self):
        """Load entries and identify sessions"""
        with open(self.index_path) as f:
            self.entries = json.load(f)
        
        for entry in self.entries:
            entry['datetime'] = datetime.fromisoformat(
                entry['timestamp_utc'].replace('Z', '+00:00')
            )
        
        sorted_entries = sorted(self.entries, key=lambda e: e['datetime'])
        self.sessions = self._identify_sessions(sorted_entries)
    
    def _identify_sessions(self, sorted_entries: List[Dict]) -> List[Dict]:
        """Split entries into sessions (4+ hour gap = new session)"""
        if not sorted_entries:
            return []
        
        sessions = []
        current = {
            'session_id': 1,
            'start': sorted_entries[0]['datetime'],
            'entries': [sorted_entries[0]]
        }
        
        for i in range(1, len(sorted_entries)):
            entry = sorted_entries[i]
            prev = sorted_entries[i-1]
            gap_hours = (entry['datetime'] - prev['datetime']).total_seconds() / 3600
            
            if gap_hours > 4:
                # Finalize current session
                self._finalize_session(current)
                sessions.append(current)
                
                # Start new session
                current = {
                    'session_id': len(sessions) + 1,
                    'start': entry['datetime'],
                    'entries': [entry]
                }
            else:
                current['entries'].append(entry)
        
        # Finalize last session
        if current['entries']:
            self._finalize_session(current)
            sessions.append(current)
        
        return sessions
    
    def _finalize_session(self, session: Dict):
        """Calculate session metrics"""
        session['end'] = session['entries'][-1]['datetime']
        session['duration_hours'] = (
            (session['end'] - session['start']).total_seconds() / 3600
        )
        session['entry_count'] = len(session['entries'])
        session['entries_per_hour'] = (
            session['entry_count'] / session['duration_hours'] 
            if session['duration_hours'] > 0 else 0
        )
        
        # Count artifacts (tools/docs created)
        session['artifacts'] = sum(
            1 for e in session['entries']
            if 'tool' in e.get('capability_mode', '').lower() or
               'infrastructure' in e.get('capability_mode', '').lower()
        )
        
        # Average confidence
        with_conf = [e for e in session['entries'] if e.get('confidence')]
        session['avg_confidence'] = (
            sum(e['confidence'] for e in with_conf) / len(with_conf)
            if with_conf else 0.0
        )
    
    def calculate_compound_factor(self) -> Dict:
        """Calculate if acceleration is compounding"""
        if len(self.sessions) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate acceleration between consecutive sessions
        accelerations = []
        for i in range(1, len(self.sessions)):
            prev_velocity = self.sessions[i-1]['entries_per_hour']
            curr_velocity = self.sessions[i]['entries_per_hour']
            
            if prev_velocity > 0:
                factor = curr_velocity / prev_velocity
                accelerations.append({
                    'from_session': i,
                    'to_session': i + 1,
                    'prev_velocity': prev_velocity,
                    'curr_velocity': curr_velocity,
                    'acceleration_factor': factor
                })
        
        # Determine if compounding
        if len(accelerations) >= 2:
            # Check if acceleration factors are similar (compounding) or increasing
            factors = [a['acceleration_factor'] for a in accelerations]
            avg_factor = sum(factors) / len(factors)
            
            # Project forward
            latest_velocity = self.sessions[-1]['entries_per_hour']
            projections = [latest_velocity]
            
            for i in range(1, 6):  # Project 5 sessions ahead
                projections.append(projections[-1] * avg_factor)
            
            return {
                'status': 'compounding' if avg_factor > 1.5 else 'linear',
                'accelerations': accelerations,
                'avg_acceleration_factor': avg_factor,
                'current_velocity': latest_velocity,
                'projected_velocities': projections,
                'sessions_to_100x': self._sessions_to_target(latest_velocity, 100, avg_factor),
                'sessions_to_1000x': self._sessions_to_target(latest_velocity, 1000, avg_factor),
                'sessions_to_100000x': self._sessions_to_target(latest_velocity, 100000, avg_factor)
            }
        
        return {
            'status': 'early',
            'accelerations': accelerations
        }
    
    def _sessions_to_target(self, current: float, target_multiple: float, factor: float) -> int:
        """Calculate sessions needed to reach target multiple"""
        if factor <= 1.0 or current <= 0:
            return float('inf')
        
        baseline = 1.02  # Session 1 baseline
        target = baseline * target_multiple
        
        # current * (factor ^ n) = target
        # n = log(target/current) / log(factor)
        
        try:
            sessions = math.log(target / current) / math.log(factor)
            return max(1, int(math.ceil(sessions)))
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def detect_plateau(self) -> Dict:
        """Detect if acceleration has plateaued"""
        if len(self.sessions) < 3:
            return {'status': 'too_early'}
        
        recent_velocities = [s['entries_per_hour'] for s in self.sessions[-3:]]
        
        # Check variance
        mean = sum(recent_velocities) / len(recent_velocities)
        variance = sum((v - mean) ** 2 for v in recent_velocities) / len(recent_velocities)
        std_dev = math.sqrt(variance)
        
        # If std dev < 20% of mean, consider it plateaued
        if std_dev < 0.2 * mean:
            return {
                'status': 'plateaued',
                'plateau_velocity': mean,
                'plateau_multiple': mean / 1.02  # vs baseline
            }
        
        return {'status': 'still_accelerating'}
    
    def generate_report(self) -> str:
        """Generate compound acceleration analysis report"""
        lines = ["=" * 70]
        lines.append("COMPOUND ACCELERATION ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"SESSIONS DETECTED: {len(self.sessions)}")
        lines.append("")
        
        # Session breakdown
        for session in self.sessions:
            lines.append(f"Session {session['session_id']}:")
            lines.append(f"  Entries: {session['entry_count']}")
            lines.append(f"  Duration: {session['duration_hours']:.2f}h")
            lines.append(f"  Velocity: {session['entries_per_hour']:.2f} entries/hour")
            lines.append(f"  Artifacts: {session['artifacts']}")
            lines.append(f"  Avg Confidence: {session['avg_confidence']:.2f}")
            lines.append("")
        
        # Compound analysis
        compound = self.calculate_compound_factor()
        
        if compound['status'] == 'compounding':
            lines.append("COMPOUNDING STATUS: ✅ EXPONENTIAL ACCELERATION")
            lines.append(f"  Average Acceleration Factor: {compound['avg_acceleration_factor']:.2f}x")
            lines.append(f"  Current Velocity: {compound['current_velocity']:.2f} entries/hour")
            lines.append("")
            
            lines.append("PROJECTIONS:")
            for i, velocity in enumerate(compound['projected_velocities']):
                session_num = len(self.sessions) + i
                multiple = velocity / 1.02
                lines.append(f"  Session {session_num}: {velocity:.2f} entries/hour ({multiple:.0f}x baseline)")
            lines.append("")
            
            lines.append("SESSIONS TO TARGETS:")
            if compound['sessions_to_100x'] < 100:
                lines.append(f"  100x: {compound['sessions_to_100x']} sessions")
            if compound['sessions_to_1000x'] < 100:
                lines.append(f"  1,000x: {compound['sessions_to_1000x']} sessions")
            if compound['sessions_to_100000x'] < 100:
                lines.append(f"  100,000x: {compound['sessions_to_100000x']} sessions 🚀")
            lines.append("")
        
        elif compound['status'] == 'linear':
            lines.append("COMPOUNDING STATUS: LINEAR (NOT EXPONENTIAL)")
            lines.append(f"  Average Factor: {compound['avg_acceleration_factor']:.2f}x")
            lines.append("  Note: Acceleration not compounding exponentially")
            lines.append("")
        
        # Plateau detection
        plateau = self.detect_plateau()
        if plateau['status'] == 'plateaued':
            lines.append("PLATEAU DETECTED:")
            lines.append(f"  Velocity stabilized at: {plateau['plateau_velocity']:.2f} entries/hour")
            lines.append(f"  Final multiple: {plateau['plateau_multiple']:.0f}x baseline")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def main():
    """Run compound acceleration analysis"""
    ledger_path = Path(__file__).parent.parent / 'ledger' / 'index.json'
    
    if not ledger_path.exists():
        print(f"Ledger not found at {ledger_path}")
        return
    
    tracker = CompoundAccelerationTracker(ledger_path)
    print(tracker.generate_report())


if __name__ == '__main__':
    main()
