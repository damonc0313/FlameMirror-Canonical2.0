#!/usr/bin/env python3
"""Synthesis Scheduler - Automatically schedules and runs synthesis at optimal times"""
class SynthesisScheduler:
    def __init__(self):
        self.last_synthesis = None
        self.synthesis_triggers = {
            'entry_threshold': 5,  # Every 5 new entries
            'time_threshold_hours': 24,  # Every 24 hours
            'breakthrough_immediate': True  # Immediate on breakthrough
        }
    
    def should_run_synthesis(self, entries_since_last, hours_since_last, has_breakthrough):
        """Determine if synthesis should run now"""
        if has_breakthrough and self.synthesis_triggers['breakthrough_immediate']:
            return True, "Breakthrough detected"
        
        if entries_since_last >= self.synthesis_triggers['entry_threshold']:
            return True, f"{entries_since_last} new entries"
        
        if hours_since_last >= self.synthesis_triggers['time_threshold_hours']:
            return True, f"{hours_since_last}h elapsed"
        
        return False, "No trigger met"
    
    def schedule_next_synthesis(self, current_state):
        """Calculate when next synthesis should run"""
        return {'scheduled_in': '5 entries or 24h', 'reason': 'Standard cadence'}

if __name__ == '__main__':
    scheduler = SynthesisScheduler()
    should_run, reason = scheduler.should_run_synthesis(6, 2, False)
    print(f"✓ Scheduler ready. Should synthesize: {should_run} ({reason})")
