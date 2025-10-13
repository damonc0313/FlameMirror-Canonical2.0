#!/usr/bin/env python3
"""
AUTONOMOUS LONG-RUN SESSION
Continuous self-improvement with quality validation

Based on Lumen Ledger patterns and autonomous operation protocols.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent / 'experiments' / 'lumen_ledger' / 'tools'))
from autonomous_self_improvement_integration import AutonomousSelfImprovement, AgentEnhancementTools


class LongRunSession:
    """
    Autonomous long-run session with checkpoints and quality validation.
    """
    
    def __init__(self, workspace: Path, max_cycles: int = 50):
        self.workspace = workspace
        self.max_cycles = max_cycles
        self.asi = AutonomousSelfImprovement(workspace)
        self.session_start = datetime.utcnow()
        self.checkpoints = []
        self.quality_history = []
        self.guardrail_triggers = []
        
    def log_checkpoint(self, cycle: int, metrics: Dict[str, Any]):
        """Log checkpoint with metrics."""
        checkpoint = {
            'cycle': cycle,
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed': (datetime.utcnow() - self.session_start).total_seconds(),
            'metrics': metrics
        }
        self.checkpoints.append(checkpoint)
        
        print(f"\n📍 CHECKPOINT {cycle}")
        print(f"   Tools: {metrics.get('total_tools', 0)}")
        print(f"   Quality: {metrics.get('avg_quality', 0):.3f}")
        print(f"   Acceleration: {metrics.get('cumulative_acceleration', 0):.2f}x")
        
        # Save checkpoint to disk
        checkpoint_file = self.workspace / f"autonomous_checkpoint_{cycle:03d}.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))
    
    def validate_code_quality(self, new_tools: List[str]) -> Dict[str, Any]:
        """
        Validate that newly generated code is genuinely better.
        
        Checks:
        - Complexity (is it solving harder problems?)
        - Abstractions (better design patterns?)
        - Error handling (more robust?)
        - Documentation (clearer?)
        """
        from production_exponential_enhancement import QualityValidator
        
        validator = QualityValidator(min_quality_score=0.80)
        
        quality_scores = []
        complexity_scores = []
        
        for tool_name in new_tools:
            tool_path = self.workspace / "autonomous_tools" / tool_name
            if tool_path.exists():
                code = tool_path.read_text()
                metrics = validator.validate(code)
                
                quality_scores.append(metrics.quality_score)
                complexity_scores.append(metrics.complexity_ratio)
        
        return {
            'tools_checked': len(quality_scores),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'avg_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'quality_improving': len(self.quality_history) > 0 and 
                               (sum(quality_scores) / len(quality_scores) if quality_scores else 0) > 
                               self.quality_history[-1]['avg_quality']
        }
    
    def check_guardrails(self) -> Dict[str, Any]:
        """
        Check system guardrails.
        
        Stops if:
        - Quality degrading consistently
        - Time limit reached
        - Too many cycles
        """
        elapsed = (datetime.utcnow() - self.session_start).total_seconds()
        
        guardrails = {
            'quality_ok': True,  # Check if last 3 cycles show improvement
            'time_ok': elapsed < 600,  # 10 minutes max
            'cycles_ok': len(self.checkpoints) < self.max_cycles
        }
        
        # Check quality trend
        if len(self.quality_history) >= 3:
            recent_qualities = [h['avg_quality'] for h in self.quality_history[-3:]]
            guardrails['quality_ok'] = recent_qualities[-1] >= recent_qualities[0]
        
        # Log any guardrail triggers
        for check, status in guardrails.items():
            if not status and check not in [g['check'] for g in self.guardrail_triggers]:
                trigger = {
                    'check': check,
                    'timestamp': datetime.utcnow().isoformat(),
                    'cycle': len(self.checkpoints)
                }
                self.guardrail_triggers.append(trigger)
                print(f"\n⚠️  GUARDRAIL TRIGGERED: {check}")
        
        return guardrails
    
    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        """Run one autonomous improvement cycle."""
        print(f"\n🔄 CYCLE {cycle}/{self.max_cycles}")
        
        # Check what capabilities are needed
        needed = self.asi.identify_needed_capabilities()
        print(f"   Capability gaps: {len(needed)}")
        
        # Generate tools
        count = min(5, len(needed)) if needed else 5
        result = self.asi.autonomous_enhancement_cycle(count=count)
        
        # Validate quality
        new_tools = [f"cache_manager_{i:03d}.py" for i in range(count)]  # Simplified
        quality_validation = self.validate_code_quality(new_tools)
        
        # Combine metrics
        metrics = {
            **result,
            **quality_validation
        }
        
        self.quality_history.append(quality_validation)
        
        return metrics
    
    def run(self):
        """Run autonomous long session."""
        print("="*70)
        print("🚀 AUTONOMOUS LONG-RUN SESSION")
        print("="*70)
        print(f"Max cycles: {self.max_cycles}")
        print(f"Started: {self.session_start.isoformat()}")
        print()
        
        cycle = 0
        
        try:
            while cycle < self.max_cycles:
                cycle += 1
                
                # Check guardrails
                guardrails = self.check_guardrails()
                if not all(guardrails.values()):
                    print("\n🛑 GUARDRAIL STOP")
                    for check, status in guardrails.items():
                        if not status:
                            print(f"   {check}: FAILED")
                    break
                
                # Run improvement cycle
                metrics = self.run_cycle(cycle)
                
                # Log checkpoint every 5 cycles
                if cycle % 5 == 0:
                    self.log_checkpoint(cycle, metrics)
                
                # Brief pause to avoid overwhelming system
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n⚠️  User interrupt")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.guardrail_triggers.append({
                'check': 'exception',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive session report."""
        elapsed = (datetime.utcnow() - self.session_start).total_seconds()
        
        print("\n" + "="*70)
        print("📊 FINAL REPORT")
        print("="*70)
        
        print(f"\nSession Duration: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Cycles Completed: {len(self.checkpoints)}")
        print(f"Guardrail Triggers: {len(self.guardrail_triggers)}")
        
        if self.quality_history:
            first_quality = self.quality_history[0]['avg_quality']
            last_quality = self.quality_history[-1]['avg_quality']
            print(f"\nQuality Trajectory:")
            print(f"  Start: {first_quality:.3f}")
            print(f"  End: {last_quality:.3f}")
            print(f"  Improvement: {'+' if last_quality > first_quality else ''}{last_quality - first_quality:.3f}")
        
        # Load final capabilities
        final_caps = json.loads((self.workspace / "autonomous_tools" / "capabilities.json").read_text())
        print(f"\nFinal State:")
        print(f"  Total tools: {final_caps['total_enhancements']}")
        print(f"  Cumulative acceleration: {final_caps['cumulative_acceleration']:.2f}x")
        
        # Save full report
        report = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.utcnow().isoformat(),
            'elapsed_seconds': elapsed,
            'cycles_completed': len(self.checkpoints),
            'checkpoints': self.checkpoints,
            'quality_history': self.quality_history,
            'guardrail_triggers': self.guardrail_triggers,
            'final_capabilities': final_caps
        }
        
        report_file = self.workspace / "autonomous_long_run_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        print(f"\n📄 Full report: {report_file}")
        
        print("\n✅ SESSION COMPLETE")


def main():
    """Launch autonomous long-run session."""
    workspace = Path("/workspace")
    
    # Longer session - 20 cycles max
    session = LongRunSession(workspace, max_cycles=20)
    session.run()


if __name__ == '__main__':
    main()
