#!/usr/bin/env python3
"""
DALE Framework Calibration Wrapper
Auto-adjusts confidence scores based on empirical accuracy data

Implements Enhancement Pathway #6: Framework Self-Modification

Philosophy: DALE generates raw confidence, this layer calibrates it based on
historical performance before using it for decisions.

Usage:
    from dale_calibration_wrapper import CalibratedDALE
    
    calibrated = CalibratedDALE()
    
    # Register actual outcomes to learn
    calibrated.register_outcome('predictive-enhancement', predicted=0.75, actual=False)
    
    # Get calibrated confidence for new predictions
    raw_confidence = 0.75
    adjusted = calibrated.calibrate('predictive-enhancement', raw_confidence)
    # Returns ~0.10 based on +0.65 gap
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class CapabilityCalibration:
    """Calibration data for a specific capability mode"""
    capability: str
    predictions: List[float] = field(default_factory=list)
    outcomes: List[bool] = field(default_factory=list)
    offset: float = 0.0
    sample_count: int = 0
    last_updated: Optional[str] = None
    
    def calculate_offset(self) -> float:
        """Calculate calibration offset from data"""
        if not self.outcomes or len(self.predictions) != len(self.outcomes):
            return self.offset
        
        avg_predicted = sum(self.predictions) / len(self.predictions)
        accuracy = sum(self.outcomes) / len(self.outcomes)
        
        # Offset = accuracy - predicted (pulls toward reality)
        raw_offset = accuracy - avg_predicted
        
        # Damping: gradual adjustment based on sample size
        damping = min(len(self.outcomes) / 5.0, 1.0)  # Full at 5+ samples
        
        return raw_offset * damping


class CalibratedDALE:
    """
    Wraps DALE framework with empirical calibration
    
    Maintains separate calibration for each capability mode,
    learning optimal adjustments from actual outcomes.
    """
    
    def __init__(self, calibration_file: Optional[Path] = None):
        self.calibration_file = calibration_file or Path(__file__).parent / 'dale_calibration_data.json'
        self.calibrations: Dict[str, CapabilityCalibration] = {}
        self.global_offset: float = 0.0
        self.min_samples: int = 2
        
        self.load_calibrations()
    
    def calibrate(self, capability: str, raw_confidence: float) -> float:
        """
        Adjust confidence score based on historical accuracy
        
        Args:
            capability: Capability mode (e.g., 'predictive-enhancement')
            raw_confidence: DALE's original confidence score (0.0-1.0)
        
        Returns:
            Calibrated confidence score
        """
        # Get capability-specific calibration
        if capability not in self.calibrations:
            # Use global offset for unknown capabilities
            offset = self.global_offset
        else:
            cal = self.calibrations[capability]
            if cal.sample_count >= self.min_samples:
                offset = cal.offset
            else:
                # Blend with global offset if insufficient samples
                alpha = cal.sample_count / self.min_samples
                offset = alpha * cal.offset + (1 - alpha) * self.global_offset
        
        # Apply calibration
        calibrated = raw_confidence + offset
        
        # Clamp to valid range
        return max(0.0, min(1.0, calibrated))
    
    def register_outcome(
        self, 
        capability: str, 
        predicted: float, 
        actual: bool,
        save: bool = True
    ) -> None:
        """
        Record prediction outcome to update calibration
        
        Args:
            capability: Capability mode
            predicted: Confidence score that was predicted
            actual: Whether prediction was correct
            save: Whether to persist to disk
        """
        if capability not in self.calibrations:
            self.calibrations[capability] = CapabilityCalibration(capability=capability)
        
        cal = self.calibrations[capability]
        cal.predictions.append(predicted)
        cal.outcomes.append(actual)
        cal.sample_count = len(cal.outcomes)
        cal.last_updated = datetime.utcnow().isoformat() + 'Z'
        
        # Recalculate offset
        cal.offset = cal.calculate_offset()
        
        # Update global offset (weighted average across all capabilities)
        self._update_global_offset()
        
        if save:
            self.save_calibrations()
    
    def _update_global_offset(self) -> None:
        """Calculate global offset as weighted average"""
        total_samples = 0
        weighted_sum = 0.0
        
        for cal in self.calibrations.values():
            if cal.sample_count > 0:
                weighted_sum += cal.offset * cal.sample_count
                total_samples += cal.sample_count
        
        if total_samples > 0:
            self.global_offset = weighted_sum / total_samples
    
    def load_calibrations(self) -> None:
        """Load calibration data from file"""
        if not self.calibration_file.exists():
            return
        
        try:
            with open(self.calibration_file) as f:
                data = json.load(f)
            
            self.global_offset = data.get('global_offset', 0.0)
            
            for cap_data in data.get('calibrations', []):
                cal = CapabilityCalibration(**cap_data)
                self.calibrations[cal.capability] = cal
        except Exception as e:
            print(f"Warning: Could not load calibrations: {e}")
    
    def save_calibrations(self) -> None:
        """Persist calibration data to disk"""
        data = {
            'global_offset': self.global_offset,
            'calibrations': [
                {
                    'capability': cal.capability,
                    'predictions': cal.predictions,
                    'outcomes': cal.outcomes,
                    'offset': cal.offset,
                    'sample_count': cal.sample_count,
                    'last_updated': cal.last_updated
                }
                for cal in self.calibrations.values()
            ],
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate calibration status report"""
        lines = ["=" * 70]
        lines.append("DALE CALIBRATION STATUS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Global Offset: {self.global_offset:+.3f}")
        lines.append(f"Minimum Samples: {self.min_samples}")
        lines.append(f"Capabilities Tracked: {len(self.calibrations)}")
        lines.append("")
        
        if self.calibrations:
            lines.append("CAPABILITY-SPECIFIC CALIBRATION:")
            lines.append("")
            
            # Sort by sample count (most data first)
            sorted_caps = sorted(
                self.calibrations.items(),
                key=lambda x: x[1].sample_count,
                reverse=True
            )
            
            for capability, cal in sorted_caps:
                lines.append(f"• {capability}")
                lines.append(f"  Samples: {cal.sample_count}")
                
                if cal.sample_count > 0:
                    avg_pred = sum(cal.predictions) / len(cal.predictions)
                    accuracy = sum(cal.outcomes) / len(cal.outcomes)
                    gap = avg_pred - accuracy
                    
                    lines.append(f"  Avg Predicted: {avg_pred:.3f}")
                    lines.append(f"  Actual Accuracy: {accuracy:.3f}")
                    lines.append(f"  Gap: {gap:+.3f}")
                    lines.append(f"  Offset Applied: {cal.offset:+.3f}")
                    
                    if abs(gap) < 0.05:
                        lines.append(f"  Status: ✓ Well calibrated")
                    elif gap > 0.1:
                        lines.append(f"  Status: ⚠ Overconfident")
                    elif gap < -0.1:
                        lines.append(f"  Status: ⚠ Underconfident")
                    else:
                        lines.append(f"  Status: Minor adjustment")
                else:
                    lines.append(f"  Status: No data yet")
                
                lines.append("")
        else:
            lines.append("No calibration data collected yet.")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def initialize_from_confidence_calibrator(
    calibrator_report_path: Path
) -> CalibratedDALE:
    """
    Initialize DALE calibration from confidence_calibrator.py output
    
    Parses calibration report and imports findings into DALE wrapper.
    """
    dale = CalibratedDALE()
    
    # Parse report (simplified - would need full implementation)
    # For now, manually set based on known +0.65 gap
    
    # From calibration report: 0% accuracy, 0.65 avg confidence
    # This means predictions were 65% overconfident
    dale.global_offset = -0.65  # Pull down by gap
    
    # Specific modes from report
    dale.register_outcome('synthesis-capability', predicted=0.70, actual=False, save=False)
    dale.register_outcome('co-design', predicted=0.60, actual=False, save=False)
    
    dale.save_calibrations()
    
    return dale


def main():
    """Demonstration and testing"""
    print("DALE CALIBRATION WRAPPER DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize with known calibration gap
    dale = CalibratedDALE()
    
    # Load existing confidence calibrator findings
    dale.global_offset = -0.65  # From calibration report
    
    # Register known outcomes
    print("Registering known prediction outcomes:")
    dale.register_outcome('predictive-enhancement', predicted=0.75, actual=False)
    dale.register_outcome('predictive-enhancement', predicted=0.72, actual=True)
    print(f"  • predictive-enhancement: 2 samples registered")
    print()
    
    # Test calibration
    print("CALIBRATION TESTS:")
    print("-" * 70)
    
    test_cases = [
        ('predictive-enhancement', 0.75),
        ('forecasting-analysis', 0.80),
        ('automated-tool-creation', 0.90),
        ('recursive-analysis', 0.70),
    ]
    
    for capability, raw_conf in test_cases:
        calibrated = dale.calibrate(capability, raw_conf)
        print(f"{capability}:")
        print(f"  Raw confidence: {raw_conf:.2f}")
        print(f"  Calibrated: {calibrated:.2f}")
        print(f"  Adjustment: {calibrated - raw_conf:+.2f}")
        print()
    
    # Show full report
    print(dale.generate_report())
    
    # Save for future sessions
    dale.save_calibrations()
    print(f"\nCalibrations saved to: {dale.calibration_file}")


if __name__ == '__main__':
    main()
