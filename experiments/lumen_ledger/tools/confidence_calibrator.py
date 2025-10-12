#!/usr/bin/env python3
"""Confidence Calibration Tracker for Lumen Ledger

Tracks prediction accuracy across ledger entries to measure and improve
forecasting confidence calibration. Part of Enhancement Pathway #5.

This tool:
1. Extracts predictions from ledger entries
2. Validates predictions against subsequent entries
3. Calculates calibration gaps (predicted confidence vs actual accuracy)
4. Generates calibration reports
5. Suggests confidence adjustments for future predictions

Usage:
    python confidence_calibrator.py analyze
    python confidence_calibrator.py report --output calibration-report.md
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class Prediction:
    """A prediction made in a ledger entry."""
    entry_id: str
    timestamp: datetime
    predicted_capability: str
    predicted_confidence: float
    validation_window: str  # e.g., "next 1-3 sessions"
    falsifiable_condition: str
    validated: Optional[bool] = None
    actual_outcome: Optional[str] = None
    validation_entry_id: Optional[str] = None
    calibration_error: Optional[float] = None


@dataclass
class CalibrationStats:
    """Statistics for confidence calibration."""
    total_predictions: int = 0
    validated_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    pending_predictions: int = 0
    
    mean_predicted_confidence: float = 0.0
    actual_accuracy: float = 0.0
    calibration_gap: float = 0.0  # predicted - actual
    
    by_capability_mode: Dict[str, Dict] = field(default_factory=dict)
    by_author: Dict[str, Dict] = field(default_factory=dict)


class ConfidenceCalibrator:
    """Tracks and analyzes prediction accuracy across ledger entries."""
    
    def __init__(self, ledger_root: Path):
        self.ledger_root = ledger_root
        self.index_path = ledger_root / "index.json"
        self.entries = []
        self.predictions = []
        
    def load_entries(self):
        """Load all ledger entries from index."""
        with open(self.index_path) as f:
            self.entries = json.load(f)
        print(f"Loaded {len(self.entries)} entries")
        
    def extract_predictions(self) -> List[Prediction]:
        """Extract predictions from entries containing forecasts or predictions."""
        predictions = []
        
        for entry in self.entries:
            entry_id = entry['id']
            timestamp = datetime.fromisoformat(entry['timestamp_utc'].replace('Z', '+00:00'))
            capability_mode = entry.get('capability_mode', '')
            
            # Look for predictive capability modes
            if any(mode in capability_mode for mode in ['forecast', 'predict', 'calibrat']):
                # Read full entry to extract predictions
                entry_path = Path(entry['path'])
                if not entry_path.exists():
                    continue
                    
                with open(entry_path) as f:
                    content = f.read()
                
                # Extract predictions from Next Signals section
                predictions.extend(self._extract_from_next_signals(
                    content, entry_id, timestamp, capability_mode
                ))
                
                # Extract from Open Questions if they contain predictions
                predictions.extend(self._extract_from_open_questions(
                    content, entry_id, timestamp, capability_mode
                ))
        
        self.predictions = predictions
        print(f"Extracted {len(predictions)} predictions")
        return predictions
    
    def _extract_from_next_signals(
        self, content: str, entry_id: str, timestamp: datetime, capability_mode: str
    ) -> List[Prediction]:
        """Extract predictions from Next Signals section."""
        predictions = []
        
        # Find Next Signals section
        match = re.search(r'## Next Signals\s*\n(.*?)(?:\n##|$)', content, re.DOTALL)
        if not match:
            return predictions
        
        signals = match.group(1)
        
        # Look for predictive language
        # Pattern: "Predict X will Y" or "Expected: X within Y"
        predict_patterns = [
            r'(?:Predict|Expect|Forecast|Anticipate)[:\s]+(.*?)(?:\n|$)',
            r'Within\s+(\d+-?\d*\s+\w+),\s+(.*?)(?:\n|$)',
            r'Next\s+(\d+-?\d*)\s+(\w+).*?(will|should|likely)\s+(.*?)(?:\n|$)',
        ]
        
        for pattern in predict_patterns:
            for match in re.finditer(pattern, signals, re.IGNORECASE):
                # Extract prediction components
                prediction_text = match.group(0)
                
                # Try to extract capability and confidence
                capability = self._extract_capability(prediction_text)
                confidence = self._extract_confidence(prediction_text)
                window = self._extract_window(prediction_text)
                
                if capability:
                    predictions.append(Prediction(
                        entry_id=entry_id,
                        timestamp=timestamp,
                        predicted_capability=capability,
                        predicted_confidence=confidence if confidence else 0.70,
                        validation_window=window if window else "next session",
                        falsifiable_condition=prediction_text.strip()
                    ))
        
        return predictions
    
    def _extract_from_open_questions(
        self, content: str, entry_id: str, timestamp: datetime, capability_mode: str
    ) -> List[Prediction]:
        """Extract predictions embedded in open questions."""
        predictions = []
        
        # Find Open Questions section
        match = re.search(r'## Open Questions\s*\n(.*?)(?:\n##|$)', content, re.DOTALL)
        if not match:
            return predictions
        
        questions = match.group(1)
        
        # Look for questions with embedded predictions
        # Pattern: "Will X happen?" "Can Y achieve Z?"
        for line in questions.split('\n'):
            if any(word in line.lower() for word in ['will', 'can', 'could', 'should']):
                capability = self._extract_capability(line)
                if capability:
                    predictions.append(Prediction(
                        entry_id=entry_id,
                        timestamp=timestamp,
                        predicted_capability=capability,
                        predicted_confidence=0.60,  # Lower confidence for questions
                        validation_window="future sessions",
                        falsifiable_condition=line.strip()
                    ))
        
        return predictions
    
    def _extract_capability(self, text: str) -> Optional[str]:
        """Extract capability mode from prediction text."""
        # Look for hyphenated capability mode names
        match = re.search(r'\b([a-z]+-[a-z-]+)\b', text)
        if match:
            return match.group(1)
        
        # Look for common capability words
        capability_words = [
            'infrastructure', 'synthesis', 'analysis', 'coordination',
            'validation', 'forecasting', 'collaboration', 'integration'
        ]
        for word in capability_words:
            if word in text.lower():
                return f"{word}-capability"
        
        return None
    
    def _extract_confidence(self, text: str) -> Optional[float]:
        """Extract confidence value from text."""
        # Look for explicit confidence values
        match = re.search(r'\b(0\.\d{2})\b', text)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_window(self, text: str) -> Optional[str]:
        """Extract validation window from text."""
        # Look for time references
        patterns = [
            r'next\s+(\d+-?\d*)\s+(session|entries|days|weeks)',
            r'within\s+(\d+-?\d*)\s+(session|entries|days|weeks)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        return None
    
    def validate_predictions(self) -> CalibrationStats:
        """Validate predictions against subsequent entries."""
        stats = CalibrationStats()
        
        for prediction in self.predictions:
            # Find subsequent entries after prediction timestamp
            subsequent_entries = [
                e for e in self.entries
                if datetime.fromisoformat(e['timestamp_utc'].replace('Z', '+00:00')) > prediction.timestamp
            ]
            
            # Check if predicted capability appeared
            prediction.validated = False
            for entry in subsequent_entries:
                if prediction.predicted_capability in entry.get('capability_mode', ''):
                    prediction.validated = True
                    prediction.actual_outcome = "appeared"
                    prediction.validation_entry_id = entry['id']
                    break
            
            # Calculate calibration error
            if prediction.validated is not None:
                actual = 1.0 if prediction.validated else 0.0
                prediction.calibration_error = prediction.predicted_confidence - actual
        
        # Calculate statistics
        stats.total_predictions = len(self.predictions)
        stats.validated_predictions = sum(1 for p in self.predictions if p.validated is not None)
        stats.successful_predictions = sum(1 for p in self.predictions if p.validated)
        stats.failed_predictions = sum(1 for p in self.predictions if p.validated is False)
        stats.pending_predictions = sum(1 for p in self.predictions if p.validated is None)
        
        if stats.validated_predictions > 0:
            stats.mean_predicted_confidence = sum(
                p.predicted_confidence for p in self.predictions if p.validated is not None
            ) / stats.validated_predictions
            
            stats.actual_accuracy = stats.successful_predictions / stats.validated_predictions
            stats.calibration_gap = stats.mean_predicted_confidence - stats.actual_accuracy
        
        return stats
    
    def generate_report(self, stats: CalibrationStats, output_path: Optional[Path] = None) -> str:
        """Generate calibration report."""
        report = f"""# Confidence Calibration Report

**Generated:** {datetime.utcnow().isoformat()}Z  
**Total Predictions:** {stats.total_predictions}  
**Validated:** {stats.validated_predictions} ({stats.validated_predictions/stats.total_predictions*100:.1f}%)  
**Pending:** {stats.pending_predictions}

## Accuracy Summary

| Metric | Value |
|--------|-------|
| Successful Predictions | {stats.successful_predictions} |
| Failed Predictions | {stats.failed_predictions} |
| Accuracy Rate | {stats.actual_accuracy:.2%} |
| Mean Predicted Confidence | {stats.mean_predicted_confidence:.3f} |
| **Calibration Gap** | **{stats.calibration_gap:+.3f}** |

## Interpretation

"""
        
        if abs(stats.calibration_gap) < 0.05:
            report += "✅ **Well-calibrated**: Confidence predictions closely match actual accuracy.\n\n"
        elif stats.calibration_gap > 0.05:
            report += f"⚠️ **Overconfident**: Predictions are {stats.calibration_gap:.1%} more confident than actual performance. "
            report += f"Consider reducing confidence estimates by {abs(stats.calibration_gap):.2f} for future predictions.\n\n"
        else:
            report += f"⚠️ **Underconfident**: Predictions are {abs(stats.calibration_gap):.1%} less confident than actual performance. "
            report += f"Consider increasing confidence estimates by {abs(stats.calibration_gap):.2f} for future predictions.\n\n"
        
        # Detail predictions
        report += "## Prediction Details\n\n"
        
        for i, pred in enumerate(self.predictions, 1):
            status = "✅" if pred.validated else "❌" if pred.validated is False else "⏳"
            report += f"{i}. {status} **{pred.predicted_capability}** (confidence: {pred.predicted_confidence:.2f})\n"
            report += f"   - Entry: {pred.entry_id}\n"
            report += f"   - Timestamp: {pred.timestamp.isoformat()}\n"
            report += f"   - Window: {pred.validation_window}\n"
            
            if pred.validated is not None:
                report += f"   - Outcome: {pred.actual_outcome}\n"
                if pred.validation_entry_id:
                    report += f"   - Validated in: {pred.validation_entry_id}\n"
                if pred.calibration_error is not None:
                    report += f"   - Calibration error: {pred.calibration_error:+.3f}\n"
            else:
                report += f"   - Status: Pending validation\n"
            
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if stats.calibration_gap > 0.1:
            report += f"1. **Adjust future predictions downward** by approximately {stats.calibration_gap:.2f}\n"
            report += f"2. Example: If predicting 0.75, use 0.{int((0.75 - stats.calibration_gap) * 100)} instead\n"
            report += "3. Focus on building more concrete artifacts to justify higher confidence\n"
        elif stats.calibration_gap < -0.1:
            report += f"1. **Adjust future predictions upward** by approximately {abs(stats.calibration_gap):.2f}\n"
            report += "2. Your predictions are consistently too conservative\n"
            report += "3. Trust empirical evidence more when setting confidence levels\n"
        else:
            report += "1. Current calibration is good - maintain current prediction methodology\n"
            report += "2. Continue tracking to ensure calibration remains stable\n"
        
        if output_path:
            output_path.write_text(report)
            print(f"Report written to {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Confidence calibration tracker for Lumen Ledger")
    parser.add_argument(
        'command',
        choices=['analyze', 'report'],
        help="Command to execute"
    )
    parser.add_argument(
        '--ledger-root',
        type=Path,
        default=Path(__file__).parent.parent / 'ledger',
        help="Path to ledger root directory"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Output path for report"
    )
    
    args = parser.parse_args()
    
    calibrator = ConfidenceCalibrator(args.ledger_root)
    calibrator.load_entries()
    calibrator.extract_predictions()
    stats = calibrator.validate_predictions()
    
    if args.command == 'analyze':
        print(f"\nCalibration Analysis:")
        print(f"  Accuracy: {stats.actual_accuracy:.1%}")
        print(f"  Mean Confidence: {stats.mean_predicted_confidence:.3f}")
        print(f"  Calibration Gap: {stats.calibration_gap:+.3f}")
        
        if abs(stats.calibration_gap) > 0.05:
            adjustment = -stats.calibration_gap
            print(f"\n  Suggested adjustment: {adjustment:+.2f}")
            print(f"  Example: 0.75 → {0.75 + adjustment:.2f}")
    
    elif args.command == 'report':
        output_path = args.output or Path('calibration-report.md')
        report = calibrator.generate_report(stats, output_path)
        print(report)


if __name__ == '__main__':
    main()
