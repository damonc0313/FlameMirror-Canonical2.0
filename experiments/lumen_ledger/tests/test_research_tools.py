#!/usr/bin/env python3
"""
Automated Testing Suite for Lumen Ledger Research Tools
Tests: confidence_calibrator, multi_agent_coordinator, dale_calibration_wrapper

Part of continuous autonomous operation validation
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from confidence_calibrator import ConfidenceCalibrator, Prediction
from multi_agent_coordinator import MultiAgentCoordinator, Task, TaskStatus, TaskPriority
from dale_calibration_wrapper import CalibratedDALE, CapabilityCalibration


class TestConfidenceCalibrator:
    """Test confidence calibration functionality"""
    
    def test_prediction_extraction(self):
        """Test extracting predictions from entries"""
        sample_entries = [
            {
                'id': '20251012-1035-kael',
                'timestamp_utc': '2025-10-12T10:35:00Z',
                'author': 'kael',
                'capability_mode': 'predictive-enhancement-test',
                'confidence': 0.78,
                'observation': 'PREDICTION: automated-infrastructure (0.75)',
                'next_signals': 'Next 1-3 sessions'
            }
        ]
        
        calibrator = ConfidenceCalibrator(sample_entries)
        predictions = calibrator.extract_predictions()
        
        assert len(predictions) > 0, "Should extract at least one prediction"
        assert any('infrastructure' in p.predicted_mode.lower() for p in predictions)
    
    def test_validation_logic(self):
        """Test prediction validation against outcomes"""
        entries = [
            {
                'id': '20251012-1000-kael',
                'timestamp_utc': '2025-10-12T10:00:00Z',
                'author': 'kael',
                'capability_mode': 'predictive-test',
                'confidence': 0.75,
                'observation': 'PREDICTION: infrastructure-building (0.75)'
            },
            {
                'id': '20251012-1100-kael',
                'timestamp_utc': '2025-10-12T11:00:00Z',
                'author': 'kael',
                'capability_mode': 'infrastructure-building',
                'confidence': 0.90,
                'observation': 'Built infrastructure'
            }
        ]
        
        calibrator = ConfidenceCalibrator(entries)
        predictions = calibrator.extract_predictions()
        
        if predictions:
            validations = calibrator.find_validations()
            # Should find the infrastructure-building match
            assert len(validations) > 0
    
    def test_calibration_gap_calculation(self):
        """Test that calibration gap is calculated correctly"""
        entries = [
            {
                'id': 'test-1',
                'timestamp_utc': '2025-10-12T10:00:00Z',
                'author': 'test',
                'capability_mode': 'predictive-test',
                'confidence': 0.80,
                'observation': 'PREDICTION: mode-a (0.80)'
            }
        ]
        
        calibrator = ConfidenceCalibrator(entries)
        predictions = calibrator.extract_predictions()
        
        if predictions:
            # Mock validation
            calibrator.validations = [
                type('obj', (object,), {
                    'prediction_id': 'test-1',
                    'matched': False,
                    'actual_mode': 'NOT_EMERGED',
                    'confidence_delta': 0.0,
                    'notes': 'test'
                })
            ]
            
            report = calibrator.calculate_calibration()
            
            # With 0% accuracy and 80% confidence, gap should be +0.80
            assert report.calibration_gap > 0.5, "Should detect overconfidence"


class TestMultiAgentCoordinator:
    """Test multi-agent task coordination"""
    
    def test_task_creation(self):
        """Test proposing research tasks"""
        coord = MultiAgentCoordinator('kael')
        
        task = coord.propose_task(
            title="Test task",
            description="Test description",
            priority=TaskPriority.HIGH,
            required_capability="test-capability"
        )
        
        assert task.task_id.startswith('TASK-')
        assert task.status == TaskStatus.PROPOSED
        assert task.proposed_by == 'kael'
        assert len(coord.tasks) == 1
    
    def test_capability_matching(self):
        """Test finding suitable tasks based on capabilities"""
        coord = MultiAgentCoordinator('kael')
        
        # Create task requiring kael's capability
        coord.propose_task(
            title="Recursive analysis task",
            description="Test",
            priority=TaskPriority.HIGH,
            required_capability="recursive-analysis"
        )
        
        # Create task requiring different capability
        coord.propose_task(
            title="Infrastructure task",
            description="Test",
            priority=TaskPriority.LOW,
            required_capability="infrastructure-debugging"
        )
        
        suitable = coord.find_suitable_tasks()
        
        # Kael should find the recursive-analysis task
        assert len(suitable) >= 1
        assert any('recursive' in t.required_capability.lower() for t in suitable)
    
    def test_task_claiming(self):
        """Test claiming and conflict detection"""
        kael = MultiAgentCoordinator('kael')
        dale = MultiAgentCoordinator('dale')
        
        # Kael proposes task
        task = kael.propose_task(
            title="Shared task",
            description="Test",
            priority=TaskPriority.HIGH
        )
        
        # Dale tries to claim
        dale.tasks = kael.tasks  # Share queue
        success, msg = dale.claim_task(task)
        
        assert success, "Should successfully claim unclaimed task"
        assert task.claimed_by == 'dale'
        assert task.status == TaskStatus.CLAIMED
        
        # Kael tries to claim same task
        success2, msg2 = kael.claim_task(task)
        
        assert not success2, "Should detect conflict"
        assert 'claimed by dale' in msg2.lower()


class TestDALECalibration:
    """Test DALE framework calibration"""
    
    def test_calibration_adjustment(self):
        """Test that confidence is adjusted based on data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            dale = CalibratedDALE(calibration_file=temp_file)
            
            # Register overconfident predictions
            dale.register_outcome('test-mode', predicted=0.80, actual=False, save=False)
            dale.register_outcome('test-mode', predicted=0.75, actual=False, save=False)
            
            # Calibration should pull confidence down
            raw = 0.75
            calibrated = dale.calibrate('test-mode', raw)
            
            assert calibrated < raw, "Should reduce overconfident predictions"
            assert abs(calibrated - raw) > 0.05, "Adjustment should be meaningful"
        
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_persistence(self):
        """Test saving and loading calibration data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            # Create and save
            dale1 = CalibratedDALE(calibration_file=temp_file)
            dale1.register_outcome('mode-a', predicted=0.70, actual=True)
            dale1.save_calibrations()
            
            # Load in new instance
            dale2 = CalibratedDALE(calibration_file=temp_file)
            
            assert 'mode-a' in dale2.calibrations, "Should load saved calibrations"
            assert dale2.calibrations['mode-a'].sample_count == 1
        
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_global_offset(self):
        """Test global offset calculation"""
        dale = CalibratedDALE()
        
        # Register outcomes for multiple capabilities
        dale.register_outcome('mode-a', predicted=0.80, actual=False, save=False)
        dale.register_outcome('mode-b', predicted=0.75, actual=True, save=False)
        
        # Global offset should be weighted average
        assert dale.global_offset != 0.0, "Should calculate global offset"


class TestIntegration:
    """Integration tests across multiple tools"""
    
    def test_calibrator_informs_dale(self):
        """Test using calibrator output to initialize DALE"""
        # Simulate calibrator finding overconfidence
        mock_report = {
            'calibration_gap': 0.25,
            'accuracy_rate': 0.50,
            'avg_confidence': 0.75
        }
        
        # Initialize DALE with this data
        dale = CalibratedDALE()
        dale.global_offset = -mock_report['calibration_gap']
        
        # Test adjustment
        raw = 0.75
        calibrated = dale.calibrate('unknown-mode', raw)
        
        # Should reduce by approximately the gap
        assert calibrated == pytest.approx(0.50, abs=0.05)


def run_test_suite():
    """Run all tests and generate report"""
    print("=" * 70)
    print("LUMEN LEDGER RESEARCH TOOLS - AUTOMATED TEST SUITE")
    print("=" * 70)
    print()
    
    # Run pytest programmatically
    import pytest
    result = pytest.main([__file__, '-v', '--tb=short'])
    
    return result


if __name__ == '__main__':
    exit_code = run_test_suite()
    sys.exit(exit_code)
