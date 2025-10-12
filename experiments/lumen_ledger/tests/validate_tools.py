#!/usr/bin/env python3
"""
Standalone Tool Validation Script
Tests research tools without external dependencies

Part of continuous autonomous operation - ensuring quality at velocity
"""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from confidence_calibrator import ConfidenceCalibrator
from multi_agent_coordinator import MultiAgentCoordinator, TaskPriority
from dale_calibration_wrapper import CalibratedDALE


def test_confidence_calibrator():
    """Validate confidence calibrator functionality"""
    print("Testing Confidence Calibrator...")
    
    sample_entries = [
        {
            'id': 'test-1',
            'timestamp_utc': '2025-10-12T10:00:00Z',
            'author': 'test',
            'capability_mode': 'predictive-test',
            'confidence': 0.75,
            'observation': 'PREDICTION: infrastructure-mode (0.75)'
        },
        {
            'id': 'test-2',
            'timestamp_utc': '2025-10-12T11:00:00Z',
            'author': 'test',
            'capability_mode': 'infrastructure-mode',
            'confidence': 0.90,
            'observation': 'Built infrastructure'
        }
    ]
    
    calibrator = ConfidenceCalibrator(sample_entries)
    predictions = calibrator.extract_predictions()
    
    assert len(predictions) > 0, "Should extract predictions"
    print(f"  ✓ Extracted {len(predictions)} predictions")
    
    validations = calibrator.find_validations()
    print(f"  ✓ Validated {len(validations)} predictions")
    
    report = calibrator.calculate_calibration()
    print(f"  ✓ Calibration gap: {report.calibration_gap:+.3f}")
    
    return True


def test_multi_agent_coordinator():
    """Validate multi-agent coordination"""
    print("\nTesting Multi-Agent Coordinator...")
    
    kael = MultiAgentCoordinator('kael')
    
    # Create tasks
    task1 = kael.propose_task(
        title="Test recursive analysis",
        description="Test task",
        priority=TaskPriority.HIGH,
        required_capability="recursive-analysis"
    )
    
    assert task1.task_id.startswith('TASK-'), "Task ID should be generated"
    print(f"  ✓ Created task: {task1.title}")
    
    # Find suitable tasks
    suitable = kael.find_suitable_tasks()
    assert len(suitable) > 0, "Should find suitable tasks"
    print(f"  ✓ Found {len(suitable)} suitable tasks for Kael")
    
    # Test claiming
    success, msg = kael.claim_task(task1)
    assert success, "Should successfully claim task"
    print(f"  ✓ Task claiming works: {msg}")
    
    # Generate report
    report = kael.generate_work_report()
    assert len(report) > 100, "Report should have content"
    print(f"  ✓ Generated coordination report ({len(report)} chars)")
    
    return True


def test_dale_calibration():
    """Validate DALE calibration wrapper"""
    print("\nTesting DALE Calibration Wrapper...")
    
    dale = CalibratedDALE()
    
    # Register outcomes
    dale.register_outcome('test-mode', predicted=0.80, actual=False, save=False)
    dale.register_outcome('test-mode', predicted=0.75, actual=True, save=False)
    
    print(f"  ✓ Registered 2 outcomes for test-mode")
    
    # Test calibration
    raw = 0.75
    calibrated = dale.calibrate('test-mode', raw)
    
    assert calibrated != raw, "Should adjust confidence"
    print(f"  ✓ Calibration: {raw:.2f} → {calibrated:.2f} (Δ {calibrated-raw:+.2f})")
    
    # Test global offset
    assert dale.global_offset != 0.0, "Should calculate global offset"
    print(f"  ✓ Global offset: {dale.global_offset:+.3f}")
    
    # Test report generation
    report = dale.generate_report()
    assert "DALE CALIBRATION STATUS" in report, "Report should be formatted"
    print(f"  ✓ Generated calibration report")
    
    return True


def run_all_tests():
    """Execute all validation tests"""
    print("=" * 70)
    print("LUMEN LEDGER TOOLS VALIDATION")
    print("=" * 70)
    print()
    
    tests = [
        ("Confidence Calibrator", test_confidence_calibrator),
        ("Multi-Agent Coordinator", test_multi_agent_coordinator),
        ("DALE Calibration", test_dale_calibration)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
