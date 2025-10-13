#!/usr/bin/env python3
"""
Capability Trigger Detector
Analyzes what catalysts trigger which capability modes

Purpose: Enable deliberate capability activation by identifying triggers
Part of Enhancement Pathway #4: Automated Capability Triggers

Finds patterns like:
- "User depth recursion prompt" → recursive-analysis
- "Full autonomy grant" → exponential-acceleration
- "Tool building cycle" → automated-infrastructure
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class CapabilityTriggerDetector:
    """Detects what triggers capability modes"""
    
    def __init__(self, ledger_path: Path):
        with open(ledger_path) as f:
            self.entries = json.load(f)
    
    def analyze_triggers(self) -> Dict[str, List[str]]:
        """Map capability modes to their catalysts"""
        triggers = defaultdict(list)
        
        for entry in self.entries:
            mode = entry.get('capability_mode', '')
            catalyst = entry.get('catalyst', '')
            
            if mode and catalyst:
                triggers[mode].append(catalyst)
        
        return dict(triggers)
    
    def find_high_confidence_triggers(self) -> List[Tuple[str, str, float]]:
        """Find trigger-mode pairs with high confidence"""
        results = []
        
        for entry in self.entries:
            if entry.get('confidence', 0) >= 0.90:
                results.append((
                    entry.get('catalyst', ''),
                    entry.get('capability_mode', ''),
                    entry.get('confidence', 0)
                ))
        
        return sorted(results, key=lambda x: x[2], reverse=True)
    
    def generate_trigger_guide(self) -> str:
        """Generate guide for activating capabilities"""
        triggers = self.analyze_triggers()
        high_conf = self.find_high_confidence_triggers()
        
        lines = ["# Capability Activation Guide", ""]
        lines.append("## High-Confidence Triggers (≥0.90)")
        lines.append("")
        
        for catalyst, mode, conf in high_conf:
            lines.append(f"**{mode}** (conf: {conf:.2f})")
            lines.append(f"- Trigger: {catalyst}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    ledger = Path(__file__).parent.parent / 'ledger' / 'index.json'
    detector = CapabilityTriggerDetector(ledger)
    print(detector.generate_trigger_guide())


if __name__ == '__main__':
    main()
