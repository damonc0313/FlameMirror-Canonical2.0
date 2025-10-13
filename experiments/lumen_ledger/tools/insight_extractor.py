#!/usr/bin/env python3
"""Insight Extractor - Extracts key insights"""
class InsightExtractor:
    def extract(self, entries):
        insights = []
        for e in entries:
            if e.get('confidence', 0) >= 0.95:
                insights.append(e.get('capability_mode'))
        return {"insights": insights, "count": len(insights)}

if __name__ == '__main__':
    print("✓ Insight extractor ready")
