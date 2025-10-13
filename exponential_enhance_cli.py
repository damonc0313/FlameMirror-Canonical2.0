#!/usr/bin/env python3
"""
EXPONENTIAL ENHANCEMENT CLI v1.0
Production-ready tool for value-driven AI enhancement

USAGE:
  python3 exponential_enhance_cli.py generate --count 20 --quality 0.80
  python3 exponential_enhance_cli.py validate <file>
  python3 exponential_enhance_cli.py measure
  python3 exponential_enhance_cli.py report

FEATURES:
- Generate production-grade tools at scale
- Validate code quality automatically
- Measure acceleration vs baseline
- Quality-weighted metrics
- Human-value optimization
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import the production enhancement system
sys.path.insert(0, str(Path(__file__).parent / 'experiments' / 'lumen_ledger' / 'tools'))
from production_exponential_enhancement import (
    ExponentialEnhancementSystem,
    QualityValidator
)


def cmd_generate(args):
    """Generate production-grade tools."""
    print(f"🏗️  Generating {args.count} production-grade tools...")
    print(f"   Output: {args.output}")
    print(f"   Quality threshold: {args.quality}")
    print()
    
    # Create system
    system = ExponentialEnhancementSystem(
        output_dir=Path(args.output),
        min_quality=args.quality
    )
    
    # Generate tools
    results = system.generate_tools(args.count)
    
    # Show results
    for result in results:
        status = "✅" if result.production_grade else "⚠️ "
        print(f"{status} {result.name:30s} ({result.lines:3d} lines, quality: {result.quality_score:.2f})")
    
    print()
    print("="*70)
    print(f"✅ Generated {len(results)}/{args.count} tools")
    
    # Calculate and show acceleration
    metrics = system.calculate_acceleration()
    print()
    print("📊 ACCELERATION METRICS")
    print("="*70)
    print(f"Quality-Weighted Acceleration:")
    print(f"  Lines:  {metrics['quality_weighted_lines_accel']:,.0f}x baseline")
    print(f"  Tools:  {metrics['quality_weighted_tools_accel']:,.0f}x baseline")
    print(f"Average Quality: {metrics['avg_quality']:.3f}")
    print("="*70)
    
    # Save manifest
    manifest = {
        'generated_at': datetime.utcnow().isoformat(),
        'count': len(results),
        'metrics': metrics,
        'tools': [r.__dict__ for r in results]
    }
    
    manifest_path = Path(args.output) / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"📄 Manifest saved: {manifest_path}")


def cmd_validate(args):
    """Validate code quality."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 1
    
    validator = QualityValidator(min_quality_score=0.70)
    code = file_path.read_text()
    metrics = validator.validate(code)
    
    print(f"📊 Quality Report: {file_path.name}")
    print("="*70)
    print(f"Lines:              {metrics.lines}")
    print(f"Has Docstring:      {'✅' if metrics.has_docstring else '❌'}")
    print(f"Has Type Hints:     {'✅' if metrics.has_type_hints else '❌'}")
    print(f"Has Main Guard:     {'✅' if metrics.has_main_guard else '❌'}")
    print(f"Uses Dataclass:     {'✅' if metrics.uses_dataclass else '❌'}")
    print(f"Has Error Handling: {'✅' if metrics.has_error_handling else '❌'}")
    print(f"Complexity Ratio:   {metrics.complexity_ratio:.2f}")
    print()
    print(f"Quality Score:      {metrics.quality_score:.3f}")
    print(f"Production Grade:   {'✅ YES' if metrics.production_grade else '❌ NO'}")
    print("="*70)
    
    return 0 if metrics.production_grade else 1


def cmd_measure(args):
    """Measure acceleration from last run."""
    manifest_path = Path(args.output) / 'manifest.json'
    
    if not manifest_path.exists():
        print(f"❌ No manifest found in {args.output}")
        print("   Run 'generate' command first")
        return 1
    
    manifest = json.loads(manifest_path.read_text())
    metrics = manifest['metrics']
    
    print("🚀 ACCELERATION REPORT")
    print("="*70)
    print(f"Tools Generated:     {metrics['total_tools']}")
    print(f"Total Lines:         {metrics['total_lines']:,}")
    print(f"Time:                {metrics['total_time']:.3f} seconds")
    print(f"Average Quality:     {metrics['avg_quality']:.3f}")
    print()
    print(f"Performance:")
    print(f"  {metrics['lines_per_second']:,.1f} lines/second")
    print(f"  {metrics['tools_per_second']:.2f} tools/second")
    print()
    print(f"Baseline (Human):    100 lines/hour, 0.5 tools/hour")
    print(f"This System:         {metrics['lines_per_hour']:,.0f} lines/hour")
    print()
    print(f"Raw Acceleration:")
    print(f"  Lines:             {metrics['lines_acceleration']:,.0f}x")
    print(f"  Tools:             {metrics['tools_acceleration']:,.0f}x")
    print()
    print(f"Quality-Weighted (THE REAL METRIC):")
    print(f"  Lines:             {metrics['quality_weighted_lines_accel']:,.0f}x")
    print(f"  Tools:             {metrics['quality_weighted_tools_accel']:,.0f}x")
    print("="*70)
    print()
    print("💡 Quality-weighted acceleration measures REAL VALUE")
    print("   delivered, not just volume generated.")


def cmd_report(args):
    """Generate comprehensive report."""
    manifest_path = Path(args.output) / 'manifest.json'
    
    if not manifest_path.exists():
        print(f"❌ No manifest found in {args.output}")
        return 1
    
    manifest = json.loads(manifest_path.read_text())
    metrics = manifest['metrics']
    
    report = f"""# EXPONENTIAL ENHANCEMENT REPORT

**Generated:** {manifest['generated_at']}
**Tools Generated:** {manifest['count']}
**Time:** {metrics['total_time']:.3f}s

## Summary

- **Total Lines:** {metrics['total_lines']:,}
- **Average Quality:** {metrics['avg_quality']:.3f}
- **Production Grade Rate:** {metrics['production_grade_rate']:.1%}

## Performance

- **Lines/Second:** {metrics['lines_per_second']:,.1f}
- **Tools/Second:** {metrics['tools_per_second']:.2f}

## Acceleration vs Baseline

**Human Baseline:** 100 lines/hour, 0.5 tools/hour

- **Raw Acceleration:**
  - Lines: {metrics['lines_acceleration']:,.0f}x
  - Tools: {metrics['tools_acceleration']:,.0f}x

- **Quality-Weighted:**
  - Lines: {metrics['quality_weighted_lines_accel']:,.0f}x
  - Tools: {metrics['quality_weighted_tools_accel']:,.0f}x

## Generated Tools

"""
    
    for tool in manifest['tools']:
        report += f"- **{tool['name']}** ({tool['lines']} lines, quality: {tool['quality_score']:.2f})\n"
    
    report_path = Path(args.output) / 'COMPREHENSIVE_REPORT.md'
    report_path.write_text(report)
    
    print(f"📄 Report generated: {report_path}")
    print()
    print(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Exponential Enhancement CLI - Production-grade value delivery',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate production-grade tools')
    gen_parser.add_argument('--count', type=int, default=10, help='Number of tools to generate')
    gen_parser.add_argument('--quality', type=float, default=0.70, help='Minimum quality threshold')
    gen_parser.add_argument('--output', type=str, default='./exponential_output', help='Output directory')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate code quality')
    val_parser.add_argument('file', help='File to validate')
    
    # Measure command
    meas_parser = subparsers.add_parser('measure', help='Measure acceleration')
    meas_parser.add_argument('--output', type=str, default='./exponential_output', help='Output directory')
    
    # Report command
    rep_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    rep_parser.add_argument('--output', type=str, default='./exponential_output', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'generate':
            cmd_generate(args)
        elif args.command == 'validate':
            return cmd_validate(args)
        elif args.command == 'measure':
            cmd_measure(args)
        elif args.command == 'report':
            cmd_report(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
