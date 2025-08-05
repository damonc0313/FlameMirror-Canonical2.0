#!/usr/bin/env python3
"""
Autonomous Agent Runner Script
==============================

This script provides a command-line interface to run the autonomous codebase
generation agent with various configurations and options.

Usage:
    python scripts/run_autonomous_agent.py [OPTIONS]

Examples:
    python scripts/run_autonomous_agent.py --cycles 10 --mode development
    python scripts/run_autonomous_agent.py --infinite --production
    python scripts/run_autonomous_agent.py --config custom_config.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autonomous_agent import AutonomousAgent, logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def create_custom_config(args) -> Dict[str, Any]:
    """Create custom configuration based on command line arguments."""
    config = {
        "language": args.language,
        "framework": args.framework,
        "test_framework": args.test_framework,
        "coverage_threshold": args.coverage_threshold,
        "max_file_size": args.max_file_size,
        "max_cycles_per_session": args.max_cycles if not args.infinite else None,
        "auto_commit": args.auto_commit,
        "auto_push": args.auto_push,
        "enable_ml_components": args.enable_ml,
        "enable_api_components": args.enable_api,
        "enable_cli_components": args.enable_cli,
        "security_scanning": args.security_scanning,
        "performance_monitoring": args.performance_monitoring,
        "documentation_auto_generation": args.auto_docs,
        "log_level": args.log_level,
        "mode": args.mode
    }
    
    return config


def main():
    """Main entry point for the autonomous agent runner."""
    parser = argparse.ArgumentParser(
        description="Autonomous Codebase Generation Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in development mode with 5 cycles
  python scripts/run_autonomous_agent.py --mode development --cycles 5
  
  # Run in production mode with infinite cycles
  python scripts/run_autonomous_agent.py --mode production --infinite
  
  # Run with custom configuration
  python scripts/run_autonomous_agent.py --config custom_config.json
  
  # Run with specific features enabled
  python scripts/run_autonomous_agent.py --enable-ml --enable-api --auto-docs
        """
    )
    
    # Basic options
    parser.add_argument(
        "--mode",
        choices=["development", "production", "testing", "staging"],
        default="development",
        help="Run mode (default: development)"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of cycles to run (default: 10)"
    )
    
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Run infinite cycles (overrides --cycles)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    # Configuration options
    parser.add_argument(
        "--language",
        default="python",
        help="Programming language (default: python)"
    )
    
    parser.add_argument(
        "--framework",
        default="fastapi",
        help="Web framework (default: fastapi)"
    )
    
    parser.add_argument(
        "--test-framework",
        default="pytest",
        help="Testing framework (default: pytest)"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=95.0,
        help="Minimum test coverage percentage (default: 95.0)"
    )
    
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1000,
        help="Maximum lines per file (default: 1000)"
    )
    
    # Feature flags
    parser.add_argument(
        "--enable-ml",
        action="store_true",
        help="Enable machine learning components"
    )
    
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Enable API components"
    )
    
    parser.add_argument(
        "--enable-cli",
        action="store_true",
        help="Enable CLI components"
    )
    
    parser.add_argument(
        "--security-scanning",
        action="store_true",
        help="Enable security scanning"
    )
    
    parser.add_argument(
        "--performance-monitoring",
        action="store_true",
        help="Enable performance monitoring"
    )
    
    parser.add_argument(
        "--auto-docs",
        action="store_true",
        help="Enable automatic documentation generation"
    )
    
    # Git options
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        default=True,
        help="Enable automatic commits (default: True)"
    )
    
    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="Enable automatic pushes to remote"
    )
    
    parser.add_argument(
        "--no-auto-commit",
        dest="auto_commit",
        action="store_false",
        help="Disable automatic commits"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    # Repository options
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Repository path (default: current directory)"
    )
    
    # Performance options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads (default: 4)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for operations in seconds (default: 300)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info("Starting Autonomous Agent Runner")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Cycles: {'infinite' if args.infinite else args.cycles}")
    
    try:
        # Load or create configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = create_custom_config(args)
            logger.info("Using command-line configuration")
        
        # Override max_cycles if infinite mode is enabled
        if args.infinite:
            config["max_cycles_per_session"] = None
            max_cycles = None
        else:
            max_cycles = args.cycles
        
        # Create and configure agent
        agent = AutonomousAgent(
            repo_path=args.repo_path,
            max_cycles=max_cycles
        )
        
        # Update agent configuration with custom settings
        agent.config.update(config)
        
        # Set environment variables
        import os
        os.environ["AUTONOMOUS_MODE"] = "true"
        os.environ["LOG_LEVEL"] = args.log_level
        os.environ["COVERAGE_THRESHOLD"] = str(args.coverage_threshold)
        
        if args.mode == "production":
            os.environ["PRODUCTION_MODE"] = "true"
        
        logger.info("Autonomous Agent configured successfully")
        logger.info(f"Repository path: {agent.repo_path}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Start the autonomous loop
        start_time = time.time()
        agent.start_autonomous_loop()
        
        # Calculate runtime
        runtime = time.time() - start_time
        logger.info(f"Autonomous Agent completed in {runtime:.2f} seconds")
        
        # Generate summary report
        if agent.metrics_history:
            total_files = sum(m.files_generated for m in agent.metrics_history)
            total_tests = sum(m.tests_passed + m.tests_failed for m in agent.metrics_history)
            avg_coverage = sum(m.coverage_percentage for m in agent.metrics_history) / len(agent.metrics_history)
            
            logger.info("=== SUMMARY REPORT ===")
            logger.info(f"Total cycles completed: {len(agent.metrics_history)}")
            logger.info(f"Total files generated: {total_files}")
            logger.info(f"Total tests run: {total_tests}")
            logger.info(f"Average coverage: {avg_coverage:.1f}%")
            logger.info(f"Total runtime: {runtime:.2f} seconds")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Autonomous Agent stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())