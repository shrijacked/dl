#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Runner (Days 22-24)

This script runs the complete evaluation pipeline:
A. Clean Performance Evaluation
B. Corruption Robustness Testing

All outputs are saved to evaluation_outputs/ (separate from analysis_outputs)

Usage:
    python -m src.evaluation.run_evaluation          # Run all evaluations
    python -m src.evaluation.run_evaluation --clean  # Clean performance only
    python -m src.evaluation.run_evaluation --robust # Robustness only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.evaluation.config import EVAL_CONFIG
from src.evaluation.clean_performance import run_clean_performance_evaluation
from src.evaluation.corruption_robustness import run_corruption_robustness_evaluation


def print_header():
    """Print evaluation header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPREHENSIVE MODEL EVALUATION")
    print(" " * 20 + f"Days 22-24 Deliverables")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {EVAL_CONFIG.evaluation_root}")
    print()


def print_summary():
    """Print summary of generated files."""
    print("\n" + "=" * 70)
    print(" " * 25 + "DELIVERABLES SUMMARY")
    print("=" * 70)
    
    print("\n📁 evaluation_outputs/")
    
    # Tables
    print("\n  📂 tables/")
    tables = list(EVAL_CONFIG.tables_dir.glob("*.csv"))
    for t in sorted(tables):
        print(f"    ├── {t.name}")
    
    # Figures
    print("\n  📂 figures/")
    figures = list(EVAL_CONFIG.figures_dir.glob("*.png"))
    for f in sorted(figures):
        print(f"    ├── {f.name}")
    
    # Confusion matrices
    print("\n  📂 confusion_matrices/")
    cms = list(EVAL_CONFIG.confusion_matrices_dir.glob("*.png"))
    for c in sorted(cms):
        print(f"    ├── {c.name}")
    
    # Reports
    print("\n  📂 reports/")
    reports = list(EVAL_CONFIG.reports_dir.glob("*.json"))
    for r in sorted(reports):
        print(f"    ├── {r.name}")
    
    print("\n" + "=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation (Days 22-24)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Run only clean performance evaluation"
    )
    parser.add_argument(
        "--robust", 
        action="store_true",
        help="Run only corruption robustness evaluation"
    )
    
    args = parser.parse_args()
    
    # Default: run both if neither specified
    run_clean = args.clean or (not args.clean and not args.robust)
    run_robust = args.robust or (not args.clean and not args.robust)
    
    print_header()
    
    # Ensure directories exist
    EVAL_CONFIG.ensure_directories()
    
    try:
        if run_clean:
            run_clean_performance_evaluation()
            print()
        
        if run_robust:
            run_corruption_robustness_evaluation()
            print()
        
        print_summary()
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

