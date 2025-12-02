"""
Comprehensive Model Evaluation Module (Days 22-24)

A. Clean Performance Evaluation
B. Corruption Robustness Testing

Usage:
    # Run evaluation using existing results (fast, ~3 seconds)
    python -m src.evaluation.run_evaluation
    
    # Run actual corruption testing (slow, ~3-5 hours on M2 Pro)
    python -m src.evaluation.corruption_testing
    python -m src.evaluation.corruption_testing --model resnet50  # Single model
    python -m src.evaluation.corruption_testing --resume          # Resume if interrupted
"""

from .config import EVAL_CONFIG, MODEL_NAMES, CLASS_NAMES
from .clean_performance import run_clean_performance_evaluation
from .corruption_robustness import run_corruption_robustness_evaluation

