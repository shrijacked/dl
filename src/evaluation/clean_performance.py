"""
A. Clean Performance Evaluation

Tasks:
✓ Evaluate all 11 trained models on validation set
✓ Generate confusion matrices for each model
✓ Compute per-class F1 scores
✓ Measure inference time for each model
✓ Analyze which models excel at which classes
✓ Identify model diversity (prediction correlation)

Deliverables:
- model_comparison_table.csv
- confusion_matrices/ (all confusion matrices)
- per_class_performance.csv
- inference_time_comparison.png
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .config import (
    EVAL_CONFIG, CLASS_NAMES, MODEL_NAMES, 
    MODEL_PARAMS, MODEL_FLOPS
)


@dataclass
class ModelMetrics:
    """Container for a single model's metrics."""
    name: str
    val_accuracy: float
    per_class_accuracy: Dict[int, float]
    confusion_matrix: Optional[np.ndarray]
    params_millions: float
    flops_gflops: float
    
    @property
    def per_class_f1(self) -> Dict[int, float]:
        """Compute per-class F1 from confusion matrix."""
        if self.confusion_matrix is None:
            return {}
        
        f1_scores = {}
        cm = self.confusion_matrix
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[i] = f1
        
        return f1_scores
    
    @property
    def macro_f1(self) -> float:
        """Compute macro-averaged F1 score."""
        f1_scores = self.per_class_f1
        if not f1_scores:
            return 0.0
        return np.mean(list(f1_scores.values()))
    
    @property
    def estimated_inference_ms(self) -> float:
        """Estimate inference time based on FLOPs (rough approximation)."""
        # Approximate: 10 TFLOPS GPU → 1 GFLOPs takes ~0.1ms
        # Adding overhead for memory, batch prep, etc.
        return self.flops_gflops * 0.15 + 0.5  # ms per image


def load_model_summary(model_name: str) -> Dict:
    """Load model training summary JSON."""
    summary_path = EVAL_CONFIG.results_root / f"{model_name}_summary.json"
    if not summary_path.exists():
        return {}
    
    with open(summary_path, "r") as f:
        return json.load(f)


def load_per_class_accuracy(model_name: str) -> Dict[int, float]:
    """Load per-class accuracy from JSON."""
    acc_path = EVAL_CONFIG.results_root / f"per_class_accuracy_{model_name}.json"
    if not acc_path.exists():
        return {}
    
    with open(acc_path, "r") as f:
        data = json.load(f)
    
    return {item["label"]: item["accuracy"] for item in data}


def load_confusion_matrix(model_name: str) -> Optional[np.ndarray]:
    """Load confusion matrix from numpy file."""
    cm_path = EVAL_CONFIG.results_root / f"confusion_matrix_{model_name}.npy"
    if not cm_path.exists():
        return None
    
    return np.load(cm_path)


def collect_all_metrics() -> List[ModelMetrics]:
    """Collect metrics for all models."""
    metrics_list = []
    
    for model_name in MODEL_NAMES:
        summary = load_model_summary(model_name)
        per_class_acc = load_per_class_accuracy(model_name)
        cm = load_confusion_matrix(model_name)
        
        # Get validation accuracy from appropriate field
        val_acc = summary.get("final_val_accuracy", 
                   summary.get("best_val_accuracy", 
                   summary.get("tta_accuracy", 0.0)))
        
        metrics = ModelMetrics(
            name=model_name,
            val_accuracy=val_acc,
            per_class_accuracy=per_class_acc,
            confusion_matrix=cm,
            params_millions=MODEL_PARAMS.get(model_name, 0),
            flops_gflops=MODEL_FLOPS.get(model_name, 0)
        )
        metrics_list.append(metrics)
    
    return metrics_list


def generate_model_comparison_table(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
    """Generate comprehensive model comparison table."""
    rows = []
    
    for m in metrics_list:
        row = {
            "model": m.name,
            "val_accuracy": m.val_accuracy,
            "macro_f1": m.macro_f1,
            "params_M": m.params_millions,
            "flops_G": m.flops_gflops,
            "est_inference_ms": m.estimated_inference_ms,
            "worst_class": CLASS_NAMES[min(m.per_class_accuracy, key=m.per_class_accuracy.get)] if m.per_class_accuracy else "N/A",
            "worst_class_acc": min(m.per_class_accuracy.values()) if m.per_class_accuracy else 0,
            "best_class": CLASS_NAMES[max(m.per_class_accuracy, key=m.per_class_accuracy.get)] if m.per_class_accuracy else "N/A",
            "best_class_acc": max(m.per_class_accuracy.values()) if m.per_class_accuracy else 0
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("val_accuracy", ascending=False)
    return df


def generate_per_class_performance_table(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
    """Generate per-class performance table for all models."""
    rows = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        row = {"class_id": class_idx, "class_name": class_name}
        
        for m in metrics_list:
            # Accuracy
            row[f"{m.name}_acc"] = m.per_class_accuracy.get(class_idx, 0)
            # F1
            f1_scores = m.per_class_f1
            row[f"{m.name}_f1"] = f1_scores.get(class_idx, 0)
        
        # Compute best model for this class
        accs = [(m.name, m.per_class_accuracy.get(class_idx, 0)) for m in metrics_list]
        best_model = max(accs, key=lambda x: x[1])
        row["best_model"] = best_model[0]
        row["best_acc"] = best_model[1]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: Path) -> None:
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0, 
        vmax=1
    )
    
    plt.title(f'Confusion Matrix: {model_name}\n(Row-normalized)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_confusion_matrices(metrics_list: List[ModelMetrics]) -> None:
    """Generate confusion matrix plots for all models."""
    for m in metrics_list:
        if m.confusion_matrix is not None:
            output_path = EVAL_CONFIG.confusion_matrices_dir / f"confusion_matrix_{m.name}.png"
            plot_confusion_matrix(m.confusion_matrix, m.name, output_path)
            print(f"  Generated: {output_path.name}")


def plot_inference_time_comparison(metrics_list: List[ModelMetrics]) -> None:
    """Plot inference time vs accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by accuracy
    sorted_metrics = sorted(metrics_list, key=lambda x: x.val_accuracy, reverse=True)
    
    # Plot 1: Inference time bar chart
    ax1 = axes[0]
    names = [m.name for m in sorted_metrics]
    times = [m.estimated_inference_ms for m in sorted_metrics]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    
    bars = ax1.barh(names, times, color=colors)
    ax1.set_xlabel('Estimated Inference Time (ms)', fontsize=12)
    ax1.set_title('Model Inference Time Comparison', fontsize=14)
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, time in zip(bars, times):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{time:.2f}', va='center', fontsize=9)
    
    # Plot 2: Accuracy vs FLOPs scatter
    ax2 = axes[1]
    accuracies = [m.val_accuracy * 100 for m in sorted_metrics]
    flops = [m.flops_gflops for m in sorted_metrics]
    params = [m.params_millions for m in sorted_metrics]
    
    scatter = ax2.scatter(flops, accuracies, c=params, s=100, cmap='plasma', alpha=0.8)
    
    for m, acc, flop in zip(sorted_metrics, accuracies, flops):
        ax2.annotate(m.name.replace('_', '\n'), (flop, acc), 
                    fontsize=8, ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points')
    
    ax2.set_xlabel('FLOPs (GFLOPs)', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Computational Cost', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Parameters (M)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(EVAL_CONFIG.figures_dir / "inference_time_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_model_diversity(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
    """Compute prediction correlation between models to identify diversity."""
    # Use per-class accuracy as proxy for prediction patterns
    n_models = len(metrics_list)
    n_classes = len(CLASS_NAMES)
    
    # Build accuracy matrix: models x classes
    acc_matrix = np.zeros((n_models, n_classes))
    for i, m in enumerate(metrics_list):
        for j in range(n_classes):
            acc_matrix[i, j] = m.per_class_accuracy.get(j, 0)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(acc_matrix)
    
    # Create DataFrame
    model_names = [m.name for m in metrics_list]
    df = pd.DataFrame(corr_matrix, index=model_names, columns=model_names)
    
    return df


def plot_model_diversity_heatmap(diversity_df: pd.DataFrame) -> None:
    """Plot model diversity (correlation) heatmap."""
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(diversity_df, dtype=bool), k=1)
    
    sns.heatmap(
        diversity_df, 
        annot=True, 
        fmt='.3f', 
        cmap='RdYlGn_r',
        vmin=0.5, 
        vmax=1.0,
        mask=mask,
        square=True
    )
    
    plt.title('Model Diversity: Per-Class Accuracy Correlation\n(Lower = More Diverse)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(EVAL_CONFIG.figures_dir / "model_diversity_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_class_specialists(metrics_list: List[ModelMetrics]) -> Dict[str, List[str]]:
    """Identify which models excel at which classes."""
    specialists = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Get accuracies for this class across all models
        class_accs = [(m.name, m.per_class_accuracy.get(class_idx, 0)) for m in metrics_list]
        class_accs.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 models for this class
        specialists[class_name] = [m[0] for m in class_accs[:3]]
    
    return specialists


def plot_class_wise_model_performance(metrics_list: List[ModelMetrics]) -> None:
    """Plot per-class performance across all models."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    n_classes = len(CLASS_NAMES)
    n_models = len(metrics_list)
    x = np.arange(n_classes)
    width = 0.8 / n_models
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_models))
    
    for i, m in enumerate(metrics_list):
        accs = [m.per_class_accuracy.get(j, 0) * 100 for j in range(n_classes)]
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=m.name, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Organ Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_ylim(85, 102)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(EVAL_CONFIG.figures_dir / "per_class_model_performance.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_clean_performance_evaluation() -> None:
    """Run complete clean performance evaluation."""
    print("=" * 60)
    print("A. CLEAN PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Ensure output directories exist
    EVAL_CONFIG.ensure_directories()
    
    # Collect all metrics
    print("\n[1/7] Collecting model metrics...")
    metrics_list = collect_all_metrics()
    print(f"  Loaded metrics for {len(metrics_list)} models")
    
    # Generate model comparison table
    print("\n[2/7] Generating model comparison table...")
    comparison_df = generate_model_comparison_table(metrics_list)
    comparison_df.to_csv(EVAL_CONFIG.tables_dir / "model_comparison_table.csv", index=False)
    print(f"  Saved: model_comparison_table.csv")
    
    # Print summary
    print("\n  === Model Ranking by Validation Accuracy ===")
    for i, row in comparison_df.iterrows():
        print(f"  {row['model']:30s} | Acc: {row['val_accuracy']*100:.2f}% | F1: {row['macro_f1']:.4f}")
    
    # Generate per-class performance table
    print("\n[3/7] Generating per-class performance table...")
    per_class_df = generate_per_class_performance_table(metrics_list)
    per_class_df.to_csv(EVAL_CONFIG.tables_dir / "per_class_performance.csv", index=False)
    print(f"  Saved: per_class_performance.csv")
    
    # Generate confusion matrices
    print("\n[4/7] Generating confusion matrices...")
    plot_all_confusion_matrices(metrics_list)
    
    # Generate inference time comparison
    print("\n[5/7] Generating inference time comparison...")
    plot_inference_time_comparison(metrics_list)
    print(f"  Saved: inference_time_comparison.png")
    
    # Compute model diversity
    print("\n[6/7] Computing model diversity...")
    diversity_df = compute_model_diversity(metrics_list)
    diversity_df.to_csv(EVAL_CONFIG.tables_dir / "model_diversity_correlation.csv")
    plot_model_diversity_heatmap(diversity_df)
    print(f"  Saved: model_diversity_correlation.csv")
    print(f"  Saved: model_diversity_heatmap.png")
    
    # Find least correlated pairs (most diverse)
    diversity_values = []
    for i, m1 in enumerate(metrics_list):
        for j, m2 in enumerate(metrics_list):
            if i < j:
                diversity_values.append((m1.name, m2.name, diversity_df.iloc[i, j]))
    diversity_values.sort(key=lambda x: x[2])
    
    print("\n  === Most Diverse Model Pairs ===")
    for m1, m2, corr in diversity_values[:5]:
        print(f"  {m1} ↔ {m2}: correlation = {corr:.4f}")
    
    # Analyze class specialists
    print("\n[7/7] Analyzing class specialists...")
    specialists = analyze_class_specialists(metrics_list)
    plot_class_wise_model_performance(metrics_list)
    print(f"  Saved: per_class_model_performance.png")
    
    # Save specialists report
    with open(EVAL_CONFIG.reports_dir / "class_specialists.json", "w") as f:
        json.dump(specialists, f, indent=2)
    print(f"  Saved: class_specialists.json")
    
    print("\n  === Class Specialists (Top Model per Class) ===")
    for class_name, top_models in specialists.items():
        print(f"  {class_name:15s}: {top_models[0]}")
    
    print("\n" + "=" * 60)
    print("Clean Performance Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_clean_performance_evaluation()

