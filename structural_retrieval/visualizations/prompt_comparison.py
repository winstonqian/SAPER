#!/usr/bin/env python3
"""
Generate bar graph comparing baseline vs hybrid weighted vs enhanced prompt methods
across four datasets (protein_function_OOD, general_function_OOD,
domain_motif_OOD, catalytic_activity_OOD).

Compares three methods:
1. Baseline (no structural embeddings) - RED
2. Hybrid Weighted Similarity (alpha=0.7, with structural embeddings) - GREEN
3. Enhanced Prompt (alpha=0.7, with structural embeddings) - YELLOW

Results are averaged over 3 independent runs of 256 samples each.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_results_file(filepath):
    """
    Parse a results file and extract metrics by task.

    Returns:
        dict: {task_name: {metric_name: [run1, run2, run3]}}
    """
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by tasks
    task_blocks = re.split(r'(?:now task:|Task:)', content)[1:]  # Skip first empty split

    for block in task_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        # Extract task name from first line
        task_match = re.search(r'(\w+_OOD)', lines[0])
        if not task_match:
            continue
        task_name = task_match.group(1)

        if task_name not in results:
            results[task_name] = {}

        # Parse metrics from this task block
        for line in lines:
            # Meta-BLEU scores (check these first as they contain "BLEU")
            if 'Meta-BLEU-2 score:' in line:
                score = float(re.search(r'Meta-BLEU-2 score:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('Meta-BLEU-2', []).append(score)
            elif 'Meta-BLEU-4 score:' in line:
                score = float(re.search(r'Meta-BLEU-4 score:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('Meta-BLEU-4', []).append(score)
            # BLEU scores (check after Meta-BLEU)
            elif 'BLEU-2 score:' in line:
                score = float(re.search(r'BLEU-2 score:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('BLEU-2', []).append(score)
            elif 'BLEU-4 score:' in line:
                score = float(re.search(r'BLEU-4 score:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('BLEU-4', []).append(score)
            # METEOR score
            elif 'Average Meteor score:' in line:
                score = float(re.search(r'Average Meteor score:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('METEOR', []).append(score)
            # ROUGE scores
            elif 'rouge1:' in line:
                score = float(re.search(r'rouge1:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('ROUGE-1', []).append(score)
            elif 'rouge2:' in line:
                score = float(re.search(r'rouge2:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('ROUGE-2', []).append(score)
            elif 'rougeL:' in line:
                score = float(re.search(r'rougeL:\s*([\d.]+)', line).group(1))
                results[task_name].setdefault('ROUGE-L', []).append(score)

    return results


def compute_averages(results):
    """
    Compute average scores across runs for each task and metric.

    Returns:
        dict: {task_name: {metric_name: average_score}}
    """
    averages = {}
    for task, metrics in results.items():
        averages[task] = {}
        for metric, values in metrics.items():
            averages[task][metric] = np.mean(values)
    return averages


def create_comparison_plot(baseline_results, hybrid_weighted_results, enhanced_prompt_results,
                          metric='METEOR', output_path='../../report/figs/performance_comparison.png'):
    """
    Create a bar plot with error bars comparing the three methods across datasets.

    Args:
        baseline_results: dict of {task: {metric: [run1, run2, run3]}}
        hybrid_weighted_results: dict of {task: {metric: [run1, run2, run3]}}
        enhanced_prompt_results: dict of {task: {metric: [run1, run2, run3]}}
        metric: which metric to plot (default: METEOR)
        output_path: where to save the figure
    """
    # Define the four tasks in order
    tasks = ['protein_function_OOD', 'general_function_OOD',
             'domain_motif_OOD', 'catalytic_activity_OOD']

    # Prettier task names for display
    task_labels = {
        'protein_function_OOD': 'Protein Function',
        'general_function_OOD': 'General Function',
        'domain_motif_OOD': 'Domain Motif',
        'catalytic_activity_OOD': 'Catalytic Activity'
    }

    # Standard colors: red, green, yellow
    color_baseline = 'red'
    color_weighted = 'green'
    color_enhanced = 'gold'

    # Extract means and standard deviations
    baseline_means = [np.mean(baseline_results[task][metric]) for task in tasks]
    weighted_means = [np.mean(hybrid_weighted_results[task][metric]) for task in tasks]
    enhanced_means = [np.mean(enhanced_prompt_results[task][metric]) for task in tasks]

    baseline_stds = [np.std(baseline_results[task][metric], ddof=1) for task in tasks]
    weighted_stds = [np.std(hybrid_weighted_results[task][metric], ddof=1) for task in tasks]
    enhanced_stds = [np.std(enhanced_prompt_results[task][metric], ddof=1) for task in tasks]

    # Set up the bar plot with narrower figure
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with error bars
    bars1 = ax.bar(x - width, baseline_means, width, label='Baseline (No Structural)',
                   color=color_baseline, alpha=0.7, yerr=baseline_stds,
                   capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
    bars2 = ax.bar(x, weighted_means, width, label='Hybrid Weighted (α=0.7, Structural)',
                   color=color_weighted, alpha=0.7, yerr=weighted_stds,
                   capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
    bars3 = ax.bar(x + width, enhanced_means, width,
                   label='Enhanced Prompt (α=0.7, Structural)',
                   color=color_enhanced, alpha=0.7, yerr=enhanced_stds,
                   capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})

    # Add value labels above error bars
    def add_value_labels(bars, means, stds):
        for bar, mean, std in zip(bars, means, stds):
            height = mean + std
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1, baseline_means, baseline_stds)
    add_value_labels(bars2, weighted_means, weighted_stds)
    add_value_labels(bars3, enhanced_means, enhanced_stds)

    # Customize plot
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison: Prompt Enhancement for Hybrid Weighted Similarity\n({metric} Score, Averaged over 3 runs of 256 samples)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in tasks])
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add some padding to y-axis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig, ax


def create_multi_metric_comparison(baseline_results, hybrid_weighted_results, enhanced_prompt_results,
                                   metrics=['Meta-BLEU-2', 'Meta-BLEU-4'],
                                   output_path='../../report/figs/multi_metric_comparison.png'):
    """
    Create a faceted bar plot with error bars comparing multiple metrics across datasets.
    """
    tasks = ['protein_function_OOD', 'general_function_OOD',
             'domain_motif_OOD', 'catalytic_activity_OOD']

    task_labels = {
        'protein_function_OOD': 'Protein Function',
        'general_function_OOD': 'General Function',
        'domain_motif_OOD': 'Domain Motif',
        'catalytic_activity_OOD': 'Catalytic Activity'
    }

    # Standard colors
    color_baseline = 'red'
    color_weighted = 'green'
    color_enhanced = 'gold'

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx] if len(metrics) > 1 else axes

        # Extract means and standard deviations
        baseline_means = [np.mean(baseline_results[task][metric]) for task in tasks]
        weighted_means = [np.mean(hybrid_weighted_results[task][metric]) for task in tasks]
        enhanced_means = [np.mean(enhanced_prompt_results[task][metric]) for task in tasks]

        baseline_stds = [np.std(baseline_results[task][metric], ddof=1) for task in tasks]
        weighted_stds = [np.std(hybrid_weighted_results[task][metric], ddof=1) for task in tasks]
        enhanced_stds = [np.std(enhanced_prompt_results[task][metric], ddof=1) for task in tasks]

        x = np.arange(len(tasks))
        width = 0.25

        # Create bars with error bars
        bars1 = ax.bar(x - width, baseline_means, width, label='Baseline',
                      color=color_baseline, alpha=0.7, yerr=baseline_stds,
                      capsize=4, error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
        bars2 = ax.bar(x, weighted_means, width, label='Hybrid Weighted',
                      color=color_weighted, alpha=0.7, yerr=weighted_stds,
                      capsize=4, error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
        bars3 = ax.bar(x + width, enhanced_means, width,
                      label='Enhanced Prompt', color=color_enhanced, alpha=0.7,
                      yerr=enhanced_stds,
                      capsize=4, error_kw={'linewidth': 1.5, 'elinewidth': 1.5})

        # Add value labels above error bars
        def add_value_labels(bars, means, stds):
            for bar, mean, std in zip(bars, means, stds):
                height = mean + std
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        add_value_labels(bars1, baseline_means, baseline_stds)
        add_value_labels(bars2, weighted_means, weighted_stds)
        add_value_labels(bars3, enhanced_means, enhanced_stds)

        # Customize
        ax.set_xlabel('Dataset', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([task_labels[t] for t in tasks], rotation=15, ha='right')
        if idx == len(metrics) - 1:
            ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add padding
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.15)

    plt.suptitle('Multi-Metric Performance Comparison (Averaged over 3 runs of 256 samples)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-metric plot to {output_path}")

    return fig, axes


def print_summary_table(baseline_avg, hybrid_weighted_avg, enhanced_prompt_avg, metric='METEOR'):
    """Print a summary table of results."""
    tasks = ['protein_function_OOD', 'general_function_OOD',
             'domain_motif_OOD', 'catalytic_activity_OOD']

    task_labels = {
        'protein_function_OOD': 'Protein Function',
        'general_function_OOD': 'General Function',
        'domain_motif_OOD': 'Domain Motif',
        'catalytic_activity_OOD': 'Catalytic Activity'
    }

    print(f"\n{'='*90}")
    print(f"{metric} Score Comparison (Averaged over 3 runs of 256 samples)")
    print(f"{'='*90}")
    print(f"{'Dataset':<25} {'Baseline':<15} {'Hybrid Weighted':<20} {'Enhanced Prompt':<20}")
    print(f"{'-'*90}")

    for task in tasks:
        baseline_score = baseline_avg[task][metric]
        weighted_score = hybrid_weighted_avg[task][metric]
        enhanced_score = enhanced_prompt_avg[task][metric]

        print(f"{task_labels[task]:<25} {baseline_score:>6.2f} {' '*8} "
              f"{weighted_score:>6.2f} {' '*13} {enhanced_score:>6.2f}")

    print(f"{'='*90}\n")

    # Print improvement percentages
    print(f"Improvement over Baseline (%):")
    print(f"{'-'*90}")
    print(f"{'Dataset':<25} {'Hybrid Weighted':<20} {'Enhanced Prompt':<20}")
    print(f"{'-'*90}")

    for task in tasks:
        baseline_score = baseline_avg[task][metric]
        weighted_improvement = ((hybrid_weighted_avg[task][metric] - baseline_score) / baseline_score) * 100
        enhanced_improvement = ((enhanced_prompt_avg[task][metric] - baseline_score) / baseline_score) * 100

        print(f"{task_labels[task]:<25} {weighted_improvement:>6.2f}% {' '*13} {enhanced_improvement:>6.2f}%")

    print(f"{'='*90}\n")


def main():
    # Define file paths
    project_root = Path(__file__).parent.parent.parent

    baseline_file = project_root / 'gemini_evaluation_256_results.txt'
    hybrid_weighted_file = project_root / 'structural_retrieval/results/hybrid_weighted_sim_alpha0.70_256_rapm_results.txt'
    enhanced_prompt_file = project_root / 'structural_retrieval/results/enhanced_prompt_alpha0.70_rapm_results.txt'

    # Parse results
    print("Parsing baseline results...")
    baseline_results = parse_results_file(baseline_file)
    baseline_avg = compute_averages(baseline_results)

    print("Parsing hybrid weighted similarity results...")
    hybrid_weighted_results = parse_results_file(hybrid_weighted_file)
    hybrid_weighted_avg = compute_averages(hybrid_weighted_results)

    print("Parsing enhanced prompt results...")
    enhanced_prompt_results = parse_results_file(enhanced_prompt_file)
    enhanced_prompt_avg = compute_averages(enhanced_prompt_results)

    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'report/figs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Single metric comparison (Meta-BLEU-2)
    create_comparison_plot(
        baseline_results, hybrid_weighted_results, enhanced_prompt_results,
        metric='Meta-BLEU-2',
        output_path=output_dir / 'prompt_meta_bleu2_comparison.png'
    )

    # Single metric comparison (Meta-BLEU-4)
    create_comparison_plot(
        baseline_results, hybrid_weighted_results, enhanced_prompt_results,
        metric='Meta-BLEU-4',
        output_path=output_dir / 'prompt_meta_bleu4_comparison.png'
    )

    # Multi-metric comparison (Meta-BLEU-2 and Meta-BLEU-4)
    create_multi_metric_comparison(
        baseline_results, hybrid_weighted_results, enhanced_prompt_results,
        metrics=['Meta-BLEU-2', 'Meta-BLEU-4'],
        output_path=output_dir / 'prompt_meta_bleu_comparison.png'
    )

    # Print summary tables
    print_summary_table(baseline_avg, hybrid_weighted_avg, enhanced_prompt_avg, metric='Meta-BLEU-2')
    print_summary_table(baseline_avg, hybrid_weighted_avg, enhanced_prompt_avg, metric='Meta-BLEU-4')

    print("Done! Generated visualizations in structural_retrieval/visualizations/")


if __name__ == '__main__':
    main()
