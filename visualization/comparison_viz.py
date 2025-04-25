"""
Visualization functions for train/test clustering comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_train_test_comparison(summary_df, output_dir="output"):
    """Plot comparison of train vs test silhouette scores.

    Args:
        summary_df (pandas.DataFrame): DataFrame with train and test scores
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(12, 8))

    # Create a bar chart for train and test scores
    bar_width = 0.35
    index = np.arange(len(summary_df))

    plt.bar(index, summary_df['Train Silhouette'], bar_width, label='Training', color='blue', alpha=0.7)
    plt.bar(index + bar_width, summary_df['Test Silhouette'], bar_width, label='Test', color='red', alpha=0.7)

    plt.xlabel('Feature Set')
    plt.ylabel('Silhouette Score')
    plt.title('Train vs Test Silhouette Scores by Feature Set')
    plt.xticks(index + bar_width / 2, summary_df['Feature Set'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'train_test_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_differences(summary_df, output_dir="output"):
    """Plot the differences between train and test scores.

    Args:
        summary_df (pandas.DataFrame): DataFrame with train and test scores
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(12, 6))

    # Sort by difference
    sorted_df = summary_df.sort_values('Difference')

    # Create a horizontal bar chart for differences
    bars = plt.barh(sorted_df['Feature Set'], sorted_df['Difference'])

    # Color bars based on positive/negative difference
    for i, bar in enumerate(bars):
        if sorted_df['Difference'].iloc[i] < 0:
            bar.set_color('red')  # Negative difference (test worse than train)
        else:
            bar.set_color('green')  # Positive difference (test better than train)

    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Test Score - Train Score')
    plt.ylabel('Feature Set')
    plt.title('Difference Between Test and Training Silhouette Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add the actual values as text next to the bars
    for i, v in enumerate(sorted_df['Difference']):
        plt.text(v + (0.01 if v >= 0 else -0.01),
                 i,
                 f"{v:.4f}",
                 va='center',
                 ha='left' if v >= 0 else 'right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_differences.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_stability_metrics(summary_df, output_dir="output"):
    """Plot stability metrics for each feature set.

    Args:
        summary_df (pandas.DataFrame): DataFrame with train and test scores
        output_dir (str): Directory to save output files
    """
    # Calculate stability metrics
    summary_df['Stability'] = 1 - abs(summary_df['Difference']) / summary_df['Train Silhouette']

    # Sort by stability (descending)
    sorted_df = summary_df.sort_values('Stability', ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_df['Feature Set'], sorted_df['Stability'], color='purple', alpha=0.7)

    plt.xlabel('Feature Set')
    plt.ylabel('Stability Score (higher is better)')
    plt.title('Clustering Stability by Feature Set')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the stability values as text above the bars
    for bar, score in zip(bars, sorted_df['Stability']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_train_test_visualizations(summary_df, output_dir="output"):
    """Create all train/test comparison visualizations.

    Args:
        summary_df (pandas.DataFrame): DataFrame with train and test scores
        output_dir (str): Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate all visualizations
    plot_train_test_comparison(summary_df, output_dir)
    plot_score_differences(summary_df, output_dir)
    plot_stability_metrics(summary_df, output_dir)

    # Create a scatterplot to visualize the relationship
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        summary_df['Train Silhouette'],
        summary_df['Test Silhouette'],
        s=100,  # point size
        alpha=0.7
    )

    # Add feature set labels to the points
    for i, txt in enumerate(summary_df['Feature Set']):
        plt.annotate(
            txt,
            (summary_df['Train Silhouette'].iloc[i], summary_df['Test Silhouette'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )

    # Add a diagonal line (y=x)
    min_val = min(summary_df['Train Silhouette'].min(), summary_df['Test Silhouette'].min()) - 0.05
    max_val = max(summary_df['Train Silhouette'].max(), summary_df['Test Silhouette'].max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    plt.xlabel('Training Silhouette Score')
    plt.ylabel('Test Silhouette Score')
    plt.title('Training vs Test Silhouette Scores')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set equal axes ranges for better visualization
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_test_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()