import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_train_test_comparison(summary_df, output_dir):
    if summary_df.empty:
        logging.info("ERROR: Summary DataFrame empty")
        return

    required_columns = ['feature set', 'train silhouette', 'test silhouette']
    if not all(col in summary_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in summary_df.columns]
        logging.info(f"WARNING: missing columns: {missing}")
        logging.info(f"df received columns are: {summary_df.columns.tolist()}")
        return

    plt.figure(figsize=(12, 6))

    df_sorted = summary_df.sort_values('Test Silhouette', ascending=False)

    feature_sets = df_sorted['Feature Set'].tolist()
    x = np.arange(len(feature_sets))
    bar_width = 0.35

    plt.bar(x - bar_width / 2, df_sorted['train silhouette'], bar_width, label='training', color='blue', alpha=0.7)
    plt.bar(x + bar_width / 2, df_sorted['test silhouette'], bar_width, label='test', color='green', alpha=0.7)

    plt.xlabel('feature set')
    plt.ylabel('silhouette score')
    plt.title('train and test Silhouette Scores by feature set')
    plt.xticks(x, feature_sets, rotation=45, ha='right')
    plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'train_test_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))

    df_sorted_diff = summary_df.sort_values('difference', key=abs, ascending=True)

    feature_sets_diff = df_sorted_diff['feature set'].tolist()
    diff_values = df_sorted_diff['difference'].tolist()

    # Create a bar chart with color based on difference sign
    colors = ['red' if x < 0 else 'green' for x in diff_values]
    plt.bar(feature_sets_diff, diff_values, color=colors, alpha=0.7)

    # Add labels
    plt.xlabel('feature Set')
    plt.ylabel('Difference (Test - Train)')
    plt.title('Stability of Feature Sets (Smaller Absolute Difference is Better)')
    plt.xticks(rotation=45, ha='right')

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'stability_comparison.png'), dpi=300)
    plt.close()


def create_comparative_visualizations(metrics_dfs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    silhouette_data = {}

    for feature_set, df in metrics_dfs.items():
        if 'Method' in df.columns and 'Silhouette Score' in df.columns:
            for _, row in df.iterrows():
                method = row['Method']
                score = row['Silhouette Score']

                if method not in silhouette_data:
                    silhouette_data[method] = {}

                silhouette_data[method][feature_set] = score

    for method, scores in silhouette_data.items():
        plt.figure(figsize=(12, 6))

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        feature_sets = [item[0] for item in sorted_items]
        score_values = [item[1] for item in sorted_items]

        plt.bar(feature_sets, score_values, color='skyblue', alpha=0.7)

        plt.xlabel('Feature Set')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Scores for {method} Across Feature Sets')
        plt.xticks(rotation=45, ha='right')

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        method_name = method.replace(' ', '_').replace('+', 'plus')
        plt.savefig(os.path.join(output_dir, f'comparison_{method_name}.png'), dpi=300)
        plt.close()


def create_train_test_visualizations(summary_df, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    required_columns = ['Feature Set', 'Train Silhouette', 'Test Silhouette', 'Difference']

    if summary_df is None or summary_df.empty or not all(col in summary_df.columns for col in required_columns):
        print(f"Warning: Summary DataFrame is missing required columns for visualization")
        print(f"Available columns: {summary_df.columns.tolist() if not summary_df.empty else '[]'}")
        print("Skipping train/test comparison visualizations")
        return

    plot_train_test_comparison(summary_df, output_dir)
    plot_score_differences(summary_df, output_dir)
    plot_stability_metrics(summary_df, output_dir)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        summary_df['Train Silhouette'],
        summary_df['Test Silhouette'],
        s=100,  # point size
        alpha=0.7
    )

    for i, txt in enumerate(summary_df['Feature Set']):
        plt.annotate(
            txt,
            (summary_df['Train Silhouette'].iloc[i], summary_df['Test Silhouette'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )

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


def plot_score_differences(summary_df, output_dir="output"):

    if summary_df is None or summary_df.empty or 'Difference' not in summary_df.columns:
        print("Skipping score_differences visualization due to missing 'Difference' column")
        return

    plt.figure(figsize=(12, 6))

    sorted_df = summary_df.sort_values('Difference')

    bars = plt.barh(sorted_df['Feature Set'], sorted_df['Difference'])

    for i, bar in enumerate(bars):
        if sorted_df['Difference'].iloc[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')

    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Test Score - Train Score')
    plt.ylabel('Feature Set')
    plt.title('Difference Between Test and Training Silhouette Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

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
    if summary_df is None or summary_df.empty or 'Difference' not in summary_df.columns or 'Train Silhouette' not in summary_df.columns:
        print("Skipping stability_metrics visualization due to missing required columns")
        return

    summary_df['Stability'] = 1 - abs(summary_df['Difference']) / summary_df['Train Silhouette']

    sorted_df = summary_df.sort_values('Stability', ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_df['Feature Set'], sorted_df['Stability'], color='purple', alpha=0.7)

    plt.xlabel('Feature Set')
    plt.ylabel('Stability Score (higher is better)')
    plt.title('Clustering Stability by Feature Set')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, score in zip(bars, sorted_df['Stability']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()



