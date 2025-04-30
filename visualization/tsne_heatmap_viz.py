"""
Additional visualization functions for t-SNE parameter optimization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_tsne_parameter_heatmap(tsne_df, best_n_components, best_perplexity,
                                best_n_clusters, best_score, output_dir="output"):
    filtered_df = tsne_df[tsne_df['n_components'] == best_n_components]

    heatmap_data = filtered_df.pivot_table(
        index='perplexity',
        columns='n_clusters',
        values='score'
    )

    plt.figure(figsize=(12, 8))

    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    if best_perplexity in heatmap_data.index and best_n_clusters in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(best_perplexity)
        optimal_col = list(heatmap_data.columns).index(best_n_clusters)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(f't-SNE Silhouette Scores for Different Perplexity and n_clusters\n' +
              f'(n_components={best_n_components}, best score={best_score:.4f})')
    plt.ylabel('Perplexity')
    plt.xlabel('Number of Clusters')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'tsne_perplexity_clusters_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne_components_perplexity_heatmap(tsne_df, best_n_components, best_perplexity,
                                            best_n_clusters, best_score, output_dir="output"):
    filtered_df = tsne_df[tsne_df['n_clusters'] == best_n_clusters]

    heatmap_data = filtered_df.pivot_table(
        index='n_components',
        columns='perplexity',
        values='score'
    )

    plt.figure(figsize=(12, 8))

    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    if best_n_components in heatmap_data.index and best_perplexity in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(best_n_components)
        optimal_col = list(heatmap_data.columns).index(best_perplexity)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(f't-SNE Silhouette Scores for Different n_components and Perplexity\n' +
              f'(n_clusters={best_n_clusters}, best score={best_score:.4f})')
    plt.ylabel('n_components')
    plt.xlabel('Perplexity')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'tsne_components_perplexity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_algorithm_silhouette_heatmap(clustering_result, output_dir="output"):
    algorithms = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
    silhouette_scores = [
        clustering_result['kmeans_silhouette'],
        clustering_result['hierarchical_silhouette'],
        clustering_result['dbscan_silhouette'],
        clustering_result['gmm_silhouette']
    ]

    # Create DataFrame for the heatmap
    df = pd.DataFrame({'Algorithm': algorithms, 'Silhouette Score': silhouette_scores})

    # Find the best algorithm by silhouette score
    best_algo_index = np.argmax(silhouette_scores)
    best_algorithm = algorithms[best_algo_index]
    best_score = silhouette_scores[best_algo_index]

    # Format data for heatmap
    df_heatmap = df.set_index('Algorithm').T

    plt.figure(figsize=(12, 4))

    # Create heatmap
    ax = sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    # Highlight the best algorithm
    ax.add_patch(plt.Rectangle((best_algo_index, 0), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(f'Silhouette Score Comparison for Different Clustering Algorithms\n' +
              f'Best: {best_algorithm} (Score: {best_score:.4f})')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'algorithm_silhouette_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_demographic_analysis(df_with_clusters, demographic_col, value_col, output_dir="output"):
    plt.figure(figsize=(12, 6))

    means = df_with_clusters.groupby(demographic_col)[value_col].mean()
    errors = df_with_clusters.groupby(demographic_col)[value_col].sem()

    bars = plt.bar(means.index, means.values, yerr=errors, capsize=10,
                   alpha=0.7, color='skyblue', edgecolor='black')

    plt.xlabel(demographic_col.capitalize())
    plt.ylabel(f'Average {value_col}')
    plt.title(f'Average {value_col} by {demographic_col.capitalize()}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45 if len(means.index) > 4 else 0)

    for bar, mean in zip(bars, means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.05 * max(means.values),
                 f'{mean:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{value_col}_by_{demographic_col}.png'), dpi=300, bbox_inches='tight')
    plt.close()
