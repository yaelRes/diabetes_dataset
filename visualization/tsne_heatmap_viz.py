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
    """Plot heatmap of t-SNE parameters (perplexity vs n_clusters).
    
    Args:
        tsne_df (pandas.DataFrame): DataFrame with t-SNE optimization results
        best_n_components (int): Optimal n_components value
        best_perplexity (float): Optimal perplexity value
        best_n_clusters (int): Optimal n_clusters value
        best_score (float): Best silhouette score achieved
        output_dir (str): Directory to save output files
    """
    # Filter data for the best n_components
    filtered_df = tsne_df[tsne_df['n_components'] == best_n_components]
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.pivot_table(
        index='perplexity',
        columns='n_clusters',
        values='score'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})
    
    # Highlight optimal parameters if they exist in the heatmap
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
    """Plot heatmap of t-SNE parameters (n_components vs perplexity).
    
    Args:
        tsne_df (pandas.DataFrame): DataFrame with t-SNE optimization results
        best_n_components (int): Optimal n_components value
        best_perplexity (float): Optimal perplexity value
        best_n_clusters (int): Optimal n_clusters value
        best_score (float): Best silhouette score achieved
        output_dir (str): Directory to save output files
    """
    # Filter data for the best n_clusters
    filtered_df = tsne_df[tsne_df['n_clusters'] == best_n_clusters]
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.pivot_table(
        index='n_components',
        columns='perplexity',
        values='score'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})
    
    # Highlight optimal parameters if they exist in the heatmap
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
    """Plot heatmap of silhouette scores for different algorithms and parameters.
    
    Args:
        clustering_result (dict): Dictionary containing clustering algorithm results
        output_dir (str): Directory to save output files
    """
    # Extract silhouette scores and algorithm names
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
    """Plot analysis of diabetes markers by demographic categories.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        demographic_col (str): Demographic column name (e.g., 'ethnicity', 'gender', 'age_group')
        value_col (str): Value column to analyze (e.g., 'glucose', 'hba1c')
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate means and standard errors
    means = df_with_clusters.groupby(demographic_col)[value_col].mean()
    errors = df_with_clusters.groupby(demographic_col)[value_col].sem()
    
    # Create bar chart
    bars = plt.bar(means.index, means.values, yerr=errors, capsize=10, 
                   alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.xlabel(demographic_col.capitalize())
    plt.ylabel(f'Average {value_col}')
    plt.title(f'Average {value_col} by {demographic_col.capitalize()}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45 if len(means.index) > 4 else 0)
    
    # Add the values as text above the bars
    for bar, mean in zip(bars, means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.05 * max(means.values),
                 f'{mean:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{value_col}_by_{demographic_col}.png'), dpi=300, bbox_inches='tight')
    plt.close()
