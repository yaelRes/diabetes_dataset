"""
Additional visualization functions for UMAP parameter optimization and correlation analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_umap_parameter_heatmap(umap_df, best_n_neighbors, best_min_dist, best_score, output_dir="output"):
    best_n_components = umap_df.loc[umap_df['score'] == umap_df['score'].max(), 'n_components'].iloc[0]
    best_n_clusters = umap_df.loc[umap_df['score'] == umap_df['score'].max(), 'n_clusters'].iloc[0]

    filtered_df = umap_df[(umap_df['n_components'] == best_n_components) & 
                          (umap_df['n_clusters'] == best_n_clusters)]

    heatmap_data = filtered_df.pivot_table(
        index='n_neighbors',
        columns='min_dist',
        values='score'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})
    
    # Highlight optimal parameters if they exist in the heatmap
    if best_n_neighbors in heatmap_data.index and best_min_dist in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(best_n_neighbors)
        optimal_col = list(heatmap_data.columns).index(best_min_dist)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title(f'UMAP Silhouette Scores for Different n_neighbors and min_dist\n' +
             f'(n_components={best_n_components}, n_clusters={best_n_clusters}, ' +
             f'best score={best_score:.4f})')
    plt.ylabel('n_neighbors')
    plt.xlabel('min_dist')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'umap_neighbors_mindist_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_umap_components_clusters_heatmap(umap_df, best_n_components, best_n_clusters, 
                                         best_n_neighbors, best_min_dist, best_score, 
                                         output_dir="output"):

    filtered_df = umap_df[(umap_df['n_neighbors'] == best_n_neighbors) & 
                          (umap_df['min_dist'] == best_min_dist)]
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.pivot_table(
        index='n_components',
        columns='n_clusters',
        values='score'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})
    
    # Highlight optimal parameters if they exist in the heatmap
    if best_n_components in heatmap_data.index and best_n_clusters in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(best_n_components)
        optimal_col = list(heatmap_data.columns).index(best_n_clusters)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title(f'UMAP Silhouette Scores for Different n_components and n_clusters\n' +
             f'(n_neighbors={best_n_neighbors}, min_dist={best_min_dist:.2f}, ' +
             f'best score={best_score:.4f})')
    plt.ylabel('n_components')
    plt.xlabel('n_clusters')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'umap_components_clusters_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df, numerical_cols, output_dir="output"):
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap with correlation coefficients
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5,
                    vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lifestyle_health_correlation(df, lifestyle_cols, health_cols, output_dir="output"):

    corr_matrix = df[lifestyle_cols + health_cols].corr()

    cross_corr = corr_matrix.loc[lifestyle_cols, health_cols]
    
    plt.figure(figsize=(12, len(lifestyle_cols) * 0.8))

    ax = sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5,
                    vmin=-1, vmax=1)
    
    plt.title('Correlation Between Lifestyle Factors and Health Markers', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lifestyle_health_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_boxplots_by_cluster(df_with_clusters, numerical_cols, output_dir="output"):

    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  # Calculate number of rows needed
    
    plt.figure(figsize=(16, 4 * n_rows))
    
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x='cluster', y=col, data=df_with_clusters)
        plt.title(f'{col} by Cluster')
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'feature_boxplots_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pairplot_diabetes_indicators(df_with_clusters, indicator_cols, output_dir="output"):
    subset = df_with_clusters[indicator_cols + ['cluster']].copy()

    subset['cluster'] = subset['cluster'].astype('category')

    g = sns.pairplot(subset, hue='cluster', palette='viridis', diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 's': 30}, height=2.5)
    
    g.fig.suptitle('Relationships Between Key Diabetes Indicators', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_indicators_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_risk_assessment(df_with_clusters, risk_col, health_col, output_dir="output"):
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(df_with_clusters[risk_col], df_with_clusters[health_col], 
                         c=df_with_clusters['cluster'], cmap='viridis', alpha=0.7, s=50)

    for cluster in df_with_clusters['cluster'].unique():
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
        centroid_x = cluster_data[risk_col].mean()
        centroid_y = cluster_data[health_col].mean()
        plt.text(centroid_x, centroid_y, str(cluster), fontsize=16, fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Diabetes Risk Score')
    plt.ylabel('Metabolic Health Score')
    plt.title('Risk Assessment by Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'risk_assessment_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
