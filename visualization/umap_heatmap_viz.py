"""
Additional visualization functions for UMAP parameter optimization and correlation analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_umap_parameter_heatmap(umap_df, best_n_neighbors, best_min_dist, best_score, output_dir="output"):
    """Plot heatmap of UMAP parameters (n_neighbors vs min_dist).
    
    Args:
        umap_df (pandas.DataFrame): DataFrame with UMAP optimization results
        best_n_neighbors (int): Optimal n_neighbors value
        best_min_dist (float): Optimal min_dist value
        best_score (float): Best silhouette score achieved
        output_dir (str): Directory to save output files
    """
    # Filter data for n_components and n_clusters matching the best ones
    best_n_components = umap_df.loc[umap_df['score'] == umap_df['score'].max(), 'n_components'].iloc[0]
    best_n_clusters = umap_df.loc[umap_df['score'] == umap_df['score'].max(), 'n_clusters'].iloc[0]
    
    # Filter data for the best n_components and n_clusters
    filtered_df = umap_df[(umap_df['n_components'] == best_n_components) & 
                          (umap_df['n_clusters'] == best_n_clusters)]
    
    # Create pivot table for heatmap
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
    """Plot heatmap of UMAP parameters (n_components vs n_clusters).
    
    Args:
        umap_df (pandas.DataFrame): DataFrame with UMAP optimization results
        best_n_components (int): Optimal n_components value
        best_n_clusters (int): Optimal n_clusters value
        best_n_neighbors (int): Optimal n_neighbors value
        best_min_dist (float): Optimal min_dist value
        best_score (float): Best silhouette score achieved
        output_dir (str): Directory to save output files
    """
    # Filter data for the best n_neighbors and min_dist
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
    """Plot correlation matrix heatmap for numerical features.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset
        numerical_cols (list): List of numerical column names
        output_dir (str): Directory to save output files
    """
    # Calculate correlation matrix
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
    """Plot correlation heatmap between lifestyle factors and health markers.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset
        lifestyle_cols (list): List of lifestyle-related column names
        health_cols (list): List of health marker column names
        output_dir (str): Directory to save output files
    """
    # Calculate correlation matrix between lifestyle factors and health markers
    corr_matrix = df[lifestyle_cols + health_cols].corr()
    
    # Extract only the cross-correlations (lifestyle vs health)
    cross_corr = corr_matrix.loc[lifestyle_cols, health_cols]
    
    plt.figure(figsize=(12, len(lifestyle_cols) * 0.8))
    
    # Generate heatmap with correlation coefficients
    ax = sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5,
                    vmin=-1, vmax=1)
    
    plt.title('Correlation Between Lifestyle Factors and Health Markers', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lifestyle_health_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_boxplots_by_cluster(df_with_clusters, numerical_cols, output_dir="output"):
    """Plot boxplots of numerical features grouped by cluster.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        numerical_cols (list): List of numerical column names
        output_dir (str): Directory to save output files
    """
    n_cols = 3  # Number of columns in the plot grid
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
    """Create pairplot for key diabetes indicators.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        indicator_cols (list): List of key diabetes indicator columns
        output_dir (str): Directory to save output files
    """
    # Create a subset with just the indicators and cluster
    subset = df_with_clusters[indicator_cols + ['cluster']].copy()
    
    # Convert cluster to categorical for better coloring
    subset['cluster'] = subset['cluster'].astype('category')
    
    # Create pairplot
    g = sns.pairplot(subset, hue='cluster', palette='viridis', diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 's': 30}, height=2.5)
    
    g.fig.suptitle('Relationships Between Key Diabetes Indicators', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_indicators_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_risk_assessment(df_with_clusters, risk_col, health_col, output_dir="output"):
    """Create a risk assessment plot.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        risk_col (str): Column name for diabetes risk score
        health_col (str): Column name for metabolic health score
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with points colored by cluster
    scatter = plt.scatter(df_with_clusters[risk_col], df_with_clusters[health_col], 
                         c=df_with_clusters['cluster'], cmap='viridis', alpha=0.7, s=50)
    
    # Add cluster labels to centroids
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
