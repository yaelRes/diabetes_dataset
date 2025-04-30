import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_algorithm_comparison(algorithms, silhouette_scores, output_dir="output"):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, silhouette_scores, color=['blue', 'red', 'green', 'purple'])
    plt.xlabel('Clustering Algorithm')
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of Clustering Algorithms')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the scores as text above the bars
    for bar, score in zip(bars, silhouette_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom')

    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_best_clustering_result(X_data, labels, algorithm_name, silhouette_score, output_dir="output"):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_data[:, 0], X_data[:, 1], c=labels,
                          cmap='viridis', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Best Clustering Result: {algorithm_name} (Silhouette Score: {silhouette_score:.4f})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_clustering_result.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_distribution(df_with_clusters, column, output_dir="output"):
    plt.figure(figsize=(12, 6))
    for cluster in np.unique(df_with_clusters['cluster']):
        sns.kdeplot(df_with_clusters[df_with_clusters['cluster'] == cluster][column],
                   label=f'Cluster {cluster}')

    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'cluster_distribution_{column}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_categorical_proportions(df_with_clusters, column, output_dir="output"):
    proportions = df_with_clusters.groupby('cluster')[column].value_counts(normalize=True).unstack().fillna(0)

    proportions.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Proportion of {column} Categories by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title=column)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_proportions_{column}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_methods_comparison(all_methods, output_dir="output"):
    plt.figure(figsize=(12, 6))
    methods = list(all_methods.keys())
    scores = list(all_methods.values())
    bars = plt.bar(methods, scores, color=['blue', 'red', 'green', 'purple', 'orange'])
    plt.xlabel('Method')
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of All Clustering Methods')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the scores as text above the bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_methods_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_contingency_heatmap(contingency, labels1_name, labels2_name, ari, ami, output_dir="output"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in range(contingency.shape[1])],
                yticklabels=[f'Cluster {i}' for i in range(contingency.shape[0])])
    plt.xlabel(f'{labels2_name} Clusters')
    plt.ylabel(f'{labels1_name} Clusters')
    plt.title(f'Cluster Comparison: {labels1_name} vs {labels2_name}\nARI: {ari:.4f}, AMI: {ami:.4f}')
    plt.tight_layout()

    comparison_name = f"{labels1_name}_vs_{labels2_name}".replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f'cluster_comparison_{comparison_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()