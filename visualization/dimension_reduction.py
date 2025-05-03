import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pca_explained_variance(explained_variance, output_dir="output"):
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% explained variance')
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance by Components')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_dimension_reduction_comparison(x_pca_2d, x_tsne, x_umap, pca_explained_variance_ratio, output_dir="output"):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # PCA plot
    axes[0].scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], alpha=0.6)
    axes[0].set_title(
        f'PCA (2D)\nPC1: {pca_explained_variance_ratio[0]:.2%}, PC2: {pca_explained_variance_ratio[1]:.2%}')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # TSNE plot
    axes[1].scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.6)
    axes[1].set_title('TSNE (2D)')
    axes[1].set_xlabel('TSNE 1')
    axes[1].set_ylabel('TSNE 2')

    # UMAP plot
    axes[2].scatter(x_umap[:, 0], x_umap[:, 1], alpha=0.6)
    axes[2].set_title('UMAP (2D)')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')

    plt.suptitle("Dimensionality Reduction Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'dimension_reduction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_dbscan_kdistance_graph(distances, elbow_index, output_dir="output"):
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=elbow_index, color='r', linestyle='--', label=f'Elbow point: eps={distances[elbow_index]:.4f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-th nearest neighbor distance')
    plt.title('K-distance Graph for DBSCAN eps Parameter Selection')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dbscan_kdistance_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_silhouette_heatmap(results_df, optimal_n_component, optimal_k, title, filename=None):
    plt.figure(figsize=(12, 8))
    heatmap_data = results_df.pivot_table(
        index='n_component',
        columns='k',
        values='score'
    )

    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    if optimal_n_component in heatmap_data.index and optimal_k in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(optimal_n_component)
        optimal_col = list(heatmap_data.columns).index(optimal_k)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(title)
    plt.ylabel('PCA Components')
    plt.xlabel('Number of Clusters (k)')
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clusters(X_2d, labels, title, filename=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
