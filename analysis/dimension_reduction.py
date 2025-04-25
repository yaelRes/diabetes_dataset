"""
Dimensionality reduction analysis for diabetes clustering.
"""

import os
import numpy as np
import logging
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

from utils.caching import cache_result
from visualization.dimension_reduction import plot_pca_explained_variance, plot_dimension_reduction_comparison
from config import UMAP_CONFIG


@cache_result()
def perform_pca_analysis(X_processed, output_dir="pics"):
    """Perform PCA analysis and return optimal number of components.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing PCA analysis results
    """
    logging.info("Performing PCA analysis...")
    os.makedirs(output_dir, exist_ok=True)

    # Analyze all components first
    pca_full = PCA()
    pca_full.fit(X_processed)
    explained_variance = pca_full.explained_variance_ratio_

    # Plot variance explained
    plot_pca_explained_variance(explained_variance, output_dir)

    # Find number of components for 95% explained variance
    n_components_95 = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
    logging.info(f"Number of components needed for 95% variance: {n_components_95}")

    # Create 2D projections for visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_processed)

    return {
        'explained_variance': explained_variance,
        'n_components_95': n_components_95,
        'X_pca_2d': X_pca_2d,
        'pca_2d': pca_2d
    }


@cache_result()
def create_dimension_reduction_visualizations(X_processed, output_dir="pics"):
    """Create visualizations for different dimension reduction techniques.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing dimension reduction results
    """
    logging.info("Creating dimension reduction visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # PCA (2D)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_processed)
    logging.info(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_}")

    # t-SNE
    logging.info("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_processed)

    # UMAP
    logging.info("Performing UMAP...")
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_processed)

    # Visualize the 2D projections
    plot_dimension_reduction_comparison(
        X_pca_2d,
        X_tsne,
        X_umap,
        pca_2d.explained_variance_ratio_,
        output_dir
    )

    return {
        'X_pca_2d': X_pca_2d,
        'X_tsne': X_tsne,
        'X_umap': X_umap
    }


@cache_result()
def evaluate_umap_parameters(X_processed, n_components, n_clusters, n_neighbors, min_dist, best_algorithm):
    """
    Evaluate a single combination of UMAP parameters.
    This function is cached so each parameter combination only runs once.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        n_components (int): Number of UMAP components
        n_clusters (int): Number of clusters
        n_neighbors (int): UMAP n_neighbors parameter
        min_dist (float): UMAP min_dist parameter
        best_algorithm (str): The clustering algorithm to use

    Returns:
        dict: Dictionary containing evaluation results
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    # Create UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_processed)

    # Apply the best clustering algorithm to the UMAP embedding
    if best_algorithm == 'K-means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    elif best_algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif best_algorithm == 'DBSCAN':
        # For DBSCAN, we'd need to adjust eps for each UMAP embedding
        nn = NearestNeighbors(n_neighbors=min(5, len(X_umap) - 1))
        nn.fit(X_umap)
        distances, _ = nn.kneighbors(X_umap)
        distances = np.sort(distances[:, -1])
        knee_idx = np.argmax(np.diff(np.diff(distances))) + 1
        eps_value = distances[knee_idx]
        model = DBSCAN(eps=eps_value, min_samples=5)
    else:  # GMM
        model = GaussianMixture(n_components=n_clusters, random_state=42)

    umap_labels = model.fit_predict(X_umap)

    # Calculate silhouette score
    if len(np.unique(umap_labels)) > 1:  # Only calculate if there are multiple clusters
        # For n_components > 3, calculate silhouette directly
        score = silhouette_score(X_umap, umap_labels)
    else:
        score = 0

    return {
        'n_components': n_components,
        'n_clusters': n_clusters,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'score': score,
        'embedding': X_umap,
        'labels': umap_labels
    }


@cache_result()
def optimize_umap_parameters(X_processed, best_algorithm, output_dir="pics"):
    """Optimize UMAP parameters for visualization.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        best_algorithm (str): Best clustering algorithm from previous analysis
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing UMAP optimization results
    """
    logging.info("Optimizing UMAP parameters for visualization...")
    os.makedirs(output_dir, exist_ok=True)

    # Define UMAP parameter ranges from config
    n_neighbors_options = UMAP_CONFIG["n_neighbors_options"]
    min_dist_options = UMAP_CONFIG["min_dist_options"]
    n_components_options = UMAP_CONFIG["n_components_options"]
    n_clusters_options = range(2, 6)  # Number of clusters to try

    umap_results = []

    # Find optimal UMAP parameters with different component counts
    logging.info("Starting UMAP grid search over n_neighbors, min_dist, n_components, and n_clusters...")
    logging.info(f"Testing n_neighbors options: {n_neighbors_options}")
    logging.info(f"Testing min_dist options: {min_dist_options}")
    logging.info(f"Testing n_components options: {list(n_components_options)}")
    logging.info(f"Testing n_clusters options: {list(n_clusters_options)}")

    # Create a progress counter
    total_iterations = len(n_neighbors_options) * len(min_dist_options) * len(n_components_options) * len(
        n_clusters_options)
    progress_count = 0

    for n_components in n_components_options:
        for n_clusters in n_clusters_options:
            for n_neighbors in n_neighbors_options:
                for min_dist in min_dist_options:
                    progress_count += 1
                    logging.info(f"Progress: {progress_count}/{total_iterations} "
                                 f"(n_components={n_components}, n_clusters={n_clusters}, "
                                 f"n_neighbors={n_neighbors}, min_dist={min_dist:.2f})")

                    # Use the cached function to evaluate this parameter combination
                    result = evaluate_umap_parameters(
                        X_processed,
                        n_components,
                        n_clusters,
                        n_neighbors,
                        min_dist,
                        best_algorithm
                    )

                    umap_results.append(result)

                    logging.info(
                        f"UMAP(n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}) + "
                        f"Clusters(k={n_clusters}): Silhouette Score: {result['score']:.4f}")

    # Find the best UMAP parameters
    best_umap_result = max(umap_results, key=lambda x: x['score'])
    best_n_components = best_umap_result['n_components']
    best_n_clusters = best_umap_result['n_clusters']
    best_n_neighbors = best_umap_result['n_neighbors']
    best_min_dist = best_umap_result['min_dist']
    best_umap_score = best_umap_result['score']
    best_umap_embedding = best_umap_result['embedding']
    best_umap_labels = best_umap_result['labels']

    logging.info(
        f"Best UMAP parameters: n_components={best_n_components}, n_clusters={best_n_clusters}, "
        f"n_neighbors={best_n_neighbors}, min_dist={best_min_dist}")
    logging.info(f"Best UMAP silhouette score: {best_umap_score:.4f}")

    # Create a DataFrame from the results
    umap_df = pd.DataFrame([{
        'n_components': r['n_components'],
        'n_clusters': r['n_clusters'],
        'n_neighbors': r['n_neighbors'],
        'min_dist': r['min_dist'],
        'score': r['score']
    } for r in umap_results])

    return {
        'umap_results': umap_results,
        'umap_df': umap_df,
        'best_n_components': best_n_components,
        'best_n_clusters': best_n_clusters,
        'best_n_neighbors': best_n_neighbors,
        'best_min_dist': best_min_dist,
        'best_umap_score': best_umap_score,
        'best_umap_embedding': best_umap_embedding,
        'best_umap_labels': best_umap_labels
    }