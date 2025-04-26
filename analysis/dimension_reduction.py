"""
Dimensionality reduction analysis for diabetes clustering.
"""

import os
import numpy as np
import logging
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from utils.caching import cache_result
from visualization.dimension_reduction import plot_pca_explained_variance, plot_dimension_reduction_comparison
from config import UMAP_CONFIG, TSNE_CONFIG
from visualization.tsne_heatmap_viz import plot_tsne_components_perplexity_heatmap, plot_tsne_parameter_heatmap
from visualization.umap_heatmap_viz import plot_umap_parameter_heatmap, plot_umap_components_clusters_heatmap


@cache_result()
def perform_pca_analysis(X_processed, output_dir="output"):
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
def evaluate_tsne_parameters(X_processed, n_components, perplexity, best_algorithm, n_clusters):
    """
    Evaluate a single combination of t-SNE parameters.
    This function is cached so each parameter combination only runs once.
    """
    try:
        logging.info(
            f"Evaluating t-SNE parameters: n_components={n_components}, perplexity={perplexity}, n_clusters={n_clusters}")

        # Create t-SNE embedding
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
        X_tsne = tsne.fit_transform(X_processed)

        # Apply the best clustering algorithm to the t-SNE embedding
        if best_algorithm == 'K-means':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        elif best_algorithm == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif best_algorithm == 'DBSCAN':
            # For DBSCAN, we'd need to adjust eps for each t-SNE embedding
            nn = NearestNeighbors(n_neighbors=min(5, len(X_tsne) - 1))
            nn.fit(X_tsne)
            distances, _ = nn.kneighbors(X_tsne)
            distances = np.sort(distances[:, -1])
            knee_idx = np.argmax(np.diff(np.diff(distances))) + 1
            eps_value = distances[knee_idx]
            model = DBSCAN(eps=eps_value, min_samples=5)
        else:  # GMM
            model = GaussianMixture(n_components=n_clusters, random_state=42)

        tsne_labels = model.fit_predict(X_tsne)

        # Calculate silhouette score
        if len(np.unique(tsne_labels)) > 1:  # Only calculate if there are multiple clusters
            score = silhouette_score(X_tsne, tsne_labels)
        else:
            score = 0
            logging.warning(f"Only one cluster found for t-SNE parameters: n_components={n_components}, "
                            f"perplexity={perplexity}, n_clusters={n_clusters}")

        logging.info(f"Successfully evaluated t-SNE parameters with score: {score:.4f}")

        return {
            'n_components': n_components,
            'perplexity': perplexity,
            'n_clusters': n_clusters,
            'score': score,
            'embedding': X_tsne,
            'labels': tsne_labels
        }
    except Exception as e:
        logging.error(f"Error in evaluate_tsne_parameters: {e}")
        raise  # Re-raise the exception to be caught by the calling function


@cache_result()
def evaluate_tsne_iteration(X_processed, n_components, perplexity, best_algorithm, n_clusters):
    """Evaluate a single t-SNE parameter combination.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        n_components (int): Number of dimensions for t-SNE
        perplexity (float): Perplexity parameter for t-SNE
        best_algorithm (str): Best clustering algorithm from previous analysis
        n_clusters (int): Number of clusters

    Returns:
        dict: Results for this parameter combination
    """
    logging.info(f"Evaluating t-SNE with n_components={n_components}, perplexity={perplexity}, "
                 f"n_clusters={n_clusters}")

    try:
        # Use the cached function to evaluate this parameter combination
        result = evaluate_tsne_parameters(
            X_processed,
            n_components,
            perplexity,
            best_algorithm,
            n_clusters
        )

        logging.info(
            f"t-SNE(n_components={n_components}, perplexity={perplexity}) + "
            f"Clusters(k={n_clusters}): Silhouette Score: {result['score']:.4f}")

        return result
    except Exception as e:
        logging.error(f"Error evaluating t-SNE parameters (n_components={n_components}, "
                      f"perplexity={perplexity}, n_clusters={n_clusters}): {e}")
        return None


@cache_result()
def optimize_tsne_parameters(X_processed, best_algorithm, output_dir="output"):
    """Optimize t-SNE parameters for visualization.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        best_algorithm (str): Best clustering algorithm from previous analysis
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing t-SNE optimization results
    """
    logging.info("Optimizing t-SNE parameters for visualization...")
    os.makedirs(output_dir, exist_ok=True)

    # Define t-SNE parameter ranges
    # Note: t-SNE typically works best when perplexity is between 5 and 50
    # and is often set to be roughly the square root of the number of samples
    try:
        n_components_options = TSNE_CONFIG["n_components_options"]
        perplexity_options = TSNE_CONFIG["perplexity_options"]
        n_clusters_options = TSNE_CONFIG["n_clusters_options"]
    except KeyError as e:
        logging.error(f"Missing key in TSNE_CONFIG: {e}")
        # Provide default values
        n_components_options = [2, 3]
        perplexity_options = [5, 15, 30, 50]
        n_clusters_options = [2, 3, 4, 5]
        logging.info(f"Using default t-SNE parameters: n_components_options={n_components_options}, "
                     f"perplexity_options={perplexity_options}, n_clusters_options={n_clusters_options}")

    tsne_results = []

    # Find optimal t-SNE parameters
    logging.info("Starting t-SNE grid search over n_components, perplexity, and n_clusters...")
    logging.info(f"Testing n_components options: {list(n_components_options)}")
    logging.info(f"Testing perplexity options: {perplexity_options}")
    logging.info(f"Testing n_clusters options: {list(n_clusters_options)}")

    # Create a progress counter
    total_iterations = len(n_components_options) * len(perplexity_options) * len(n_clusters_options)
    progress_count = 0

    try:
        for n_components in n_components_options:
            for perplexity in perplexity_options:
                for n_clusters in n_clusters_options:
                    progress_count += 1
                    logging.info(f"Progress: {progress_count}/{total_iterations} "
                                 f"(n_components={n_components}, perplexity={perplexity}, "
                                 f"n_clusters={n_clusters})")

                    # Use the cached function for each iteration
                    result = evaluate_tsne_iteration(
                        X_processed,
                        n_components,
                        perplexity,
                        best_algorithm,
                        n_clusters
                    )

                    if result:
                        tsne_results.append(result)
    except Exception as e:
        logging.error(f"Error in t-SNE optimization loop: {e}")

    # Check if we have any results
    if not tsne_results:
        logging.warning("No t-SNE results were collected. Using default parameters.")
        # Create default t-SNE embedding with standard parameters
        try:
            logging.info("Creating fallback t-SNE embedding with default parameters")
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            X_tsne = tsne.fit_transform(X_processed)

            # Apply clustering with default n_clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(X_tsne)

            # Return default values
            return {
                'tsne_results': [],
                'tsne_df': pd.DataFrame(),
                'best_n_components': 2,
                'best_perplexity': 30,
                'best_n_clusters': 3,
                'best_tsne_score': 0.0,
                'best_tsne_embedding': X_tsne,
                'best_tsne_labels': labels
            }
        except Exception as e:
            logging.error(f"Error creating fallback t-SNE embedding: {e}")
            # Return minimal default values if even the fallback fails
            return {
                'tsne_results': [],
                'tsne_df': pd.DataFrame(),
                'best_n_components': 2,
                'best_perplexity': 30,
                'best_n_clusters': 3,
                'best_tsne_score': 0.0,
                'best_tsne_embedding': None,
                'best_tsne_labels': None
            }

    # Find the best t-SNE parameters
    best_tsne_result = max(tsne_results, key=lambda x: x['score'])
    best_n_components = best_tsne_result['n_components']
    best_perplexity = best_tsne_result['perplexity']
    best_n_clusters = best_tsne_result['n_clusters']
    best_tsne_score = best_tsne_result['score']
    best_tsne_embedding = best_tsne_result['embedding']
    best_tsne_labels = best_tsne_result['labels']

    logging.info(
        f"Best t-SNE parameters: n_components={best_n_components}, "
        f"perplexity={best_perplexity}, n_clusters={best_n_clusters}")
    logging.info(f"Best t-SNE silhouette score: {best_tsne_score:.4f}")

    # Create a DataFrame from the results
    tsne_df = pd.DataFrame([{
        'n_components': r['n_components'],
        'perplexity': r['perplexity'],
        'n_clusters': r['n_clusters'],
        'score': r['score']
    } for r in tsne_results])
    plot_tsne_parameter_heatmap(
        tsne_df,
        best_n_components,
        best_perplexity,
        best_n_clusters,
        best_tsne_score,
        output_dir
    )

    plot_tsne_components_perplexity_heatmap(
        tsne_df,
        best_n_components,
        best_perplexity,
        best_n_clusters,
        best_tsne_score,
        output_dir
    )
    return {
        'tsne_results': tsne_results,
        'tsne_df': tsne_df,
        'best_n_components': best_n_components,
        'best_perplexity': best_perplexity,
        'best_n_clusters': best_n_clusters,
        'best_tsne_score': best_tsne_score,
        'best_tsne_embedding': best_tsne_embedding,
        'best_tsne_labels': best_tsne_labels
    }


@cache_result()
def create_dimension_reduction_visualizations(X_processed, best_tsne_params=None, best_umap_params=None, output_dir="output"):
    """Create visualizations for different dimension reduction techniques.

    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        best_tsne_params (dict, optional): Best t-SNE parameters from optimization
        best_umap_params (dict, optional): Best UMAP parameters from optimization
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

    # t-SNE with optimized parameters if available
    logging.info("Performing t-SNE with optimized parameters if available...")
    # FIX: Check if best_tsne_params is a dictionary before calling get()
    if best_tsne_params is not None and isinstance(best_tsne_params, dict):
        n_components = best_tsne_params.get('best_n_components', 2)
        perplexity = best_tsne_params.get('best_perplexity', 30)
        logging.info(f"Using optimized t-SNE parameters: n_components={n_components}, perplexity={perplexity}")
    else:
        n_components = 2
        perplexity = 30
        logging.info(f"Using default t-SNE parameters: n_components={n_components}, perplexity={perplexity}")

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_processed)

    # UMAP with optimized parameters if available
    logging.info("Performing UMAP with optimized parameters if available...")
    # FIX: Check if best_umap_params is a dictionary before calling get()
    if best_umap_params is not None and isinstance(best_umap_params, dict):
        n_components = best_umap_params.get('best_n_components', 2)
        n_neighbors = best_umap_params.get('best_n_neighbors', 15)
        min_dist = best_umap_params.get('best_min_dist', 0.1)
        logging.info(f"Using optimized UMAP parameters: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    else:
        n_components = 2
        n_neighbors = 15
        min_dist = 0.1
        logging.info(f"Using default UMAP parameters: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_processed)

    # For visualization, we need to ensure we're using 2D projections
    # If optimized parameters resulted in higher dimensions, project down to 2D for visualization
    if X_tsne.shape[1] > 2:
        logging.info("Projecting optimized t-SNE dimensions to 2D for visualization...")
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne_2d = tsne_2d.fit_transform(X_processed)
    else:
        X_tsne_2d = X_tsne

    if X_umap.shape[1] > 2:
        logging.info("Projecting optimized UMAP dimensions to 2D for visualization...")
        reducer_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        X_umap_2d = reducer_2d.fit_transform(X_processed)
    else:
        X_umap_2d = X_umap

    # Visualize the 2D projections
    from visualization.dimension_reduction import plot_dimension_reduction_comparison
    plot_dimension_reduction_comparison(
        X_pca_2d,
        X_tsne_2d,
        X_umap_2d,
        pca_2d.explained_variance_ratio_,
        output_dir
    )

    return {
        'X_pca_2d': X_pca_2d,
        'X_tsne': X_tsne,
        'X_umap': X_umap,
        'X_tsne_2d': X_tsne_2d if 'X_tsne_2d' in locals() else X_tsne,
        'X_umap_2d': X_umap_2d if 'X_umap_2d' in locals() else X_umap
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
def optimize_umap_parameters(X_processed, best_algorithm, output_dir="output"):
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
    n_clusters_options = UMAP_CONFIG["n_clusters_options"]

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

    plot_umap_parameter_heatmap(
        umap_df,
        best_n_neighbors,
        best_min_dist,
        best_umap_score,
        output_dir
    )

    plot_umap_components_clusters_heatmap(
        umap_df,
        best_n_components,
        best_n_clusters,
        best_n_neighbors,
        best_min_dist,
        best_umap_score,
        output_dir
    )
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