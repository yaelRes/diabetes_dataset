"""
Functions for evaluating clustering performance on test data.
"""

import os
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import umap.umap_ as umap

from visualization.clustering_viz import plot_best_clustering_result
from visualization.feature_viz import plot_top_features_pairplot


def evaluate_on_test_set(df_test, X_test_processed, clustering_result, pca_result,
                         umap_result, eval_result, preprocessor, numerical_cols,
                         categorical_cols, output_dir="output/test"):
    """
    Evaluate clustering performance on test set.

    Args:
        df_test (pandas.DataFrame): Test set dataframe
        X_test_processed (numpy.ndarray): Preprocessed test data matrix
        clustering_result (dict): Results from clustering algorithms comparison
        pca_result (dict): Results from PCA analysis
        umap_result (dict): Results from UMAP optimization
        eval_result (dict): Results from final evaluation
        preprocessor (ColumnTransformer): Preprocessor used to transform the data
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing test evaluation results
    """
    logging.info("Evaluating clustering performance on test set...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Apply PCA transformation to test data
    if 'pca_2d' in pca_result:
        X_test_pca_2d = pca_result['pca_2d'].transform(X_test_processed)
    else:
        # Create a new PCA if not available
        pca_2d = PCA(n_components=2)
        X_test_pca_2d = pca_2d.fit_transform(X_test_processed)

    # 2. Apply the best algorithm
    best_algorithm = clustering_result['best_algorithm']
    optimal_k = clustering_result['best_algorithm_labels'].max() + 1

    # Apply PCA transformation with optimal components
    pca_optimal = PCA(n_components=clustering_result['X_pca_optimal'].shape[1])
    pca_optimal.fit(X_test_processed)
    X_test_pca_optimal = pca_optimal.transform(X_test_processed)

    logging.info(f"Applying best algorithm: {best_algorithm} with k={optimal_k}")

    if best_algorithm == 'K-means':
        # Apply K-means (use the pre-fitted model if possible)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans.fit(X_test_pca_optimal)  # Fit on test data
        test_labels = kmeans.labels_

    elif best_algorithm == 'Hierarchical':
        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        test_labels = hierarchical.fit_predict(X_test_pca_optimal)

    elif best_algorithm == 'DBSCAN':
        # Apply DBSCAN (need to calculate eps for test set)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(10, len(X_test_pca_optimal) - 1))
        nn.fit(X_test_pca_optimal)
        distances, _ = nn.kneighbors(X_test_pca_optimal)
        distances = np.sort(distances[:, -1])
        elbow_idx = np.argmax(np.diff(np.diff(distances))) + 1
        eps_value = distances[elbow_idx]

        dbscan = DBSCAN(eps=eps_value, min_samples=5)
        test_labels = dbscan.fit_predict(X_test_pca_optimal)

    else:  # GMM
        # Apply GMM
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        test_labels = gmm.fit_predict(X_test_pca_optimal)

    # 3. Calculate silhouette score on test set
    if len(np.unique(test_labels)) > 1:
        test_silhouette = silhouette_score(X_test_pca_optimal, test_labels)
        logging.info(f"Test set silhouette score: {test_silhouette:.4f}")
    else:
        test_silhouette = 0
        logging.warning("Only one cluster found in test set, silhouette score set to 0")

    # 4. Visualize test set clusters
    plot_best_clustering_result(
        X_test_pca_2d,
        test_labels,
        f"{best_algorithm} (Test Set)",
        test_silhouette,
        output_dir
    )

    # 5. If UMAP was the best method, apply it to test set
    if eval_result['best_method'] == 'UMAP + Best Algorithm':
        logging.info("Applying best UMAP configuration to test set")

        # Create UMAP with best parameters
        reducer = umap.UMAP(
            n_neighbors=umap_result['best_n_neighbors'],
            min_dist=umap_result['best_min_dist'],
            n_components=umap_result['best_n_components'],
            random_state=42
        )

        X_test_umap = reducer.fit_transform(X_test_processed)

        # Apply clustering to UMAP result
        if best_algorithm == 'K-means':
            model = KMeans(n_clusters=umap_result['best_n_clusters'], random_state=42, n_init=10)
        elif best_algorithm == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=umap_result['best_n_clusters'])
        elif best_algorithm == 'DBSCAN':
            nn = NearestNeighbors(n_neighbors=min(5, len(X_test_umap) - 1))
            nn.fit(X_test_umap)
            distances, _ = nn.kneighbors(X_test_umap)
            distances = np.sort(distances[:, -1])
            knee_idx = np.argmax(np.diff(np.diff(distances))) + 1
            eps_value = distances[knee_idx]
            model = DBSCAN(eps=eps_value, min_samples=5)
        else:  # GMM
            model = GaussianMixture(n_components=umap_result['best_n_clusters'], random_state=42)

        test_umap_labels = model.fit_predict(X_test_umap)

        # Calculate silhouette score
        if len(np.unique(test_umap_labels)) > 1:
            test_umap_silhouette = silhouette_score(X_test_umap, test_umap_labels)
            logging.info(f"Test set UMAP silhouette score: {test_umap_silhouette:.4f}")
        else:
            test_umap_silhouette = 0
            logging.warning("Only one cluster found in UMAP test set, silhouette score set to 0")

        # Use UMAP labels if they're better
        if test_umap_silhouette > test_silhouette:
            test_labels = test_umap_labels
            test_silhouette = test_umap_silhouette
            logging.info("Using UMAP clustering results for test set (better silhouette score)")

    # 6. Analyze cluster characteristics on test set
    df_test_with_clusters = df_test.copy()
    df_test_with_clusters['cluster'] = test_labels

    # Create visualizations for top features
    feature_cols = numerical_cols + categorical_cols
    if len(feature_cols) >= 3:
        plot_top_features_pairplot(df_test_with_clusters, numerical_cols[:min(3, len(numerical_cols))], output_dir)

    # 7. Generate cluster profiles for test set
    test_cluster_stats = {}
    for col in numerical_cols:
        test_cluster_stats[col] = df_test_with_clusters.groupby('cluster')[col].agg(['mean', 'std', 'min', 'max'])

    # 8. Compare test and training cluster profiles
    train_test_comparison = {
        'test_silhouette': test_silhouette,
        'training_silhouette': clustering_result['best_algorithm_silhouette'],
        'difference': test_silhouette - clustering_result['best_algorithm_silhouette']
    }

    logging.info(f"Silhouette score: Training: {clustering_result['best_algorithm_silhouette']:.4f}, "
                 f"Test: {test_silhouette:.4f}, "
                 f"Difference: {train_test_comparison['difference']:.4f}")

    # Save test set with cluster labels
    df_test_with_clusters.to_csv(os.path.join(output_dir, 'test_set_with_clusters.csv'), index=False)

    return {
        'test_labels': test_labels,
        'test_silhouette': test_silhouette,
        'test_cluster_stats': test_cluster_stats,
        'train_test_comparison': train_test_comparison,
        'df_test_with_clusters': df_test_with_clusters
    }