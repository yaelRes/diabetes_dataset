"""
Clustering analysis for diabetes clustering.
"""

import os
import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from utils.caching import cache_result
from visualization.dimension_reduction import plot_silhouette_heatmap, plot_dbscan_kdistance_graph
from visualization.clustering_viz import (
    plot_algorithm_comparison, 
    plot_best_clustering_result,
    plot_cluster_distribution,
    plot_categorical_proportions,
    plot_all_methods_comparison
)


@cache_result()
def grid_search_clustering_parameters(X_processed, n_components_list, k_range, output_dir="output"):
    """Run grid search over PCA components and number of clusters.
    
    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        n_components_list (list): List of PCA component counts to try
        k_range (range): Range of cluster counts to try
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing grid search results
    """
    logging.info("Running grid search for optimal clustering parameters...")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables to track results
    max_silhouette_score = 0
    optimal_n_component = 0
    optimal_k = 0
    optimal_pca = None

    # Run grid search over PCA components and number of clusters
    results = []
    for n_component in n_components_list:
        if isinstance(n_component, float) and n_component < 1:
            # For percentage of variance explained
            pca = PCA(n_components=n_component)
        else:
            # For specific number of components
            pca = PCA(n_components=min(n_component, X_processed.shape[1] - 1))

        X_pca = pca.fit_transform(X_processed)

        if isinstance(n_component, float):
            actual_components = X_pca.shape[1]
            logging.info(f"Using {actual_components} components to preserve {n_component * 100:.0f}% of variance")
        else:
            actual_components = n_component
            logging.info(f"Using {actual_components} components")

        for k in k_range:
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(X_pca)

            # Calculate silhouette score
            score = silhouette_score(X_pca, labels)

            # Store results
            results.append({
                'n_component': actual_components if not isinstance(n_component, float) else n_component,
                'k': k,
                'score': score,
                'labels': labels,
                'pca_data': X_pca
            })

            # Track optimal parameters
            if score > max_silhouette_score:
                max_silhouette_score = score
                optimal_n_component = actual_components if not isinstance(n_component, float) else n_component
                optimal_k = k
                optimal_pca = X_pca

            logging.info(f"PCA n_component={n_component}, K={k}, Silhouette Score: {score:.4f}")

    # Create DataFrame for easier manipulation
    results_df = pd.DataFrame([(r['n_component'], r['k'], r['score']) for r in results],
                              columns=['n_component', 'k', 'score'])

    # Log optimal parameters
    logging.info(f"Optimal parameters: PCA n_component={optimal_n_component}, K={optimal_k}")
    logging.info(f"Best silhouette score: {max_silhouette_score:.4f}")

    # Create heatmap visualization
    plot_silhouette_heatmap(
        results_df,
        optimal_n_component,
        optimal_k,
        'Silhouette Scores for Different PCA Components and K-Means Clusters',
        os.path.join(output_dir, 'pca_kmeans_heatmap.png')
    )

    # Get the best result and its data
    best_result = [r for r in results if (
        r['n_component'] == optimal_n_component if not isinstance(optimal_n_component, float)
        else r['n_component'] == optimal_n_component) and r['k'] == optimal_k][0]

    best_pca_data = best_result['pca_data']
    best_labels = best_result['labels']

    return {
        'results': results,
        'results_df': results_df,
        'optimal_n_component': optimal_n_component,
        'optimal_k': optimal_k,
        'max_silhouette_score': max_silhouette_score,
        'best_pca_data': best_pca_data,
        'best_labels': best_labels
    }


@cache_result()
def compare_clustering_algorithms(X_processed, optimal_n_component, optimal_k, output_dir="output"):
    """Compare different clustering algorithms using the optimal parameters.
    
    Args:
        X_processed (numpy.ndarray): Preprocessed data matrix
        optimal_n_component (int): Optimal number of PCA components
        optimal_k (int): Optimal number of clusters
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing clustering algorithm comparison results
    """
    logging.info("Comparing different clustering algorithms...")
    os.makedirs(output_dir, exist_ok=True)

    # Use the optimal PCA n_components
    pca_optimal = PCA(n_components=optimal_n_component if not isinstance(optimal_n_component, float) else None)
    if isinstance(optimal_n_component, float):
        pca_optimal.set_params(n_components=optimal_n_component)

    X_pca_optimal = pca_optimal.fit_transform(X_processed)

    # 1. K-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca_optimal)
    kmeans_silhouette = silhouette_score(X_pca_optimal, kmeans_labels)
    logging.info(f"K-means Silhouette Score: {kmeans_silhouette:.4f}")

    # 2. Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_pca_optimal)
    hierarchical_silhouette = silhouette_score(X_pca_optimal, hierarchical_labels)
    logging.info(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.4f}")

    # 3. DBSCAN
    # Find eps using k-distance graph
    nn = NearestNeighbors(n_neighbors=min(10, len(X_pca_optimal) - 1))
    nn.fit(X_pca_optimal)
    distances, indices = nn.kneighbors(X_pca_optimal)
    distances = np.sort(distances[:, -1])

    # Find the elbow point
    knee_point = np.diff(np.diff(distances))
    elbow_index = np.argmax(knee_point) + 1
    eps_value = distances[elbow_index]
    logging.info(f"Selected DBSCAN eps value: {eps_value:.4f}")

    # Plot the k-distance graph to find the elbow
    plot_dbscan_kdistance_graph(distances, elbow_index, output_dir)

    # Run DBSCAN with the selected eps
    dbscan = DBSCAN(eps=eps_value, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca_optimal)

    # Handle the case if DBSCAN returns mostly noise (-1)
    if len(np.unique(dbscan_labels)) <= 1 or -1 in np.unique(dbscan_labels):
        # Adjust eps to get more clusters
        from config import CLUSTERING_CONFIG
        eps_factors = CLUSTERING_CONFIG["dbscan_eps_factors"]
        eps_attempts = [eps_value * factor for factor in eps_factors]
        best_eps = eps_value
        best_n_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
        best_noise_ratio = np.sum(dbscan_labels == -1) / len(dbscan_labels)

        for eps in eps_attempts:
            temp_dbscan = DBSCAN(eps=eps, min_samples=5)
            temp_labels = temp_dbscan.fit_predict(X_pca_optimal)
            n_clusters = len(np.unique(temp_labels[temp_labels != -1]))
            noise_ratio = np.sum(temp_labels == -1) / len(temp_labels)

            logging.info(f"DBSCAN with eps={eps:.4f}: {n_clusters} clusters, {noise_ratio:.2%} noise")

            # Better result has more clusters and less noise
            if n_clusters > best_n_clusters and noise_ratio < 0.5:
                best_eps = eps
                best_n_clusters = n_clusters
                best_noise_ratio = noise_ratio
                dbscan_labels = temp_labels

        logging.info(f"Selected better DBSCAN eps value: {best_eps:.4f}")

    # Calculate DBSCAN silhouette (if there are multiple clusters and not all noise)
    if len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels):
        dbscan_silhouette = silhouette_score(X_pca_optimal, dbscan_labels)
    elif len(np.unique(dbscan_labels)) > 1:
        # Calculate silhouette only on non-noise points
        non_noise = dbscan_labels != -1
        if np.sum(non_noise) > 1 and len(np.unique(dbscan_labels[non_noise])) > 1:
            dbscan_silhouette = silhouette_score(X_pca_optimal[non_noise], dbscan_labels[non_noise])
        else:
            dbscan_silhouette = 0
    else:
        dbscan_silhouette = 0
    logging.info(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")

    # 4. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca_optimal)
    gmm_silhouette = silhouette_score(X_pca_optimal, gmm_labels)
    logging.info(f"GMM Silhouette Score: {gmm_silhouette:.4f}")

    # Compare silhouette scores
    algorithms = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
    silhouette_scores = [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette, gmm_silhouette]

    # Plot comparison
    plot_algorithm_comparison(algorithms, silhouette_scores, output_dir)

    # Pick the best algorithm based on silhouette score
    best_algorithm_index = np.argmax(silhouette_scores)
    best_algorithm = algorithms[best_algorithm_index]
    best_algorithm_silhouette = silhouette_scores[best_algorithm_index]
    logging.info(f"Best clustering algorithm: {best_algorithm} with silhouette score {best_algorithm_silhouette:.4f}")

    # Get the labels from the best algorithm
    if best_algorithm == 'K-means':
        best_algorithm_labels = kmeans_labels
    elif best_algorithm == 'Hierarchical':
        best_algorithm_labels = hierarchical_labels
    elif best_algorithm == 'DBSCAN':
        best_algorithm_labels = dbscan_labels
    else:  # GMM
        best_algorithm_labels = gmm_labels

    # Visualize the best clustering result on the 2D PCA plot
    plot_best_clustering_result(
        X_pca_optimal[:, :2], 
        best_algorithm_labels, 
        best_algorithm, 
        best_algorithm_silhouette,
        output_dir
    )

    return {
        'X_pca_optimal': X_pca_optimal,
        'kmeans_labels': kmeans_labels,
        'kmeans_silhouette': kmeans_silhouette,
        'hierarchical_labels': hierarchical_labels,
        'hierarchical_silhouette': hierarchical_silhouette,
        'dbscan_labels': dbscan_labels,
        'dbscan_silhouette': dbscan_silhouette,
        'gmm_labels': gmm_labels,
        'gmm_silhouette': gmm_silhouette,
        'best_algorithm': best_algorithm,
        'best_algorithm_labels': best_algorithm_labels,
        'best_algorithm_silhouette': best_algorithm_silhouette
    }


@cache_result()
def analyze_cluster_characteristics(df, best_algorithm_labels, numerical_cols, categorical_cols, output_dir="output"):
    """Analyze and visualize characteristics of each cluster.
    
    Args:
        df (pandas.DataFrame): Original dataframe
        best_algorithm_labels (numpy.ndarray): Best clustering labels
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing cluster characteristics
    """
    logging.info("Analyzing cluster characteristics...")
    os.makedirs(output_dir, exist_ok=True)

    # Create a dataframe with the original data and cluster labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = best_algorithm_labels

    # Analyze cluster characteristics for numerical columns
    cluster_stats = {}
    for column in numerical_cols:
        cluster_stats[column] = df_with_clusters.groupby('cluster')[column].agg(['mean', 'std', 'min', 'max'])

        # Plot distribution by cluster
        plot_cluster_distribution(df_with_clusters, column, output_dir)

    # Display summary statistics for each cluster
    for column in numerical_cols:
        logging.info(f"\nCluster statistics for {column}:")
        logging.info(cluster_stats[column])

    # For categorical columns, calculate proportions in each cluster
    for column in categorical_cols:
        logging.info(f"\nCluster proportions for {column}:")
        proportions = df_with_clusters.groupby('cluster')[column].value_counts(normalize=True).unstack().fillna(0)
        logging.info(proportions)

        # Plot proportions
        plot_categorical_proportions(df_with_clusters, column, output_dir)

    return {
        'df_with_clusters': df_with_clusters,
        'cluster_stats': cluster_stats
    }


@cache_result()
def final_evaluation(pca_result, clustering_result, umap_result, tsne_result, X_processed, output_dir="output"):
    """Perform final evaluation of different clustering approaches.

    Args:
        pca_result (dict): Results from PCA analysis
        clustering_result (dict): Results from clustering comparison
        umap_result (dict): Results from UMAP optimization
        tsne_result (dict): Results from t-SNE optimization
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary containing final evaluation results
    """

    logging.info("Performing final evaluation of different clustering approaches...")
    os.makedirs(output_dir, exist_ok=True)

    # Get the best clustering labels
    best_algorithm_labels = clustering_result['best_algorithm_labels']

    # Get the UMAP optimized labels
    best_umap_labels = umap_result['best_umap_labels']

    # Get the t-SNE optimized labels
    best_tsne_labels = tsne_result['best_tsne_labels']

    # Calculate metrics for each method on the original data

    metrics = {}

    # Calculate metrics for best algorithm on raw data
    if len(np.unique(best_algorithm_labels)) > 1:
        metrics['Best Algorithm'] = {
            'silhouette': silhouette_score(X_processed, best_algorithm_labels),
            'davies_bouldin': davies_bouldin_score(X_processed, best_algorithm_labels),
            'calinski_harabasz': calinski_harabasz_score(X_processed, best_algorithm_labels)
        }
    else:
        metrics['Best Algorithm'] = {
            'silhouette': 0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0
        }

    # Calculate metrics for UMAP + best algorithm
    if len(np.unique(best_umap_labels)) > 1:
        metrics['UMAP + Best Algorithm'] = {
            'silhouette': silhouette_score(X_processed, best_umap_labels),
            'davies_bouldin': davies_bouldin_score(X_processed, best_umap_labels),
            'calinski_harabasz': calinski_harabasz_score(X_processed, best_umap_labels)
        }
    else:
        metrics['UMAP + Best Algorithm'] = {
            'silhouette': 0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0
        }

    # Calculate metrics for t-SNE + best algorithm
    if len(np.unique(best_tsne_labels)) > 1:
        metrics['t-SNE + Best Algorithm'] = {
            'silhouette': silhouette_score(X_processed, best_tsne_labels),
            'davies_bouldin': davies_bouldin_score(X_processed, best_tsne_labels),
            'calinski_harabasz': calinski_harabasz_score(X_processed, best_tsne_labels)
        }
    else:
        metrics['t-SNE + Best Algorithm'] = {
            'silhouette': 0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0
        }

    # Create comparison DataFrame
    metrics_df = pd.DataFrame({
        'Method': list(metrics.keys()),
        'Silhouette Score': [m['silhouette'] for m in metrics.values()],
        'Davies-Bouldin Index': [m['davies_bouldin'] for m in metrics.values()],
        'Calinski-Harabasz Index': [m['calinski_harabasz'] for m in metrics.values()]
    })

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, "final_evaluation_metrics.csv"), index=False)

    # Determine best method based on silhouette score (higher is better)
    best_method = metrics_df.loc[metrics_df['Silhouette Score'].idxmax()]['Method']
    logging.info(f"Best method based on silhouette score: {best_method}")

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Plot silhouette scores
    plt.subplot(3, 1, 1)
    plt.bar(metrics_df['Method'], metrics_df['Silhouette Score'], color='skyblue')
    plt.title('Silhouette Score (higher is better)')
    plt.xticks(rotation=45)

    # Plot Davies-Bouldin index
    plt.subplot(3, 1, 2)
    plt.bar(metrics_df['Method'], metrics_df['Davies-Bouldin Index'], color='salmon')
    plt.title('Davies-Bouldin Index (lower is better)')
    plt.xticks(rotation=45)

    # Plot Calinski-Harabasz index
    plt.subplot(3, 1, 3)
    plt.bar(metrics_df['Method'], metrics_df['Calinski-Harabasz Index'], color='lightgreen')
    plt.title('Calinski-Harabasz Index (higher is better)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_evaluation_metrics.png"), dpi=300)
    plt.close()

    return {
        'metrics': metrics,
        'metrics_df': metrics_df,
        'best_method': best_method
    }

def generate_cluster_profiles(df, final_labels, numerical_cols, categorical_cols, output_dir="output"):
    """Generate and visualize final cluster profiles.
    
    Args:
        df (pandas.DataFrame): Original dataframe
        final_labels (numpy.ndarray): Final cluster labels
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing cluster profiles
    """
    logging.info("Generating final cluster profiles...")
    os.makedirs(output_dir, exist_ok=True)

    # Create a dataframe with the original data and final cluster labels
    final_df = df.copy()
    final_df['cluster'] = final_labels

    # Summary statistics for each cluster
    logging.info("\n===== FINAL CLUSTER PROFILES =====")
    cluster_profiles = {}

    for cluster in np.unique(final_labels):
        cluster_data = final_df[final_df['cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_pct = cluster_size / len(final_df) * 100

        logging.info(f"\nCluster {cluster} ({cluster_size} samples, {cluster_pct:.2f}% of data):")

        # Numerical features
        numerical_stats = {}
        for col in numerical_cols:
            stat = {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max()
            }
            numerical_stats[col] = stat
            logging.info(f"{col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")

        # Top 3 categories for each categorical feature
        categorical_stats = {}
        for col in categorical_cols:
            top_categories = cluster_data[col].value_counts(normalize=True).nlargest(3)
            categorical_stats[col] = top_categories.to_dict()
            logging.info(
                f"{col} top categories: {', '.join([f'{cat}: {val:.2%}' for cat, val in top_categories.items()])}")

        cluster_profiles[cluster] = {
            'size': cluster_size,
            'percentage': cluster_pct,
            'numerical_stats': numerical_stats,
            'categorical_stats': categorical_stats
        }

    # Save results to file
    final_df.to_csv(os.path.join(output_dir, 'diabetes_clustering_results.csv'), index=False)
    logging.info("Results saved to diabetes_clustering_results.csv")

    return {
        'final_df': final_df,
        'cluster_profiles': cluster_profiles
    }