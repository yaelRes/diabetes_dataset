"""
Clustering analysis for diabetes clustering.
"""
import json
import logging
import os
import pickle

import hdbscan
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from config import PCA_CONFIG, CLUSTERING_CONFIG, TSNE_CONFIG, UMAP_CONFIG
from utils.caching import cache_result
from visualization.clustering_viz import (
    plot_cluster_distribution,
    plot_categorical_proportions
)
from visualization.correlation_matrix_viz import plot_detailed_correlation_matrix, create_multivariate_analysis, \
    calculate_feature_importance_for_clusters, create_risk_score
from visualization.dimension_reduction import plot_dbscan_kdistance_graph
from visualization.tsne_heatmap_viz import plot_demographic_analysis
from visualization.umap_heatmap_viz import plot_correlation_matrix, plot_feature_boxplots_by_cluster, \
    plot_lifestyle_health_correlation, plot_pairplot_diabetes_indicators


def compare_clustering_algorithms(x_processed, output_dir="output"):
    logging.info("Comparing different clustering algorithms...")
    os.makedirs(output_dir, exist_ok=True)

    n_components_list = PCA_CONFIG["n_components_range"]
    n_components_list_reduced = [nc for nc in n_components_list if nc <= x_processed.shape[1]]
    diff = set(n_components_list) - set(n_components_list_reduced)
    clustering = ["KMeans", "AgglomerativeClustering", "GaussianMixture", "GaussianMixture", "HDBSCAN"]
    reduction = ["PCA", "TSNE"]
    results = {r: {c: [] for c in clustering} for r in reduction}

    max_silhouette_score = 0
    for n_components in n_components_list:
        pca_filename = os.path.join(output_dir, f"x_pca_n{n_components}.pkl")
        with open(pca_filename, "rb") as f:
            x_pca = pickle.load(f)
            ext_data = {}
            ext_str = ""
            cluster_reduced_parameters_data(max_silhouette_score, n_components, ext_data, ext_str,
                                            x_pca, "PCA", results)

    n_components_options = TSNE_CONFIG["n_components_options"]
    perplexity_options = TSNE_CONFIG["perplexity_options"]
    results["TSNE"] = {}
    for n_components in n_components_options:
        for perplexity in perplexity_options:
            tsne_filename = os.path.join(output_dir, f"x_tsne_n{n_components}_p{perplexity}.pkl")
            with open(tsne_filename, "rb") as f:
                x_tsne = pickle.load(f)
                ext_data = {"perplexity": perplexity}
                ext_str = f"perplexity:{perplexity}"
                cluster_reduced_parameters_data(max_silhouette_score, n_components, ext_data, ext_str,
                                                x_tsne, "TSNE", results)

    n_neighbors_options = UMAP_CONFIG["n_neighbors_options"]
    min_dist_options = UMAP_CONFIG["min_dist_options"]
    n_components_options = UMAP_CONFIG["n_components_options"]
    for n_components in n_components_options:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                umap_filename = os.path.join(output_dir, f"x_umap_n{n_components}_nn{n_neighbors}_md{min_dist}.pkl")
                with open(umap_filename, "rb") as f:
                    x_umap = pickle.load(f)
                    ext_data = {"n_neighbors": n_neighbors, "min_dist": min_dist}
                    ext_str = f"n_neighbors: {n_neighbors} min_dist:{min_dist}"
                    cluster_reduced_parameters_data(max_silhouette_score, n_components,
                                                    ext_data, ext_str, x_umap, "UMAP", results)

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info(f"PCA max_silhouette_score={max_silhouette_score} ")


def cluster_reduced_parameters_data(max_silhouette_score, n_components,
                                    ext_data, ext_str, x_data, reduction_algo, results):
    k_range = CLUSTERING_CONFIG["k_range"]
    hdbscan_min_samples_list = CLUSTERING_CONFIG["hdbscan_min_samples"]
    random_state = CLUSTERING_CONFIG["random_state"]
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=500, n_init=10)
            kmeans_labels = kmeans.fit_predict(x_data)
            kmeans_silhouette = silhouette_score(x_data, kmeans_labels)
            logging.info(f"K-means {reduction_algo} n_components={n_components} {ext_str} n_clusters={k} "
                         f"Silhouette: {kmeans_silhouette:.4f}")
            results[reduction_algo]["KMeans"].append({
                'n_component': n_components,
                'k': k,
                'score': kmeans_silhouette,
                'ext_data': ext_data,
                'labels': kmeans_labels
            })
            max_silhouette_score = max(max_silhouette_score, kmeans_silhouette)
            hierarchical = AgglomerativeClustering(n_clusters=k)
            hierarchical_labels = hierarchical.fit_predict(x_data)
            hierarchical_silhouette = silhouette_score(x_data, hierarchical_labels)
            logging.info(
                f"Hierarchical {reduction_algo} n_components={n_components} {ext_str} n_clusters={k}"
                f" silhouette: {hierarchical_silhouette:.4f}")
            results[reduction_algo]["AgglomerativeClustering"].append({
                'n_component': n_components,
                'k': k,
                'score': hierarchical_silhouette,
                'ext_data': ext_data,
                'labels': hierarchical_labels
            })
            max_silhouette_score = max(max_silhouette_score, hierarchical_silhouette)
            gmm = GaussianMixture(n_components=k, random_state=random_state)
            gmm_labels = gmm.fit_predict(x_data)
            gmm_silhouette = silhouette_score(x_data, gmm_labels)
            logging.info(
                f"GMM {reduction_algo} n_components={n_components} n_clusters={k} {ext_str}"
                f" silhouette: {gmm_silhouette:.4f}")

            results[reduction_algo]["GaussianMixture"].append({
                'n_component': n_components,
                'k': k,
                'score': gmm_silhouette,
                'ext_data': ext_data,
                'labels': gmm_labels
            })
            max_silhouette_score = max(max_silhouette_score, gmm_silhouette)
            for min_samples in hdbscan_min_samples_list:
                hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=k, min_samples=min_samples)
                hdbscan_labels = hdbscan_cluster.fit_predict(x_data)
                hdbscan_silhouette = silhouette_score(x_data, hdbscan_labels)
                max_silhouette_score = max(max_silhouette_score, hdbscan_silhouette)
                mask = hdbscan_labels != -1
                X_no_noise = x_data[mask]
                labels_no_noise = hdbscan_labels[mask]
                silhouette_avg = 0
                if len(np.unique(labels_no_noise)) > 1:
                    silhouette_avg = silhouette_score(X_no_noise, labels_no_noise)
                    max_silhouette_score = max(max_silhouette_score, silhouette_avg)
                logging.info(
                    f"HDBSCAN {reduction_algo} n_components={n_components} {ext_str}"
                    f" n_clusters={k} min_samples={min_samples} "
                    f"silhouette: {hdbscan_silhouette:.4f}"
                    f" excluding noise if 0 one cluster): {silhouette_avg:.3f}")
                results[reduction_algo]["HDBSCAN"].append({
                    'n_component': n_components,
                    'k': k,
                    'min_samples': min_samples,
                    'score_with_noise': hdbscan_silhouette,
                    'score_with_no_noise': silhouette_avg,
                    'ext_data': ext_data,
                    'labels': hdbscan_labels
                })
                max_silhouette_score = max(max_silhouette_score, hdbscan_silhouette)
        except Exception as e:
            logging.error(f"Error during clustering: {e}")
            import traceback
            logging.error(traceback.format_exc())


def visual_results(results):
    pass


def dbscan_cluster(x_data, optimal_k, output_dir="output"):
    nn = NearestNeighbors(n_neighbors=min(10, len(x_data) - 1))
    nn.fit(x_data)
    distances, indices = nn.kneighbors(x_data)
    distances = np.sort(distances[:, -1])

    # Find the elbow point
    knee_point = np.diff(np.diff(distances))
    elbow_index = np.argmax(knee_point) + 1
    eps_value = distances[elbow_index]
    logging.info(f"selected DBSCAN eps value: {eps_value:.4f}")

    # Plot the k-distance graph to find the elbow
    plot_dbscan_kdistance_graph(distances, elbow_index, output_dir)

    # Run DBSCAN with the selected eps
    dbscan = DBSCAN(eps=eps_value, min_samples=5)
    dbscan_labels = dbscan.fit_predict(x_data)

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
            temp_labels = temp_dbscan.fit_predict(x_data)
            n_clusters = len(np.unique(temp_labels[temp_labels != -1]))
            noise_ratio = np.sum(temp_labels == -1) / len(temp_labels)

            logging.info(f"DBSCAN with eps={eps:.4f}: {n_clusters} clusters, {noise_ratio:.2%} noise")

            # Better result has more clusters and less noise
            if n_clusters > best_n_clusters and noise_ratio < 0.5:
                best_eps = eps
                best_n_clusters = n_clusters
                best_noise_ratio = noise_ratio
                dbscan_labels = temp_labels

        logging.info(f"selected better DBSCAN eps value: {best_eps:.4f}")

    # Calculate DBSCAN silhouette (if there are multiple clusters and not all noise)
    if len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels):
        dbscan_silhouette = silhouette_score(x_data, dbscan_labels)
    elif len(np.unique(dbscan_labels)) > 1:
        # Calculate silhouette only on non-noise points
        non_noise = dbscan_labels != -1
        if np.sum(non_noise) > 1 and len(np.unique(dbscan_labels[non_noise])) > 1:
            dbscan_silhouette = silhouette_score(x_data[non_noise], dbscan_labels[non_noise])
        else:
            dbscan_silhouette = 0
    else:
        dbscan_silhouette = 0
    logging.info(f"DBSCAN silhouette : {dbscan_silhouette:.4f}")


def analyze_cluster_characteristics(df, best_algorithm_labels, numerical_cols, categorical_cols, output_dir="output"):
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
    plot_correlation_matrix(df, numerical_cols, output_dir)

    # Generate detailed correlation matrix with significance
    plot_detailed_correlation_matrix(df, numerical_cols, output_dir)

    # Generate boxplots for all numerical features by cluster
    plot_feature_boxplots_by_cluster(df_with_clusters, numerical_cols, output_dir)

    # Check if there are lifestyle and health columns to create the specialized correlation heatmap
    lifestyle_cols = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                           ['activity', 'exercise', 'physical', 'diet', 'alcohol',
                                                            'smoking', 'sleep'])]

    health_cols = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                        ['hba1c', 'glucose', 'bmi', 'pressure', 'cholesterol', 'blood',
                                                         'weight', 'insulin'])]

    if lifestyle_cols and health_cols:
        plot_lifestyle_health_correlation(df, lifestyle_cols, health_cols, output_dir)

    # Create pairplot for key diabetes indicators if they exist
    diabetes_indicators = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                                ['hba1c', 'glucose', 'bmi', 'age', 'waist', 'insulin'])]

    if len(diabetes_indicators) >= 3:
        plot_pairplot_diabetes_indicators(df_with_clusters, diabetes_indicators[:5], output_dir)

    # Create demographic analyses if demographic columns exist
    demographic_cols = [col for col in categorical_cols if any(kw in col.lower() for kw in
                                                               ['ethnicity', 'gender', 'sex', 'age_group', 'race'])]

    if demographic_cols:
        for demo_col in demographic_cols:
            for health_col in health_cols[:3]:  # Limit to first 3 health columns
                plot_demographic_analysis(df_with_clusters, demo_col, health_col, output_dir)

    # Create multivariate analysis
    create_multivariate_analysis(df_with_clusters, numerical_cols, output_dir)

    # Calculate feature importance for clusters
    calculate_feature_importance_for_clusters(df_with_clusters, numerical_cols, categorical_cols, output_dir)

    # Create risk score if appropriate risk factors exist
    risk_factors = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                         ['glucose', 'bmi', 'hba1c', 'insulin', 'pressure',
                                                          'cholesterol', 'triglycerides'])]

    if risk_factors:
        create_risk_score(df, df_with_clusters, risk_factors, output_dir)
    return {
        'df_with_clusters': df_with_clusters,
        'cluster_stats': cluster_stats
    }


@cache_result()
def final_evaluation(pca_result, clustering_result, umap_result, tsne_result, X_processed, output_dir="output"):
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
