"""
Anomaly detection functions for diabetes clustering analysis.
"""

import os
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from utils.caching import cache_result
from visualization.feature_viz import plot_anomaly_histograms, plot_anomaly_visualization


@cache_result()
def perform_anomaly_detection(X_pca_optimal, optimal_k, output_dir="pics"):
    """Perform anomaly detection using multiple methods.
    
    Args:
        X_pca_optimal (numpy.ndarray): Optimal PCA-transformed data
        optimal_k (int): Optimal number of clusters
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing anomaly detection results
    """
    logging.info("Performing anomaly detection...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. K-means distance-based anomaly detection
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_pca_optimal)
    distances = kmeans.transform(X_pca_optimal).min(axis=1)

    # Define anomalies as points with distance > mean + 3*std
    anomaly_threshold_kmeans = distances.mean() + 3 * distances.std()
    anomalies_kmeans = distances > anomaly_threshold_kmeans
    anomaly_ratio_kmeans = anomalies_kmeans.mean()
    logging.info(f"K-means anomalies: {anomaly_ratio_kmeans:.2%} of data")

    # 2. GMM log-likelihood anomaly detection
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm.fit(X_pca_optimal)
    log_likelihood = gmm.score_samples(X_pca_optimal)

    # Define anomalies as points with log likelihood < mean - 3*std
    anomaly_threshold_gmm = log_likelihood.mean() - 3 * log_likelihood.std()
    anomalies_gmm = log_likelihood < anomaly_threshold_gmm
    anomaly_ratio_gmm = anomalies_gmm.mean()
    logging.info(f"GMM anomalies: {anomaly_ratio_gmm:.2%} of data")

    # 3. One-Class SVM anomaly detection
    svm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')  # nu is approximately the proportion of outliers
    svm.fit(X_pca_optimal)
    svm_scores = svm.decision_function(X_pca_optimal)
    anomalies_svm = svm.predict(X_pca_optimal) == -1
    anomaly_ratio_svm = anomalies_svm.mean()
    logging.info(f"One-Class SVM anomalies: {anomaly_ratio_svm:.2%} of data")

    # Plot histograms of anomaly scores
    anomaly_ratios = {
        'kmeans': anomaly_ratio_kmeans,
        'gmm': anomaly_ratio_gmm,
        'svm': anomaly_ratio_svm
    }
    
    plot_anomaly_histograms(
        distances, 
        anomaly_threshold_kmeans, 
        log_likelihood, 
        anomaly_threshold_gmm, 
        svm_scores, 
        anomaly_ratios,
        output_dir
    )

    # Visualize anomalies in 2D PCA space
    anomalies_dict = {
        'kmeans': anomalies_kmeans,
        'gmm': anomalies_gmm,
        'svm': anomalies_svm
    }
    
    plot_anomaly_visualization(X_pca_optimal[:, :2], anomalies_dict, output_dir)

    return {
        'anomalies_kmeans': anomalies_kmeans,
        'anomaly_ratio_kmeans': anomaly_ratio_kmeans,
        'anomalies_gmm': anomalies_gmm,
        'anomaly_ratio_gmm': anomaly_ratio_gmm,
        'anomalies_svm': anomalies_svm,
        'anomaly_ratio_svm': anomaly_ratio_svm
    }