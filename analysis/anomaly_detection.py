import logging
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from visualization.feature_viz import plot_anomaly_histograms, plot_anomaly_visualization


def perform_anomaly_detection(x_pca_optimal, optimal_k, output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(x_pca_optimal)
    distances = kmeans.transform(x_pca_optimal).min(axis=1)

    anomaly_threshold_kmeans = distances.mean() + 3 * distances.std()
    anomalies_kmeans = distances > anomaly_threshold_kmeans
    anomaly_ratio_kmeans = anomalies_kmeans.mean()
    logging.info(f"K-means anomalies: {anomaly_ratio_kmeans:.2%} of data")

    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm.fit(x_pca_optimal)
    log_likelihood = gmm.score_samples(x_pca_optimal)

    anomaly_threshold_gmm = log_likelihood.mean() - 3 * log_likelihood.std()
    anomalies_gmm = log_likelihood < anomaly_threshold_gmm
    anomaly_ratio_gmm = anomalies_gmm.mean()
    logging.info(f"GMM anomalies: {anomaly_ratio_gmm:.2%} of data")

    svm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')
    svm.fit(x_pca_optimal)
    svm_scores = svm.decision_function(x_pca_optimal)
    anomalies_svm = svm.predict(x_pca_optimal) == -1
    anomaly_ratio_svm = anomalies_svm.mean()
    logging.info(f"One-Class SVM anomalies: {anomaly_ratio_svm:.2%} of data")

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

    anomalies_dict = {
        'kmeans': anomalies_kmeans,
        'gmm': anomalies_gmm,
        'svm': anomalies_svm
    }
    
    plot_anomaly_visualization(x_pca_optimal[:, :2], anomalies_dict, output_dir)