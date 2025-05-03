import logging
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from visualization.feature_viz import plot_anomaly_histograms, plot_anomaly_visualization


def perform_anomaly_detection(diabetes_pca, n_clusters, diabetes_output_dir="output"):

    os.makedirs(diabetes_output_dir, exist_ok=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(diabetes_pca)
    kmeans_dist = kmeans.transform(diabetes_pca).min(axis=1)

    kmeans_thresh = kmeans_dist.mean() + 3 * kmeans_dist.std()
    kmeans_anom = kmeans_dist > kmeans_thresh
    kmeans_ratio = kmeans_anom.mean()
    logging.info(f"K-means anomalies: {kmeans_ratio:.2%} of data")

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(diabetes_pca)
    gmm_ll = gmm.score_samples(diabetes_pca)

    gmm_thresh = gmm_ll.mean() - 3 * gmm_ll.std()
    gmm_anom = gmm_ll < gmm_thresh
    gmm_ratio = gmm_anom.mean()
    logging.info(f"GMM anomalies: {gmm_ratio:.2%} of data")

    svm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')
    svm.fit(diabetes_pca)
    svm_scores = svm.decision_function(diabetes_pca)
    svm_anom = svm.predict(diabetes_pca) == -1
    svm_ratio = svm_anom.mean()
    logging.info(f"One-Class SVM anomalies: {svm_ratio:.2%} of data")

    anomaly_ratios = {
        'kmeans': kmeans_ratio,
        'gmm': gmm_ratio,
        'svm': svm_ratio
    }
    
    plot_anomaly_histograms(
        kmeans_dist, 
        kmeans_thresh, 
        gmm_ll, 
        gmm_thresh, 
        svm_scores, 
        anomaly_ratios,
        diabetes_output_dir
    )

    anomalies_dict = {
        'kmeans': kmeans_anom,
        'gmm': gmm_anom,
        'svm': svm_anom
    }
    
    plot_anomaly_visualization(diabetes_pca[:, :2], anomalies_dict, diabetes_output_dir)