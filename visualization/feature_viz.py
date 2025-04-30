import os

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance_f_values(features, f_vals, p_values, output_dir="output"):
    plt.figure(figsize=(12, 8))

    significance = ['***' if p_values[feat] < 0.001 else
                   '**' if p_values[feat] < 0.01 else
                   '*' if p_values[feat] < 0.05 else
                   '' for feat in features]

    bars = plt.bar(features, f_vals)

    for bar, sig in zip(bars, significance):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1 * max(f_vals),
                sig, ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Features')
    plt.ylabel('F-value')
    plt.title('Feature Importance for Cluster Separation (ANOVA F-test)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_f_values.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_chi2(features, chi2_vals, p_values, output_dir="output"):
    plt.figure(figsize=(12, 6))

    significance = ['***' if p_values[feat] < 0.001 else
                    '**' if p_values[feat] < 0.01 else
                    '*' if p_values[feat] < 0.05 else
                    '' for feat in features]

    bars = plt.bar(features, chi2_vals)

    for bar, sig in zip(bars, significance):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1 * max(chi2_vals),
                 sig, ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Categorical Features')
    plt.ylabel('Chi-square Value')
    plt.title('Categorical Feature Importance for Cluster Separation (Chi-square Test)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_chi2_values.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_features_pairplot(df_with_clusters, top_features, output_dir="output"):
    if len(top_features) <= 1:
        return
    
    plt.figure(figsize=(15, 15))
    for i, feat1 in enumerate(top_features):
        for j, feat2 in enumerate(top_features):
            if i < j:  # Only plot lower triangle
                plt.subplot(len(top_features), len(top_features), i * len(top_features) + j + 1)
                scatter = plt.scatter(df_with_clusters[feat1], df_with_clusters[feat2],
                                     c=df_with_clusters['cluster'], cmap='viridis', alpha=0.7)
                plt.xlabel(feat1)
                plt.ylabel(feat2)
                if i == 0 and j == len(top_features) - 1:
                    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_anomaly_histograms(distances, threshold_kmeans, log_likelihood, threshold_gmm, 
                           svm_scores, anomaly_ratios, output_dir="output"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].hist(distances, bins=50, alpha=0.7, color='blue')
    axes[0].axvline(x=threshold_kmeans, color='red', linestyle='--',
                   label=f'Threshold: {threshold_kmeans:.2f}')
    axes[0].set_title(f'K-means Distance Scores\n({anomaly_ratios["kmeans"]:.2%} anomalies)')
    axes[0].set_xlabel('Distance to Nearest Centroid')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    axes[1].hist(log_likelihood, bins=50, alpha=0.7, color='green')
    axes[1].axvline(x=threshold_gmm, color='red', linestyle='--',
                   label=f'Threshold: {threshold_gmm:.2f}')
    axes[1].set_title(f'GMM Log-Likelihood Scores\n({anomaly_ratios["gmm"]:.2%} anomalies)')
    axes[1].set_xlabel('Log-Likelihood')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    axes[2].hist(svm_scores, bins=50, alpha=0.7, color='purple')
    axes[2].axvline(x=0, color='red', linestyle='--', label='Threshold: 0')
    axes[2].set_title(f'One-Class SVM Decision Scores\n({anomaly_ratios["svm"]:.2%} anomalies)')
    axes[2].set_xlabel('Decision Function Value')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_detection_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_anomaly_visualization(X_pca, anomalies_dict, output_dir="output"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    anomalies_kmeans = anomalies_dict['kmeans']
    axes[0].scatter(X_pca[~anomalies_kmeans, 0], X_pca[~anomalies_kmeans, 1],
                   c='blue', alpha=0.5, s=10, label='Normal')
    axes[0].scatter(X_pca[anomalies_kmeans, 0], X_pca[anomalies_kmeans, 1],
                   c='red', alpha=0.9, s=20, label='Anomaly')
    axes[0].set_title(f'K-means Anomalies ({np.mean(anomalies_kmeans):.2%})')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()

    anomalies_gmm = anomalies_dict['gmm']
    axes[1].scatter(X_pca[~anomalies_gmm, 0], X_pca[~anomalies_gmm, 1],
                   c='blue', alpha=0.5, s=10, label='Normal')
    axes[1].scatter(X_pca[anomalies_gmm, 0], X_pca[anomalies_gmm, 1],
                   c='red', alpha=0.9, s=20, label='Anomaly')
    axes[1].set_title(f'GMM Anomalies ({np.mean(anomalies_gmm):.2%})')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend()

    anomalies_svm = anomalies_dict['svm']
    axes[2].scatter(X_pca[~anomalies_svm, 0], X_pca[~anomalies_svm, 1],
                   c='blue', alpha=0.5, s=10, label='Normal')
    axes[2].scatter(X_pca[anomalies_svm, 0], X_pca[anomalies_svm, 1],
                   c='red', alpha=0.9, s=20, label='Anomaly')
    axes[2].set_title(f'One-Class SVM Anomalies ({np.mean(anomalies_svm):.2%})')
    axes[2].set_xlabel('Principal Component 1')
    axes[2].set_ylabel('Principal Component 2')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()