"""
Visualization functions for feature importance and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance_f_values(features, f_vals, p_values, output_dir="pics"):
    """Plot feature importance based on F-values from ANOVA.
    
    Args:
        features (list): List of feature names
        f_vals (list): List of F-values
        p_values (dict): Dictionary mapping features to p-values
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(12, 8))
    
    # Create significance indicators
    significance = ['***' if p_values[feat] < 0.001 else
                   '**' if p_values[feat] < 0.01 else
                   '*' if p_values[feat] < 0.05 else
                   '' for feat in features]
    
    # Plot F-values with significance indicators
    bars = plt.bar(features, f_vals)
    
    # Add significance stars
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


def plot_feature_importance_chi2(features, chi2_vals, p_values, output_dir="pics"):
    """Plot feature importance for categorical features based on Chi-square.
    
    Args:
        features (list): List of feature names
        chi2_vals (list): List of Chi-square values
        p_values (dict): Dictionary mapping features to p-values
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(12, 6))
    
    # Create significance indicators
    significance = ['***' if p_values[feat] < 0.001 else
                    '**' if p_values[feat] < 0.01 else
                    '*' if p_values[feat] < 0.05 else
                    '' for feat in features]
    
    # Plot chi-square values with significance indicators
    bars = plt.bar(features, chi2_vals)
    
    # Add significance stars
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


def plot_top_features_pairplot(df_with_clusters, top_features, output_dir="pics"):
    """Create pairwise scatter plots for top features.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        top_features (list): List of top feature names
        output_dir (str): Directory to save output files
    """
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
                           svm_scores, anomaly_ratios, output_dir="pics"):
    """Plot histograms of anomaly scores for different anomaly detection methods.
    
    Args:
        distances (numpy.ndarray): K-means distance scores
        threshold_kmeans (float): K-means anomaly threshold
        log_likelihood (numpy.ndarray): GMM log-likelihood scores
        threshold_gmm (float): GMM anomaly threshold
        svm_scores (numpy.ndarray): One-Class SVM decision scores
        anomaly_ratios (dict): Dictionary with anomaly ratios for each method
        output_dir (str): Directory to save output files
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # K-means distance histogram
    axes[0].hist(distances, bins=50, alpha=0.7, color='blue')
    axes[0].axvline(x=threshold_kmeans, color='red', linestyle='--',
                   label=f'Threshold: {threshold_kmeans:.2f}')
    axes[0].set_title(f'K-means Distance Scores\n({anomaly_ratios["kmeans"]:.2%} anomalies)')
    axes[0].set_xlabel('Distance to Nearest Centroid')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # GMM log-likelihood histogram
    axes[1].hist(log_likelihood, bins=50, alpha=0.7, color='green')
    axes[1].axvline(x=threshold_gmm, color='red', linestyle='--',
                   label=f'Threshold: {threshold_gmm:.2f}')
    axes[1].set_title(f'GMM Log-Likelihood Scores\n({anomaly_ratios["gmm"]:.2%} anomalies)')
    axes[1].set_xlabel('Log-Likelihood')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # One-Class SVM decision function histogram
    axes[2].hist(svm_scores, bins=50, alpha=0.7, color='purple')
    axes[2].axvline(x=0, color='red', linestyle='--', label='Threshold: 0')
    axes[2].set_title(f'One-Class SVM Decision Scores\n({anomaly_ratios["svm"]:.2%} anomalies)')
    axes[2].set_xlabel('Decision Function Value')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_detection_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_anomaly_visualization(X_pca, anomalies_dict, output_dir="pics"):
    """Visualize anomalies in 2D PCA space.
    
    Args:
        X_pca (numpy.ndarray): 2D PCA projection
        anomalies_dict (dict): Dictionary with anomaly boolean masks for each method
        output_dir (str): Directory to save output files
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # K-means anomalies
    anomalies_kmeans = anomalies_dict['kmeans']
    axes[0].scatter(X_pca[~anomalies_kmeans, 0], X_pca[~anomalies_kmeans, 1],
                   c='blue', alpha=0.5, s=10, label='Normal')
    axes[0].scatter(X_pca[anomalies_kmeans, 0], X_pca[anomalies_kmeans, 1],
                   c='red', alpha=0.9, s=20, label='Anomaly')
    axes[0].set_title(f'K-means Anomalies ({np.mean(anomalies_kmeans):.2%})')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    
    # GMM anomalies
    anomalies_gmm = anomalies_dict['gmm']
    axes[1].scatter(X_pca[~anomalies_gmm, 0], X_pca[~anomalies_gmm, 1],
                   c='blue', alpha=0.5, s=10, label='Normal')
    axes[1].scatter(X_pca[anomalies_gmm, 0], X_pca[anomalies_gmm, 1],
                   c='red', alpha=0.9, s=20, label='Anomaly')
    axes[1].set_title(f'GMM Anomalies ({np.mean(anomalies_gmm):.2%})')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend()
    
    # One-Class SVM anomalies
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