import logging
import os
import pickle
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from utils.data_utils import load_dataset, get_column_types
from visualization.clustering_viz import plot_categorical_proportions, plot_cluster_distribution
from visualization.correlation_matrix_viz import create_risk_score, calculate_feature_importance_for_clusters, \
    create_multivariate_analysis, plot_detailed_correlation_matrix
from visualization.tsne_heatmap_viz import plot_demographic_analysis
from visualization.umap_heatmap_viz import plot_pairplot_diabetes_indicators, plot_lifestyle_health_correlation, \
    plot_feature_boxplots_by_cluster, plot_correlation_matrix


def diabetes_clustering_heatmaps(top_n=20, diabetes_out="output"):
    clustering = ["KMeans", "AgglomerativeClustering", "GaussianMixture", "HDBSCAN"]
    csv_dir = os.path.join(diabetes_out, "clustering_csv_results")
    labels_dir = os.path.join(diabetes_out, "labels_pkl")
    f, axs = plt.subplots(3, 3, figsize=(21, 21))
    for i, rdm in enumerate(["PCA", "TSNE", "UMAP"]):
        df = pd.read_csv(os.path.join(csv_dir, f"{rdm}_all_results.csv"))
        hdb = df[df.clustering_algorithm == 'HDBSCAN']
        if len(hdb):
            hdb['score'] = hdb.score_with_noise.fillna(hdb.score)
            hdb_top = hdb.nlargest(top_n, 'score')
            hdb_top['n_comp'] = hdb_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(hdb_top, values='score', index='n_comp', columns='min_samples', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 0], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 0].set_title(f'{rdm}-HDBSCAN with noise')
            hdb['score'] = hdb.score_without_noise.fillna(hdb.score)
            hdb_top = hdb.nlargest(top_n, 'score')
            hdb_top['n_comp'] = hdb_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(hdb_top, values='score', index='n_comp', columns='min_samples', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 1], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 1].set_title(f'{rdm}-HDBSCAN without noise')
        reg = df[df.clustering_algorithm.isin(['KMeans', 'AgglomerativeClustering', 'GaussianMixture'])]
        if len(reg):
            reg_top = reg.nlargest(top_n, 'score')
            reg_top['n_comp'] = reg_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(reg_top, values='score', index=['clustering_algorithm', 'n_comp'], columns='k', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 2], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 2].set_title(f'{rdm}-Other')
    plt.tight_layout()
    plt.savefig(os.path.join(diabetes_out, 'clustering_heatmaps.png'), dpi=300)


def diabetes_kmeans_loss(smallest_n=20, diabetes_out="output"):
    csv_dir = os.path.join(diabetes_out, "clustering_csv_results")
    labels_dir = os.path.join(diabetes_out, "labels_pkl")
    f, axs = plt.subplots(1, 1, figsize=(10, 8))
    for rdm in ["PCA"]:
        df = pd.read_csv(os.path.join(csv_dir, f"{rdm}_all_results.csv"))
        reg = df[df.clustering_algorithm == 'KMeans']
        if len(reg):
            reg_top = reg.nsmallest(smallest_n, 'loss')
            reg_top['n_comp'] = reg_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(reg_top, values='loss', index=['clustering_algorithm', 'n_comp'], columns='k', aggfunc=np.min)
            sns.heatmap(pivot, ax=axs, cmap='viridis', annot=True, fmt='.2f')
            axs.set_title(f'{rdm}-KMeans loss')
    plt.tight_layout()
    plt.savefig(os.path.join(diabetes_out, 'k_means_loss_clustering_heatmaps.png'), dpi=300)


def diabetes_cluster_stats(df, cluster_labels, num_cols, cat_cols, diabetes_out="output"):
    os.makedirs(diabetes_out, exist_ok=True)
    dfc = df.copy()
    dfc['cluster'] = cluster_labels
    stats = {}
    for col in num_cols:
        stats[col] = dfc.groupby('cluster')[col].agg(['mean', 'std', 'min', 'max'])
        plot_cluster_distribution(dfc, col, diabetes_out)
    for col in num_cols:
        logging.info(f"cluster statistics for {col}:")
        logging.info(stats[col])
    for col in cat_cols:
        logging.info(f"cluster proportions for {col}:")
        props = dfc.groupby('cluster')[col].value_counts(normalize=True).unstack().fillna(0)
        logging.info(props)
        plot_categorical_proportions(dfc, col, diabetes_out)
    plot_correlation_matrix(df, num_cols, diabetes_out)
    plot_detailed_correlation_matrix(df, num_cols, diabetes_out)
    plot_feature_boxplots_by_cluster(dfc, num_cols, diabetes_out)
    lifestyle_cols = [col for col in num_cols if any(kw in col.lower() for kw in ['activity', 'exercise', 'physical', 'diet', 'alcohol', 'smoking', 'sleep'])]
    health_cols = [col for col in num_cols if any(kw in col.lower() for kw in ['hba1c', 'glucose', 'bmi', 'pressure', 'cholesterol', 'blood', 'weight', 'insulin'])]
    if lifestyle_cols and health_cols:
        plot_lifestyle_health_correlation(df, lifestyle_cols, health_cols, diabetes_out)
    diabetes_inds = [col for col in num_cols if any(kw in col.lower() for kw in ['hba1c', 'glucose', 'bmi', 'age', 'waist', 'insulin'])]
    if len(diabetes_inds) >= 3:
        plot_pairplot_diabetes_indicators(dfc, diabetes_inds[:5], diabetes_out)
    demo_cols = [col for col in cat_cols if any(kw in col.lower() for kw in ['ethnicity', 'gender', 'sex', 'age_group', 'race'])]
    if demo_cols:
        for demo_col in demo_cols:
            for health_col in health_cols[:3]:
                plot_demographic_analysis(dfc, demo_col, health_col, diabetes_out)
    create_multivariate_analysis(dfc, num_cols, diabetes_out)
    calculate_feature_importance_for_clusters(dfc, num_cols, cat_cols, diabetes_out)
    risk_factors = [col for col in num_cols if any(kw in col.lower() for kw in ['glucose', 'bmi', 'hba1c', 'insulin', 'pressure', 'cholesterol', 'triglycerides'])]
    if risk_factors:
        create_risk_score(df, dfc, risk_factors, diabetes_out)


def diabetes_best_algo_eval(pca_res, umap_res, tsne_res, x, diabetes_out="output"):
    os.makedirs(diabetes_out, exist_ok=True)
    opts = ['PCA + Best Algorithm', 'UMAP + Best Algorithm', 't-SNE + Best Algorithm']
    res_list = [pca_res['best_pca_labels'], umap_res['best_umap_labels'], tsne_res['best_tsne_labels']]
    metrics = {o: {} for o in opts}
    for idx, r in enumerate(res_list):
        if len(np.unique(r)) > 1:
            metrics[opts[idx]] = {
                'silhouette': silhouette_score(x, r),
                'davies_bouldin': davies_bouldin_score(x, r),
                'calinski_harabasz': calinski_harabasz_score(x, r)
            }
        else:
            metrics['Best Algorithm'] = {
                'silhouette': 0,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0
            }
    df = pd.DataFrame({
        'Method': list(metrics.keys()),
        'Silhouette Score': [m['silhouette'] for m in metrics.values()],
        'Davies-Bouldin Index': [m['davies_bouldin'] for m in metrics.values()],
        'Calinski-Harabasz Index': [m['calinski_harabasz'] for m in metrics.values()]
    })
    df.to_csv(os.path.join(diabetes_out, "best_algorithm_evaluation_metrics.csv"), index=False)
    best_method = df.loc[df['Silhouette Score'].idxmax()]['Method']
    logging.info(f"Best method based on silhouette score: {best_method}")
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.bar(df['Method'], df['Silhouette Score'], color='skyblue')
    plt.title('Silhouette Score (higher is better)')
    plt.xticks(rotation=45)
    plt.subplot(3, 1, 2)
    plt.bar(df['Method'], df['Davies-Bouldin Index'], color='salmon')
    plt.title('Davies-Bouldin Index (lower is better)')
    plt.xticks(rotation=45)
    plt.subplot(3, 1, 3)
    plt.bar(df['Method'], df['Calinski-Harabasz Index'], color='lightgreen')
    plt.title('Calinski-Harabasz Index (higher is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(diabetes_out, "best_algorithm_metrics.png"), dpi=300)
    plt.close()


def diabetes_cluster_profiles(df, final_labels, num_cols, cat_cols, diabetes_out="output"):
    os.makedirs(diabetes_out, exist_ok=True)
    final_df = df.copy()
    final_df['cluster'] = final_labels
    logging.info("=====cluster profiles =====")
    for cluster in np.unique(final_labels):
        cluster_data = final_df[final_df['cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_pct = cluster_size / len(final_df) * 100
        logging.info(f"cluster {cluster} ({cluster_size} samples, {cluster_pct:.2f}% of data):")
        for col in num_cols:
            stat = {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max()
            }
            logging.info(f"{col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")
        for col in cat_cols:
            top_categories = cluster_data[col].value_counts(normalize=True).nlargest(3)
            logging.info(f"{col} top categories: {', '.join([f'{cat}: {val:.2%}' for cat, val in top_categories.items()])}")

    final_df.to_csv(os.path.join(diabetes_out, 'diabetes_clustering_results.csv'), index=False)
    logging.info("Results saved to diabetes_clustering_results.csv")
