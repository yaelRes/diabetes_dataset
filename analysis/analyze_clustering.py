import logging
import os
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from visualization.clustering_viz import plot_categorical_proportions, plot_cluster_distribution
from visualization.correlation_matrix_viz import create_risk_score, calculate_feature_importance_for_clusters, \
    create_multivariate_analysis, plot_detailed_correlation_matrix
from visualization.tsne_heatmap_viz import plot_demographic_analysis
from visualization.umap_heatmap_viz import plot_pairplot_diabetes_indicators, plot_lifestyle_health_correlation, \
    plot_feature_boxplots_by_cluster, plot_correlation_matrix


def analyze_clustering(top_n=50, output_dir="output"):
    logging.info("analyze_clustering")

    clustering = ["KMeans", "AgglomerativeClustering", "GaussianMixture","HDBSCAN"]
    csv_dir = os.path.join(output_dir, "clustering_csv_results")
    if not os.path.exists(csv_dir):
        logging.warning(f"csv directory {csv_dir} does not exist")
    labels_dir = os.path.join(output_dir, "labels_pkl")
    if not os.path.exists(labels_dir):
        logging.warning(f"clustering PKL directory {labels_dir} does not exist")

    f, axs = plt.subplots(3, 3, figsize=(21, 21))
    for i,rdm in enumerate(["PCA", "TSNE", "UMAP"]):
        reduction_method_csv_file = os.path.join(csv_dir, f"{rdm}_all_results.csv")
        df = pd.read_csv(reduction_method_csv_file)

        hdb = df[df.clustering_algorithm == 'HDBSCAN']
        if len(hdb):
            hdb['score'] = hdb.score_with_noise.fillna(hdb.score)
            hdb_top = hdb.nlargest(top_n, 'score')
            hdb_top['n_comp'] = hdb_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(hdb_top, values='score',
                                   index='n_comp',
                                   columns='min_samples', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 0], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 0].set_title(f'{rdm}-HDBSCAN with noice')

            hdb['score'] = hdb.score_without_noise.fillna(hdb.score)
            hdb_top = hdb.nlargest(top_n, 'score')
            hdb_top['n_comp'] = hdb_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(hdb_top, values='score',
                                   index='n_comp',
                                   columns='min_samples', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 1], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 1].set_title(f'{rdm}-HDBSCAN without noice')

        reg = df[df.clustering_algorithm.isin(['KMeans', 'AgglomerativeClustering', 'GaussianMixture'])]
        if len(reg):
            reg_top = reg.nlargest(top_n, 'score')
            reg_top['n_comp'] = reg_top.n_component.fillna('auto').astype(str)
            pivot = pd.pivot_table(reg_top, values='score',
                                   index=['clustering_algorithm', 'n_comp'],
                                   columns='k', aggfunc=np.max)
            sns.heatmap(pivot, ax=axs[i, 2], cmap='viridis', annot=True, fmt='.2f')
            axs[i, 2].set_title(f'{rdm}-Other')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'clustering_heatmaps.png'), dpi=300)


def analyze_cluster_characteristics(df, best_algorithm_labels, numerical_cols, categorical_cols, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = best_algorithm_labels

    cluster_stats = {}
    for column in numerical_cols:
        cluster_stats[column] = df_with_clusters.groupby('cluster')[column].agg(['mean', 'std', 'min', 'max'])
        plot_cluster_distribution(df_with_clusters, column, output_dir)

    for column in numerical_cols:
        logging.info(f"cluster statistics for {column}:")
        logging.info(cluster_stats[column])

    for column in categorical_cols:
        logging.info(f"cluster proportions for {column}:")
        proportions = df_with_clusters.groupby('cluster')[column].value_counts(normalize=True).unstack().fillna(0)
        logging.info(proportions)
        plot_categorical_proportions(df_with_clusters, column, output_dir)

    plot_correlation_matrix(df, numerical_cols, output_dir)

    plot_detailed_correlation_matrix(df, numerical_cols, output_dir)

    plot_feature_boxplots_by_cluster(df_with_clusters, numerical_cols, output_dir)

    lifestyle_cols = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                           ['activity', 'exercise', 'physical', 'diet', 'alcohol',
                                                            'smoking', 'sleep'])]

    health_cols = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                        ['hba1c', 'glucose', 'bmi', 'pressure', 'cholesterol', 'blood',
                                                         'weight', 'insulin'])]

    if lifestyle_cols and health_cols:
        plot_lifestyle_health_correlation(df, lifestyle_cols, health_cols, output_dir)

    diabetes_indicators = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                                ['hba1c', 'glucose', 'bmi', 'age', 'waist', 'insulin'])]

    if len(diabetes_indicators) >= 3:
        plot_pairplot_diabetes_indicators(df_with_clusters, diabetes_indicators[:5], output_dir)

    demographic_cols = [col for col in categorical_cols if any(kw in col.lower() for kw in
                                                               ['ethnicity', 'gender', 'sex', 'age_group', 'race'])]

    if demographic_cols:
        for demo_col in demographic_cols:
            for health_col in health_cols[:3]:  # Limit to first 3 health columns
                plot_demographic_analysis(df_with_clusters, demo_col, health_col, output_dir)

    create_multivariate_analysis(df_with_clusters, numerical_cols, output_dir)

    calculate_feature_importance_for_clusters(df_with_clusters, numerical_cols, categorical_cols, output_dir)

    risk_factors = [col for col in numerical_cols if any(kw in col.lower() for kw in
                                                         ['glucose', 'bmi', 'hba1c', 'insulin', 'pressure',
                                                          'cholesterol', 'triglycerides'])]

    if risk_factors:
        create_risk_score(df, df_with_clusters, risk_factors, output_dir)


def best_algorithm_evaluation(pca_result, umap_result, tsne_result, x_processed, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    clustering_options = ['PCA + Best Algorithm', 'UMAP + Best Algorithm', 't-SNE + Best Algorithm']
    reduction_res_list = [pca_result['best_pca_labels'], umap_result['best_umap_labels'],
                          tsne_result['best_tsne_labels']]
    metrics: Dict[str, Dict] = {co: {} for co in clustering_options}
    for idx, reduction_res in reduction_res_list:
        if len(np.unique(reduction_res)) > 1:
            metrics[clustering_options[idx]] = {
                'silhouette': silhouette_score(x_processed, reduction_res),
                'davies_bouldin': davies_bouldin_score(x_processed, reduction_res),
                'calinski_harabasz': calinski_harabasz_score(x_processed, reduction_res)
            }
        else:
            metrics['Best Algorithm'] = {
                'silhouette': 0,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0
            }

    metrics_df = pd.DataFrame({
        'Method': list(metrics.keys()),
        'Silhouette Score': [m['silhouette'] for m in metrics.values()],
        'Davies-Bouldin Index': [m['davies_bouldin'] for m in metrics.values()],
        'Calinski-Harabasz Index': [m['calinski_harabasz'] for m in metrics.values()]
    })

    metrics_df.to_csv(os.path.join(output_dir, "best_algorithm_evaluation_metrics.csv"), index=False)

    best_method = metrics_df.loc[metrics_df['Silhouette Score'].idxmax()]['Method']
    logging.info(f"Best method based on silhouette score: {best_method}")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.bar(metrics_df['Method'], metrics_df['Silhouette Score'], color='skyblue')
    plt.title('Silhouette Score (higher is better)')
    plt.xticks(rotation=45)

    plt.subplot(3, 1, 2)
    plt.bar(metrics_df['Method'], metrics_df['Davies-Bouldin Index'], color='salmon')
    plt.title('Davies-Bouldin Index (lower is better)')
    plt.xticks(rotation=45)

    plt.subplot(3, 1, 3)
    plt.bar(metrics_df['Method'], metrics_df['Calinski-Harabasz Index'], color='lightgreen')
    plt.title('Calinski-Harabasz Index (higher is better)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_algorithm_metrics.png"), dpi=300)
    plt.close()


def generate_cluster_profiles(df, final_labels, numerical_cols, categorical_cols, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    final_df = df.copy()
    final_df['cluster'] = final_labels

    logging.info("=====cluster profiles =====")
    cluster_profiles = {}

    for cluster in np.unique(final_labels):
        cluster_data = final_df[final_df['cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_pct = cluster_size / len(final_df) * 100

        logging.info(f"cluster {cluster} ({cluster_size} samples, {cluster_pct:.2f}% of data):")

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

    final_df.to_csv(os.path.join(output_dir, 'diabetes_clustering_results.csv'), index=False)
    logging.info("Results saved to diabetes_clustering_results.csv")


analyze_clustering(50,output_dir=r"D:\projects\goFish\pythonProject\FINAL\All_Features\train")