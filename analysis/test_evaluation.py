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
from sklearn.neighbors import NearestNeighbors
from visualization.clustering_viz import plot_best_clustering_result
from visualization.feature_viz import plot_top_features_pairplot


def diabetes_test_eval(diabetes_test_df, diabetes_test_x, cluster_res, pca_res, umap_res, eval_res, num_cols, cat_cols, diabetes_out="output/test"):
    logging.info("Evaluating clustering performance on test set...")
    os.makedirs(diabetes_out, exist_ok=True)
    if 'pca_2d' in pca_res:
        test_pca_2d = pca_res['pca_2d'].transform(diabetes_test_x)
    else:
        test_pca_2d = PCA(n_components=2).fit_transform(diabetes_test_x)
    best_algo = cluster_res['best_algorithm']
    k = cluster_res['best_algorithm_labels'].max() + 1
    pca_opt = PCA(n_components=cluster_res['X_pca_optimal'].shape[1]).fit(diabetes_test_x)
    test_pca_opt = pca_opt.transform(diabetes_test_x)
    logging.info(f"Applying best algorithm: {best_algo} with k={k}")
    if best_algo == 'K-means':
        test_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit(test_pca_opt).labels_
    elif best_algo == 'Hierarchical':
        test_labels = AgglomerativeClustering(n_clusters=k).fit_predict(test_pca_opt)
    elif best_algo == 'DBSCAN':
        nn = NearestNeighbors(n_neighbors=min(10, len(test_pca_opt) - 1)).fit(test_pca_opt)
        d = np.sort(nn.kneighbors(test_pca_opt)[0][:, -1])
        eps = d[np.argmax(np.diff(np.diff(d))) + 1]
        test_labels = DBSCAN(eps=eps, min_samples=5).fit_predict(test_pca_opt)
    else:
        test_labels = GaussianMixture(n_components=k, random_state=42).fit_predict(test_pca_opt)
    if len(np.unique(test_labels)) > 1:
        test_sil = silhouette_score(test_pca_opt, test_labels)
        logging.info(f"Test set silhouette score: {test_sil:.4f}")
    else:
        test_sil = 0
    plot_best_clustering_result(test_pca_2d, test_labels, f"{best_algo} (Test Set)", test_sil, diabetes_out)
    if eval_res['best_method'] == 'UMAP + Best Algorithm':
        logging.info("Applying best UMAP configuration to test set")
        reducer = umap.UMAP(
            n_neighbors=umap_res['best_n_neighbors'],
            min_dist=umap_res['best_min_dist'],
            n_components=umap_res['best_n_components'],
            random_state=42
        )
        test_umap = reducer.fit_transform(diabetes_test_x)
        if best_algo == 'K-means':
            model = KMeans(n_clusters=umap_res['best_n_clusters'], random_state=42, n_init=10)
        elif best_algo == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=umap_res['best_n_clusters'])
        elif best_algo == 'DBSCAN':
            nn = NearestNeighbors(n_neighbors=min(5, len(test_umap) - 1)).fit(test_umap)
            d = np.sort(nn.kneighbors(test_umap)[0][:, -1])
            eps = d[np.argmax(np.diff(np.diff(d))) + 1]
            model = DBSCAN(eps=eps, min_samples=5)
        else:
            model = GaussianMixture(n_components=umap_res['best_n_clusters'], random_state=42)
        test_umap_labels = model.fit_predict(test_umap)
        if len(np.unique(test_umap_labels)) > 1:
            test_umap_sil = silhouette_score(test_umap, test_umap_labels)
            logging.info(f"Test set UMAP silhouette score: {test_umap_sil:.4f}")
        else:
            test_umap_sil = 0
        if test_umap_sil > test_sil:
            test_labels = test_umap_labels
            test_sil = test_umap_sil
            logging.info("Using UMAP clustering results for test set (better silhouette score)")
    test_df = diabetes_test_df.copy()
    test_df['cluster'] = test_labels
    if len(num_cols + cat_cols) >= 3:
        plot_top_features_pairplot(test_df, num_cols[:min(3, len(num_cols))], diabetes_out)
    test_stats = {col: test_df.groupby('cluster')[col].agg(['mean', 'std', 'min', 'max']) for col in num_cols}
    comp = {
        'test_silhouette': test_sil,
        'training_silhouette': cluster_res['best_algorithm_silhouette'],
        'difference': test_sil - cluster_res['best_algorithm_silhouette']
    }
    logging.info(f"Silhouette score: Training: {cluster_res['best_algorithm_silhouette']:.4f}, Test: {test_sil:.4f}, Difference: {comp['difference']:.4f}")
    test_df.to_csv(os.path.join(diabetes_out, 'test_set_with_clusters.csv'), index=False)
    return {
        'test_labels': test_labels,
        'test_silhouette': test_sil,
        'test_cluster_stats': test_stats,
        'train_test_comparison': comp,
        'df_test_with_clusters': test_df
    }