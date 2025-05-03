import logging
import os
import pickle
import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from config import PCA_CONFIG, CLUSTERING_CONFIG, TSNE_CONFIG, UMAP_CONFIG
from visualization.dimension_reduction import plot_dbscan_kdistance_graph

def diabetes_compare_clustering(diabetes_x, diabetes_out="output"):
    os.makedirs(diabetes_out, exist_ok=True)
    clustering = ["KMeans", "AgglomerativeClustering", "GaussianMixture", "GaussianMixture", "HDBSCAN"]
    reduction = ["PCA", "TSNE", "UMAP"]
    results = {r: {c: [] for c in clustering} for r in reduction}
    max_tsne = 0
    for n in TSNE_CONFIG["n_components_options"]:
        for p in TSNE_CONFIG["perplexity_options"]:
            tsne_file = os.path.join(diabetes_out, f"x_tsne_n{n}_p{p}.pkl")
            with open(tsne_file, "rb") as f:
                x_tsne = pickle.load(f)
                ext_data = {"perplexity": p}
                ext_str = f"perplexity:{p}"
                max_tsne = diabetes_cluster_reduced(max_tsne, n, ext_data, ext_str, x_tsne, "TSNE", results)
    logging.info(f"max_tsne_silhouette_score={max_tsne} ")
    diabetes_save_clustering(results, diabetes_out, "TSNE")
    max_umap = 0
    for n in UMAP_CONFIG["n_components_options"]:
        for nn in UMAP_CONFIG["n_neighbors_options"]:
            for md in UMAP_CONFIG["min_dist_options"]:
                umap_file = os.path.join(diabetes_out, f"x_umap_n{n}_nn{nn}_md{md}.pkl")
                with open(umap_file, "rb") as f:
                    x_umap = pickle.load(f)
                    ext_data = {"n_neighbors": nn, "min_dist": md}
                    ext_str = f"n_neighbors: {nn} min_dist:{md}"
                    max_umap = diabetes_cluster_reduced(max_umap, n, ext_data, ext_str, x_umap, "UMAP", results)
    logging.info(f"max_umap_silhouette_score={max_umap} ")
    diabetes_save_clustering(results, diabetes_out, "UMAP")
    max_pca = 0
    n_list = [nc for nc in PCA_CONFIG["n_components_range"] if nc is None or nc <= diabetes_x.shape[1]]
    for n in n_list:
        pca_file = os.path.join(diabetes_out, f"x_pca_n{n}.pkl")
        with open(pca_file, "rb") as f:
            x_pca = pickle.load(f)
            ext_data = {}
            ext_str = ""
            max_pca = diabetes_cluster_reduced(max_pca, n, ext_data, ext_str, x_pca, "PCA", results)
    logging.info(f"max_pca_silhouette_score={max_pca} ")
    diabetes_save_clustering(results, diabetes_out, "PCA")

def diabetes_save_clustering(results, diabetes_out, reduction_method_name=None):
    csv_dir = os.path.join(diabetes_out, "clustering_csv_results")
    labels_dir = os.path.join(diabetes_out, "labels_pkl")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for reduction_method in results:
        if reduction_method_name and reduction_method != reduction_method_name:
            continue
        for clustering_algo in results[reduction_method]:
            algo_results = results[reduction_method][clustering_algo]
            if not algo_results:
                continue
            rows = []
            for i, result in enumerate(algo_results):
                row = {
                    'reduction_method': reduction_method,
                    'clustering_algorithm': clustering_algo,
                    'n_component': result['n_component'],
                    'k': result['k']
                }
                if clustering_algo == "HDBSCAN":
                    row['score_with_noise'] = result['score_with_noise']
                    row['score_without_noise'] = result['score_with_no_noise']
                    row['min_samples'] = result['min_samples']
                else:
                    row['score'] = result['score']
                    if clustering_algo == "KMeans":
                        row['loss'] = result['loss']
                if 'ext_data' in result and result['ext_data']:
                    for key, value in result['ext_data'].items():
                        row[key] = value
                if 'labels' in result:
                    params_str = f"n{result['n_component']}_k{result['k']}"
                    if clustering_algo == "HDBSCAN":
                        params_str += f"_ms{result['min_samples']}"
                    if 'ext_data' in result and result['ext_data']:
                        for key, value in result['ext_data'].items():
                            val_str = str(value).replace('.', 'p')
                            params_str += f"_{key[:2]}{val_str}"
                    labels_filename = f"{reduction_method}_{clustering_algo}_{params_str}_labels.pkl"
                    labels_path = os.path.join(labels_dir, labels_filename)
                    with open(labels_path, 'wb') as f:
                        pickle.dump(result['labels'], f)
                    row['labels_file'] = labels_path
                rows.append(row)
            df = pd.DataFrame(rows)
            filename = f"{reduction_method}_{clustering_algo}_results.csv"
            df.to_csv(os.path.join(csv_dir, filename), index=False)
    
    for reduction_method in results:
        reduction_dfs = [pd.read_csv(os.path.join(csv_dir, f"{reduction_method}_{algo}_results.csv")) for algo in results[reduction_method] if os.path.exists(os.path.join(csv_dir, f"{reduction_method}_{algo}_results.csv"))]
        if reduction_dfs:
            combined_df = pd.concat(reduction_dfs, ignore_index=True)
            combined_filename = f"{reduction_method}_all_results.csv"
            combined_csv_path = os.path.join(csv_dir, combined_filename)
            combined_df.to_csv(combined_csv_path, index=False)
    logging.info(f"all results saved to csv files in {csv_dir}")

def diabetes_cluster_reduced(max_sil, n_comp, ext_data, ext_str, x, red_algo, results):
    k_range = CLUSTERING_CONFIG["k_range"]
    hdbscan_min_samples_list = CLUSTERING_CONFIG["hdbscan_min_samples"]
    random_state = CLUSTERING_CONFIG["random_state"]
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=500, n_init=10)
        kmeans_labels = kmeans.fit_predict(x)
        kmeans_sil = silhouette_score(x, kmeans_labels)
        results[red_algo]["KMeans"].append({
            'n_component': n_comp,
            'k': k,
            'score': kmeans_sil,
            'loss': kmeans.inertia_,
            'ext_data': ext_data,
            'labels': kmeans_labels
        })
        max_sil = max(max_sil, kmeans_sil)
        hierarchical = AgglomerativeClustering(n_clusters=k)
        hierarchical_labels = hierarchical.fit_predict(x)
        hierarchical_sil = silhouette_score(x, hierarchical_labels)
        results[red_algo]["AgglomerativeClustering"].append({
            'n_component': n_comp,
            'k': k,
            'score': hierarchical_sil,
            'ext_data': ext_data,
            'labels': hierarchical_labels
        })
        max_sil = max(max_sil, hierarchical_sil)
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm_labels = gmm.fit_predict(x)
        gmm_sil = silhouette_score(x, gmm_labels)
        results[red_algo]["GaussianMixture"].append({
            'n_component': n_comp,
            'k': k,
            'score': gmm_sil,
            'ext_data': ext_data,
            'labels': gmm_labels
        })
        max_sil = max(max_sil, gmm_sil)
        for min_samples in hdbscan_min_samples_list:
            hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=k, min_samples=min_samples)
            hdbscan_labels = hdbscan_cluster.fit_predict(x)
            hdbscan_sil = silhouette_score(x, hdbscan_labels)
            max_sil = max(max_sil, hdbscan_sil)
            mask = hdbscan_labels != -1
            X_no_noise = x[mask]
            labels_no_noise = hdbscan_labels[mask]
            silhouette_avg = 0
            if len(np.unique(labels_no_noise)) > 1:
                silhouette_avg = silhouette_score(X_no_noise, labels_no_noise)
                max_sil = max(max_sil, silhouette_avg)
            results[red_algo]["HDBSCAN"].append({
                'n_component': n_comp,
                'k': k,
                'min_samples': min_samples,
                'score_with_noise': hdbscan_sil,
                'score_with_no_noise': silhouette_avg,
                'ext_data': ext_data,
                'labels': hdbscan_labels
            })
            max_sil = max(max_sil, hdbscan_sil)
    return max_sil

def diabetes_dbscan_cluster(x, diabetes_out="output"):
    nn = NearestNeighbors(n_neighbors=min(10, len(x) - 1)).fit(x)
    d = np.sort(nn.kneighbors(x)[0][:, -1])
    knee_point = np.diff(np.diff(d))
    elbow_index = np.argmax(knee_point) + 1
    eps = d[elbow_index]
    logging.info(f"find DBSCAN eps value: {eps:.4f}")
    plot_dbscan_kdistance_graph(d, elbow_index, diabetes_out)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(x)
    if len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels):
        dbscan_sil = silhouette_score(x, dbscan_labels)
    elif len(np.unique(dbscan_labels)) > 1:
        non_noise = dbscan_labels != -1
        if np.sum(non_noise) > 1 and len(np.unique(dbscan_labels[non_noise])) > 1:
            dbscan_sil = silhouette_score(x[non_noise], dbscan_labels[non_noise])
        else:
            dbscan_sil = 0
    else:
        dbscan_sil = 0
    logging.info(f"DBSCAN silhouette : {dbscan_sil:.4f}")


