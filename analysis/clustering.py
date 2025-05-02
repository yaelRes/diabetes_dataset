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


def compare_clustering_algorithms(x_processed, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    clustering = ["KMeans", "AgglomerativeClustering", "GaussianMixture", "GaussianMixture", "HDBSCAN"]
    reduction = ["PCA", "TSNE", "UMAP"]
    results = {r: {c: [] for c in clustering} for r in reduction}

    max_tsne_silhouette_score = 0
    n_components_options = TSNE_CONFIG["n_components_options"]
    perplexity_options = TSNE_CONFIG["perplexity_options"]
    for n_components in n_components_options:
        for perplexity in perplexity_options:
            tsne_filename = os.path.join(output_dir, f"x_tsne_n{n_components}_p{perplexity}.pkl")
            with open(tsne_filename, "rb") as f:
                x_tsne = pickle.load(f)
                ext_data = {"perplexity": perplexity}
                ext_str = f"perplexity:{perplexity}"
                max_tsne_silhouette_score = cluster_reduced_parameters_data(max_tsne_silhouette_score, n_components,
                                                                            ext_data, ext_str,
                                                                            x_tsne, "TSNE", results)
    logging.info(f"max_tsne_silhouette_score={max_tsne_silhouette_score} ")
    save_results_to_csv(results, output_dir, "TSNE")

    max_umap_silhouette_score = 0
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
                    max_umap_silhouette_score = cluster_reduced_parameters_data(max_umap_silhouette_score, n_components,
                                                                                ext_data, ext_str, x_umap, "UMAP",
                                                                                results)

    logging.info(f"max_umap_silhouette_score={max_umap_silhouette_score} ")
    save_results_to_csv(results, output_dir, "UMAP")
    max_pca_silhouette_score = 0
    n_components_list = PCA_CONFIG["n_components_range"]
    n_components_list_reduced = [nc for nc in n_components_list if nc is None or nc <= x_processed.shape[1]]
    diff = set(n_components_list) - set(n_components_list_reduced)
    for n_components in n_components_list_reduced:
        pca_filename = os.path.join(output_dir, f"x_pca_n{n_components}.pkl")
        with open(pca_filename, "rb") as f:
            x_pca = pickle.load(f)
            ext_data = {}
            ext_str = ""
            max_pca_silhouette_score = cluster_reduced_parameters_data(max_pca_silhouette_score, n_components, ext_data,
                                                                       ext_str,
                                                                       x_pca, "PCA", results)

    logging.info(f"max_pca_silhouette_score={max_pca_silhouette_score} ")
    save_results_to_csv(results, output_dir, "PCA")


def save_results_to_csv(results, output_dir, reduction_method_name=None):
    csv_dir = os.path.join(output_dir, "clustering_csv_results")
    labels_dir = os.path.join(output_dir, "labels_pkl")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for reduction_method in results:
        if reduction_method_name is not None and reduction_method != reduction_method_name:
            continue
        logging.info(f"processing {reduction_method} results")

        reduction_dfs = {}

        for clustering_algo in results[reduction_method]:
            algo_results = results[reduction_method][clustering_algo]

            if not algo_results:
                logging.warning(f"No results for {clustering_algo} with {reduction_method}")
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
            reduction_dfs[clustering_algo] = df

            filename = f"{reduction_method}_{clustering_algo}_results.csv"
            csv_path = os.path.join(csv_dir, filename)
            df.to_csv(csv_path, index=False)
            logging.info(f"saved {csv_path}")

        if reduction_dfs:
            combined_df = pd.concat(reduction_dfs.values(), ignore_index=True)
            combined_filename = f"{reduction_method}_all_results.csv"
            combined_csv_path = os.path.join(csv_dir, combined_filename)
            combined_df.to_csv(combined_csv_path, index=False)
            logging.info(f"saved combined results to {combined_csv_path}")

    logging.info(f"all results saved to csv files in {csv_dir}")


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
                         f"Silhouette={kmeans_silhouette:.4f} loss={kmeans.inertia_}")
            results[reduction_algo]["KMeans"].append({
                'n_component': n_components,
                'k': k,
                'score': kmeans_silhouette,
                'loss': kmeans.inertia_,
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
        return max_silhouette_score


def dbscan_cluster(x_data, output_dir="output"):
    nn = NearestNeighbors(n_neighbors=min(10, len(x_data) - 1))
    nn.fit(x_data)
    distances, indices = nn.kneighbors(x_data)
    distances = np.sort(distances[:, -1])

    knee_point = np.diff(np.diff(distances))
    elbow_index = np.argmax(knee_point) + 1
    eps_value = distances[elbow_index]
    logging.info(f"find DBSCAN eps value: {eps_value:.4f}")

    plot_dbscan_kdistance_graph(distances, elbow_index, output_dir)

    dbscan = DBSCAN(eps=eps_value, min_samples=5)
    dbscan_labels = dbscan.fit_predict(x_data)

    if len(np.unique(dbscan_labels)) <= 1 or -1 in np.unique(dbscan_labels):
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

    if len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels):
        dbscan_silhouette = silhouette_score(x_data, dbscan_labels)
    elif len(np.unique(dbscan_labels)) > 1:
        non_noise = dbscan_labels != -1
        if np.sum(non_noise) > 1 and len(np.unique(dbscan_labels[non_noise])) > 1:
            dbscan_silhouette = silhouette_score(x_data[non_noise], dbscan_labels[non_noise])
        else:
            dbscan_silhouette = 0
    else:
        dbscan_silhouette = 0
    logging.info(f"DBSCAN silhouette : {dbscan_silhouette:.4f}")


