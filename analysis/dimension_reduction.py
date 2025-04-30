import logging
import os
import pickle

import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import UMAP_CONFIG, TSNE_CONFIG, PCA_CONFIG
from visualization.dimension_reduction import plot_dimension_reduction_comparison


def pca_dim_reduction_to_pkl(x_processed, output_dir="output"):
    n_components_list = PCA_CONFIG["n_components_range"]
    n_components_list_reduced = [nc for nc in n_components_list if nc <= x_processed.shape[1]]
    diff = set(n_components_list) - set(n_components_list_reduced)
    logging.warning(f"pca didn't reduced to {diff} since it is bigger than the original {x_processed.shape[1]}")
    total_iterations = len(n_components_list)
    for progress_count, n_components in enumerate(n_components_list):
        logging.info(f"PCA progress: {progress_count}/{total_iterations} "
                     f"(n_components={n_components}")
        pca_filename = os.path.join(output_dir, f"x_pca_n{n_components}.pkl")
        if os.path.exists(pca_filename):
            logging.info(f"file {pca_filename} already exist")
            #with open(pca_filename, "rb") as f:
            #    x_pca = pickle.load(f)
        else:
            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(x_processed)
            if isinstance(n_components, float):
                actual_components = x_pca.shape[1]
                logging.info(f"using {actual_components} components to preserve {n_components * 100:.0f}% of variance")
            else:
                actual_components = n_components
                logging.info(f"using {actual_components} components")
            with open(pca_filename, "wb") as f:
                pickle.dump(x_pca, f)


def tsne_dim_reduction_to_pkl(x_processed, output_dir="output"):
    n_components_options = TSNE_CONFIG["n_components_options"]
    perplexity_options = TSNE_CONFIG["perplexity_options"]
    random_state = TSNE_CONFIG["random_state"]
    total_iterations = len(n_components_options) * len(perplexity_options)
    progress_count = 0
    for n_components in n_components_options:
        for perplexity in perplexity_options:
            progress_count += 1
            tsne_filename = os.path.join(output_dir, f"x_tsne_n{n_components}_p{perplexity}.pkl")
            logging.info(f"TSNE progress: {progress_count}/{total_iterations} "
                         f"(n_components={n_components}, perplexity={perplexity}")
            if os.path.exists(tsne_filename):
                logging.info(f"file {tsne_filename} already exist")
                # with open(tsne_filename, "rb") as f:
                #    x_tsne = pickle.load(f)
            else:
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=random_state,
                    n_iter=1000
                )
                x_tsne = tsne.fit_transform(x_processed)
                with open(tsne_filename, "wb") as f:
                    pickle.dump(x_tsne, f)


def umap_dim_reduction_to_pkl(x_processed, output_dir="output"):
    n_neighbors_options = UMAP_CONFIG["n_neighbors_options"]
    min_dist_options = UMAP_CONFIG["min_dist_options"]
    n_components_options = UMAP_CONFIG["n_components_options"]
    random_state = UMAP_CONFIG["random_state"]
    total_iterations = len(n_components_options) * len(n_neighbors_options) * len(min_dist_options)
    progress_count = 0
    for n_components in n_components_options:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                umap_filename = os.path.join(output_dir, f"x_umap_n{n_components}_nn{n_neighbors}_md{min_dist}.pkl")
                progress_count += 1
                logging.info(f"umap progress: {progress_count}/{total_iterations} "
                             f"(n_components={n_components}"
                             f"n_neighbors={n_neighbors}, min_dist={min_dist:.2f})")
                if os.path.exists(umap_filename):
                    logging.info(f"file {umap_filename} already exist")
                    # with open(umap_filename, "rb") as f:
                    #    x_umap = pickle.load(f)
                else:
                    umap_reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        random_state=random_state
                    )
                    x_umap = umap_reducer.fit_transform(x_processed)
                    with open(umap_filename, "wb") as f:
                        pickle.dump(x_umap, f)


def create_dimension_reduction_images(x_processed, output_dir="output"):
    logging.info("creating dimension reduction images")
    os.makedirs(output_dir, exist_ok=True)

    n_components = 2
    logging.info(f"PCA parameters: n_components={n_components}")
    pca_filename = os.path.join(output_dir, f"x_pca_n{n_components}.pkl")
    explained_variance_filename = os.path.join(output_dir, f"x_pca_n{n_components}_explained_variance_ratio.pkl")
    if os.path.exists(pca_filename) and os.path.exists(explained_variance_filename):
        with open(pca_filename, "rb") as f:
            x_pca_2d = pickle.load(f)
        logging.info(f"loading PCA results from {pca_filename}")
        with open(explained_variance_filename, "rb") as f:
            explained_variance = pickle.load(f)
        logging.info(f"loaded PCA n_components={n_components} explained variance: {explained_variance}")
    else:
        pca_2d = PCA(n_components=2)
        x_pca_2d = pca_2d.fit_transform(x_processed)
        explained_variance = pca_2d.explained_variance_ratio_
        logging.info(f"PCA 2D explained variance: {explained_variance}")
        with open(pca_filename, "wb") as f:
            pickle.dump(x_pca_2d, f)
        with open(explained_variance_filename, "wb") as f:
            pickle.dump(explained_variance, f)

    perplexity = 30
    logging.info(f"t-SNE parameters: n_components={n_components}, perplexity={perplexity}")
    tsne_filename = os.path.join(output_dir, f"x_tsne_n{n_components}_p{perplexity}.pkl")
    if os.path.exists(tsne_filename):
        with open(tsne_filename, "rb") as f:
            x_tsne_2d = pickle.load(f)
        logging.info(f"loading t-SNE results from {tsne_filename}")
    else:
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        x_tsne_2d = tsne.fit_transform(x_processed)
        with open(tsne_filename, "wb") as f:
            pickle.dump(x_tsne_2d, f)

    n_neighbors = 15
    min_dist = 0.1
    logging.info(f"UMAP parameters: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    umap_filename = os.path.join(output_dir, f"x_umap_n{n_components}_nn{n_neighbors}_md{min_dist}.pkl")
    if os.path.exists(umap_filename):
        with open(umap_filename, "rb") as f:
            x_umap_2d = pickle.load(f)
    else:
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        x_umap_2d = umap_reducer.fit_transform(x_processed)
        with open(umap_filename, "wb") as f:
            pickle.dump(x_umap_2d, f)

    plot_dimension_reduction_comparison(
        x_pca_2d,
        x_tsne_2d,
        x_umap_2d,
        explained_variance,
        output_dir
    )
