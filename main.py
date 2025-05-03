import json
import logging
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from analysis.clustering import diabetes_compare_clustering
from analysis.dimension_reduction import \
    create_dimension_reduction_images, pca_dim_reduction_to_pkl, tsne_dim_reduction_to_pkl, umap_dim_reduction_to_pkl
from config import feature_sets
from utils.data_utils import load_dataset, get_column_types
from utils.data_utils import split_train_test
from utils.logging_utils import setup_logging
from utils.weighted_preprocess import preprocess_data, create_diabetes_features
from visualization.dimension_reduction import plot_pca_explained_variance
from visualization.comparison_viz import create_comparative_visualizations


def save_parameters_to_file(params, filename="parameters.json"):
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)
    logging.info(f"Parameters saved to {filename}")


def load_parameters_from_file(filename="parameters.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        return params
    else:
        return None


def run_analysis(data_path="diabetes_dataset.csv", output_dir="output/", selected_features=None, test_size=0.2,
                sample_ratio=1.0, add_diabetes_cols=False, inc_marker_weight=False, weight=None):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    np.random.seed(42)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    train_csv = os.path.join(train_dir, "df_train.csv")
    test_csv = os.path.join(test_dir, "df_test.csv")
    train_pkl = os.path.join(train_dir, "x_processed_train.pkl")
    test_pkl = os.path.join(test_dir, "x_processed_test.pkl")
    main_params = os.path.join(output_dir, "main_parameters.json")
    params = {
        "data_path": data_path,
        "output_dir": output_dir,
        "selected_features": selected_features,
        "test_size": test_size,
        "sample_ratio": sample_ratio,
        "add_diabetes_cols": add_diabetes_cols,
        "inc_marker_weight": inc_marker_weight,
        "weight": weight
    }
    df = load_dataset(data_path)
    saved_params = load_parameters_from_file(main_params)
    if saved_params is not None and saved_params == params:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        x_train = pd.read_pickle(train_pkl)
        x_test = pd.read_pickle(test_pkl)
    else:
        if sample_ratio < 1.0:
            df = df.sample(n=int(len(df) * sample_ratio), random_state=42)
        if add_diabetes_cols:
            df_ext = create_diabetes_features(df)
        else:
            df_ext = df
        if selected_features is not None and add_diabetes_cols:
            new_cols = ['is_diabetic_HbA1c', 'is_diabetic_FBG', 'diabetes_score']
            sel_ext = selected_features.copy()
            sel_ext.extend([f for f in new_cols if f in df_ext.columns])
        else:
            sel_ext = selected_features
        cat_cols, num_cols = get_column_types(df_ext, sel_ext)
        df_train, df_test = split_train_test(df_ext, test_size=test_size, random_state=42)
        df_train.to_csv(train_csv, index=False, header=True)
        df_test.to_csv(test_csv, index=False, header=True)
        if inc_marker_weight and weight:
            markers = [m for m in ['HbA1c', 'Fasting_Blood_Glucose'] if m in num_cols]
        else:
            markers = []
            weight = None
        x_train, preproc = preprocess_data(
            df=df_train,
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
            diabetes_markers=markers,
            weight=weight
        )
        with open(train_pkl, "wb") as f:
            pickle.dump(x_train, f)
        x_test = preproc.transform(df_test[num_cols + cat_cols])
        with open(test_pkl, "wb") as f:
            pickle.dump(x_test, f)
    dim_res = create_dimension_reduction_images(x_train, train_dir)
    pca = PCA(random_state=42)
    pca.fit(x_train)
    plot_pca_explained_variance(pca.explained_variance_ratio_, train_dir)
    pca_dim_reduction_to_pkl(x_train, train_dir)
    tsne_dim_reduction_to_pkl(x_train, train_dir)
    umap_dim_reduction_to_pkl(x_train, train_dir)
    clust = diabetes_compare_clustering(x_train, train_dir)
    return {
        'df_train': df_train,
        'df_test': df_test,
        'x_train_processed': x_train,
        'x_test_processed': x_test,
        'dim_red_result': dim_res,
        'clustering_result': clust,
    }


def main():
    data_path = "diabetes_dataset.csv"
    test_size = 0.2
    sample_ratio = 1.0
    from config import feature_sets
    run_comparative_analysis(
        data_path=data_path,
        feature_sets=feature_sets,
        test_size=test_size,
        sample_ratio=sample_ratio,
        recieved_dir=None
    )


def run_comparative_analysis(data_path="diabetes_dataset.csv",
                             feature_sets=None, test_size=0.2, sample_ratio=1.0, recieved_dir=None):
    if recieved_dir:
        comp_dir = recieved_dir
    else:
        comp_dir = f"output_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(comp_dir, exist_ok=True)
    results = {}
    summary = {}
    for name, feats in feature_sets.items():
        feat_dir = os.path.join(comp_dir, name.replace(" ", "_"))
        os.makedirs(feat_dir, exist_ok=True)
        res = run_analysis(
            data_path=data_path,
            output_dir=feat_dir,
            selected_features=feats,
            test_size=test_size,
            sample_ratio=sample_ratio
        )
        results[name] = res
    try:
        if dfs:
            create_comparative_visualizations(dfs, comp_dir)
    except:
        pass
    return {
        'comparative_results': results,
        'test_train_summary': summary
    }


if __name__ == "__main__":
    main()
