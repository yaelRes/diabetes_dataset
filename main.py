import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA

from analysis.clustering import compare_clustering_algorithms
from analysis.dimension_reduction import \
    create_dimension_reduction_images, pca_dim_reduction_to_pkl, tsne_dim_reduction_to_pkl, umap_dim_reduction_to_pkl
from config import feature_sets
from utils.data_utils import load_dataset, get_column_types
from utils.data_utils import split_train_test
from utils.logging_utils import setup_logging
from utils.weighted_preprocess import preprocess_data, create_diabetes_features
from visualization.dimension_reduction import plot_pca_explained_variance


def save_parameters_to_file(params, filename="parameters.json"):
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {filename}")


def load_parameters_from_file(filename="parameters.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        return params
    else:
        return None


def main(data_path="diabetes_dataset.csv", output_dir="output/", selected_features=None, test_size=0.2,
         sample_ratio=1.0, add_diabetes_columns=False, increase_diabetes_markers_weight=False, weight=None):
    os.makedirs(output_dir, exist_ok=True)

    log_file = setup_logging(output_dir)
    np.random.seed(42)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    df_train_filename = os.path.join(train_dir, "df_train.csv")
    df_test_filename = os.path.join(test_dir, "df_test.csv")
    x_train_processed_filename = os.path.join(train_dir, "x_processed_train.pkl")
    x_test_processed_filename = os.path.join(test_dir, "x_processed_test.pkl")

    main_params = os.path.join(output_dir, "main_parameters.json")

    params = {
        "data_path": data_path,
        "output_dir": output_dir,
        "selected_features": selected_features,
        "test_size": test_size,
        "sample_ratio": sample_ratio,
        "add_diabetes_columns": add_diabetes_columns,
        "increase_diabetes_markers_weight": increase_diabetes_markers_weight,
        "weight": weight
    }

    logging.info(f"start diabetes clustering analysis the log is saved in {log_file}")

    if selected_features:
        logging.info(f"use selected features: {selected_features}")
    else:
        logging.info("use all available features")

    if sample_ratio < 1.0:
        logging.info(f"WARNING: use {sample_ratio * 100:.1f}% of the dataset for faster processing")

    df = load_dataset(data_path)
    saved_params = load_parameters_from_file(main_params)
    if saved_params is not None and saved_params == params:
        if (os.path.exists(df_train_filename) and os.path.exists(df_test_filename)
                and os.path.exists(x_train_processed_filename) and
                os.path.exists(x_test_processed_filename)):
            logging.info(f"loading saved files:\n"
                         f"{df_train_filename}\n"
                         f"\n{df_test_filename}\n"
                         f"{x_train_processed_filename}\n"
                         f"{x_test_processed_filename}")

            df_train = pd.read_csv(df_train_filename)
            df_test = pd.read_csv(df_test_filename)
            x_train_processed = pd.read_pickle(x_train_processed_filename)
            x_test_processed = pd.read_pickle(x_test_processed_filename)
            logging.info(f"loading processed test data shape: {x_test_processed.shape}")
    else:
        if sample_ratio < 1.0:
            original_size = len(df)
            sample_size = int(original_size * sample_ratio)
            df = df.sample(n=sample_size, random_state=42)
            logging.info(f"sample dataset from {original_size} to {len(df)} samples ({sample_ratio * 100:.1f}%)")

        if add_diabetes_columns:
            df_extended = create_diabetes_features(df)
        else:
            df_extended = df

        if selected_features is not None and add_diabetes_columns is True:
            new_diabetes_features = ['is_diabetic_HbA1c', 'is_diabetic_FBG', 'diabetes_score']
            selected_features_extended = selected_features.copy()
            selected_features_extended.extend([f for f in new_diabetes_features if f in df_extended.columns])
            logging.info(f"extended the selected features: {selected_features_extended}")
        else:
            selected_features_extended = selected_features

        categorical_cols, numerical_cols = get_column_types(df_extended, selected_features_extended)

        df_train, df_test = split_train_test(df_extended, test_size=test_size, random_state=42)
        logging.info(f"split data : {len(df_train)} training samples, {len(df_test)} test samples")

        df_train.to_csv(df_train_filename, index=False, header=True)
        df_test.to_csv(df_test_filename, index=False, header=True)

        if increase_diabetes_markers_weight and weight:
            diabetes_markers = [
                'HbA1c', 'Fasting_Blood_Glucose',
            ]
            diabetes_markers = [marker for marker in diabetes_markers if marker in numerical_cols]
        else:
            diabetes_markers = []
            weight = None

        x_train_processed, preprocessor = preprocess_data(
            df=df_train,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            diabetes_markers=diabetes_markers,
            weight=weight
        )
        with open(x_train_processed_filename, "wb") as f:
            pickle.dump(x_train_processed, f)

        logging.info(f"processed training data shape: {x_train_processed.shape}")

        x_test = df_test[numerical_cols + categorical_cols]
        x_test_processed = preprocessor.transform(x_test)

        with open(x_test_processed_filename, "wb") as f:
            pickle.dump(x_test_processed, f)

        logging.info(f"Processed test data shape: {x_test_processed.shape}")

    dim_red_result = create_dimension_reduction_images(x_train_processed, train_dir)

    pca_full = PCA(random_state=42)
    pca_full.fit(x_train_processed)
    plot_pca_explained_variance(pca_full.explained_variance_ratio_, train_dir)

    pca_dim_reduction_to_pkl(x_train_processed, train_dir)
    tsne_dim_reduction_to_pkl(x_train_processed, train_dir)
    umap_dim_reduction_to_pkl(x_train_processed, train_dir)

    clustering_result = compare_clustering_algorithms(
        x_train_processed,
        train_dir
    )

    # anomaly_result = perform_anomaly_detection(
    #     clustering_result['X_pca_optimal'],
    #     grid_search_result['optimal_k'],
    #     train_dir
    # )
    #
    # cluster_analysis = analyze_cluster_characteristics(
    #     df_train,
    #     clustering_result['best_algorithm_labels'],
    #     numerical_cols,
    #     categorical_cols,
    #     train_dir
    # )
    #
    # feature_importance = analyze_feature_importance(
    #     df_train,
    #     clustering_result['best_algorithm_labels'],
    #     numerical_cols,
    #     categorical_cols,
    #     preprocessor,
    #     train_dir
    # )
    #
    # umap_result = optimize_umap_parameters(x_train_processed, clustering_result['best_algorithm'], train_dir)
    #
    # tsne_result = optimize_tsne_parameters(x_train_processed, clustering_result['best_algorithm'], train_dir)
    #
    # eval_result = final_evaluation(pca_result, clustering_result, umap_result, tsne_result, x_train_processed,
    #                                train_dir)
    #
    # if eval_result['best_method'] == 'UMAP + Best Algorithm':
    #     final_labels_train = umap_result['best_umap_labels']
    # else:
    #     final_labels_train = clustering_result['best_algorithm_labels']
    #
    # profile_result = generate_cluster_profiles(
    #     df_train,
    #     final_labels_train,
    #     numerical_cols,
    #     categorical_cols,
    #     train_dir
    # )
    #
    # test_evaluation = evaluate_on_test_set(
    #     df_test,
    #     x_test_processed,
    #     clustering_result,
    #     pca_result,
    #     umap_result,
    #     eval_result,
    #     preprocessor,
    #     numerical_cols,
    #     categorical_cols,
    #     test_dir
    # )

    #logging.info("Analysis complete!")

    # except Exception as e:
    #     logging.error(f"Error during analysis: {e}")
    #     import traceback
    #     logging.error(traceback.format_exc())
    #     raise


def run_comparative_analysis(data_path="diabetes_dataset.csv",
                             feature_sets=None, test_size=0.2, sample_ratio=1.0, recieved_dir=None):
    if recieved_dir:
        comparative_dir = recieved_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparative_dir = f"pics_comparative_{timestamp}"
        os.makedirs(comparative_dir, exist_ok=True)

    logging.info(f"Starting comparative analysis with {len(feature_sets)} feature sets")

    if sample_ratio < 1.0:
        logging.info(f"Using {sample_ratio * 100:.1f}% of the dataset for faster processing")

    # Store results for each feature set
    comparative_results = {}
    test_train_summary = {}

    for feature_set_name, selected_features in feature_sets.items():
        logging.info(f"Analyzing feature set: {feature_set_name}")

        # Create output directory for this feature set
        feature_set_dir = os.path.join(comparative_dir, feature_set_name.replace(" ", "_"))
        os.makedirs(feature_set_dir, exist_ok=True)

        try:
            # Run the main analysis with this feature set
            main(
                data_path=data_path,
                output_dir=feature_set_dir,
                selected_features=selected_features,
                test_size=test_size,
                sample_ratio=sample_ratio  # Pass the sample ratio to main
            )

        except Exception as e:
            logging.error(f"Error analyzing feature set {feature_set_name}: {e}")
            import traceback
            logging.error(traceback.format_exc())

    # Create comparative visualizations
    try:
        from visualization.comparison_viz import create_comparative_visualizations

        # Extract metric dataframes from each feature set
        metrics_dfs = {}
        for feature_set_name, result in comparative_results.items():
            if 'eval_result' in result and 'metrics_df' in result['eval_result']:
                metrics_dfs[feature_set_name] = result['eval_result']['metrics_df']

        # Create combined visualizations
        if metrics_dfs:
            create_comparative_visualizations(metrics_dfs, comparative_dir)

    except Exception as e:
        logging.error(f"Error creating comparative visualizations: {e}")


if __name__ == "__main__":

    from visualization.comparison_viz import create_train_test_visualizations
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(description='Run diabetes clustering analysis with train/test split.')
    parser.add_argument('--data_path', type=str, default="diabetes_dataset.csv", help='Path to the dataset CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--sample_ratio', type=float, default=1,
                        help='Proportion of the dataset to use (for faster runs)')
    args = parser.parse_args()

    log_file = setup_logging()
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created 'pics' directory")
    else:
        print("'pics' directory already exists")

    results = run_comparative_analysis(
        data_path=args.data_path,
        feature_sets=feature_sets,
        test_size=args.test_size,
        sample_ratio=args.sample_ratio,
        recieved_dir=None
    )

    print("Analysis completed successfully!")
    print(f"All visualizations saved to the 'pics_comparative_*' folder")

    if 'test_train_summary' in results and results['test_train_summary']:
        summary_data = []
        for feature_set, metrics in results['test_train_summary'].items():
            summary_data.append({
                'Feature Set': feature_set,
                'Train Silhouette': metrics['train_silhouette'],
                'Test Silhouette': metrics['test_silhouette'],
                'Difference': metrics['difference']
            })

        summary_df = pd.DataFrame(summary_data)
        print("Columns in summary_df:", summary_df.columns.tolist())
        print("First few rows of summary_df:")
        print(summary_df.head())

        if not summary_df.empty:
            create_train_test_visualizations(
                summary_df,
                f"pics_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}/train_test_comparison"
            )

            print("\nTrain vs Test Performance Summary:")
            print(summary_df.sort_values('Test Silhouette', ascending=False).to_string(index=False))

            most_stable = summary_df.loc[summary_df['Difference'].abs().idxmin()]['Feature Set']
            print(f"\nMost stable feature set: {most_stable}")

            best_test = summary_df.loc[summary_df['Test Silhouette'].idxmax()]['Feature Set']
            print(f"Best feature set on test data: {best_test}")
        else:
            print("\nNo valid results were found for comparison. Check the logs for errors in feature set analysis.")
    else:
        print("\nNo test/train summary data available. Check the logs for errors in feature set analysis.")
