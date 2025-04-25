#!/usr/bin/env python3
"""
Main entry point for diabetes clustering analysis.
"""

import os
import logging
import numpy as np
from datetime import datetime
import pandas as pd

from analysis.comparative import run_comparative_analysis
from analysis.test_evaluation import evaluate_on_test_set
from utils.logging_utils import setup_logging
from utils.data_utils import load_dataset, get_column_types, preprocess_data
from analysis.dimension_reduction import perform_pca_analysis, create_dimension_reduction_visualizations
from analysis.clustering import grid_search_clustering_parameters, compare_clustering_algorithms, analyze_cluster_characteristics
from analysis.anomaly_detection import perform_anomaly_detection
from analysis.feature_importance import analyze_feature_importance
from config import feature_sets, CLUSTERING_CONFIG
from visualization.comparison_viz import create_train_test_visualizations


def main(data_path="diabetes_dataset.csv", output_dir="pics", selected_features=None, test_size=0.2):
    """
    Main function to run the entire analysis pipeline with train/test split.

    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    output_dir : str
        Directory to save output files
    selected_features : list or None
        List of feature names to include in the analysis. If None, all features are used.
    test_size : float
        Proportion of the dataset to include in the test split
    """
    # Set up logging


    log_file = setup_logging()
    logging.info("Starting diabetes clustering analysis with train/test split")

    if selected_features:
        logging.info(f"Using selected features: {selected_features}")
    else:
        logging.info("Using all available features")

    # Set seed for reproducibility
    np.random.seed(42)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    try:
        # 1. Load dataset
        df = load_dataset(data_path)
        categorical_cols, numerical_cols = get_column_types(df, selected_features)

        # 2. Split into train and test sets
        from utils.data_utils import split_train_test
        df_train, df_test = split_train_test(df, test_size=test_size, random_state=42)
        logging.info(f"Dataset split: {len(df_train)} training samples, {len(df_test)} test samples")

        # 3. Preprocess training data
        X_train_processed, preprocessor = preprocess_data(df_train, categorical_cols, numerical_cols)
        logging.info(f"Processed training data shape: {X_train_processed.shape}")

        # 4. Preprocess test data (using the same preprocessor)
        X_test = df_test[numerical_cols + categorical_cols]
        X_test_processed = preprocessor.transform(X_test)
        logging.info(f"Processed test data shape: {X_test_processed.shape}")

        # 5. PCA Analysis on training data
        pca_result = perform_pca_analysis(X_train_processed, train_dir)

        # 6. Create dimension reduction visualizations
        dim_red_result = create_dimension_reduction_visualizations(X_train_processed, None, None, train_dir)

        # 7. Grid search for optimal clustering parameters
        n_components_list = [pca_result['n_components_95']] + list(range(5, min(30, X_train_processed.shape[1]), 5))
        k_range = CLUSTERING_CONFIG["k_range"]
        grid_search_result = grid_search_clustering_parameters(X_train_processed, n_components_list, k_range, train_dir)

        # 8. Compare different clustering algorithms
        clustering_result = compare_clustering_algorithms(
            X_train_processed,
            grid_search_result['optimal_n_component'],
            grid_search_result['optimal_k'],
            train_dir
        )

        # 9. Perform anomaly detection
        anomaly_result = perform_anomaly_detection(
            clustering_result['X_pca_optimal'],
            grid_search_result['optimal_k'],
            train_dir
        )

        # 10. Analyze cluster characteristics
        cluster_analysis = analyze_cluster_characteristics(
            df_train,
            clustering_result['best_algorithm_labels'],
            numerical_cols,
            categorical_cols,
            train_dir
        )

        # 11. Analyze feature importance for clustering
        feature_importance = analyze_feature_importance(
            df_train,
            clustering_result['best_algorithm_labels'],
            numerical_cols,
            categorical_cols,
            preprocessor,
            train_dir
        )

        # 12. Optimize UMAP parameters for visualization
        from analysis.dimension_reduction import optimize_umap_parameters
        umap_result = optimize_umap_parameters(X_train_processed, clustering_result['best_algorithm'], train_dir)

        # Add the t-SNE optimization after it:
        # 12b. Optimize t-SNE parameters for visualization
        from analysis.dimension_reduction import optimize_tsne_parameters
        tsne_result = optimize_tsne_parameters(X_train_processed, clustering_result['best_algorithm'], train_dir)

        # 13. Final evaluation and summary
        from analysis.clustering import final_evaluation, generate_cluster_profiles
        eval_result = final_evaluation(pca_result, clustering_result, umap_result, tsne_result, X_train_processed,
                                       train_dir)

        # 14. Generate final cluster profiles
        # Determine which labels to use for the final profiles (best method)
        if eval_result['best_method'] == 'UMAP + Best Algorithm':
            final_labels_train = umap_result['best_umap_labels']
        else:
            final_labels_train = clustering_result['best_algorithm_labels']

        profile_result = generate_cluster_profiles(
            df_train,
            final_labels_train,
            numerical_cols,
            categorical_cols,
            train_dir
        )

        # 15. Test set evaluation
        test_evaluation = evaluate_on_test_set(
            df_test,
            X_test_processed,
            clustering_result,
            pca_result,
            umap_result,
            eval_result,
            preprocessor,
            numerical_cols,
            categorical_cols,
            test_dir
        )

        logging.info("Analysis complete!")

        return {
            'pca_result': pca_result,
            'grid_search_result': grid_search_result,
            'clustering_result': clustering_result,
            'anomaly_result': anomaly_result,
            'umap_result': umap_result,
            'eval_result': eval_result,
            'profile_result': profile_result,
            'feature_importance': feature_importance,
            'test_evaluation': test_evaluation,
            'train_samples': len(df_train),
            'test_samples': len(df_test)
        }

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Import the visualization module for train/test comparisons
    from visualization.comparison_viz import create_train_test_visualizations
    import pandas as pd
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run diabetes clustering analysis with train/test split.')
    parser.add_argument('--data_path', type=str, default="diabetes_dataset.csv", help='Path to the dataset CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    args = parser.parse_args()

    # Create the pics directory if it doesn't exist
    log_file = setup_logging()
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created 'pics' directory")
    else:
        print("'pics' directory already exists")

    # Run the main analysis function with train/test split
    results = run_comparative_analysis(
        data_path=args.data_path,
        feature_sets=feature_sets,
        test_size=args.test_size
    )

    print("Analysis completed successfully!")
    print(f"All visualizations saved to the 'pics_comparative_*' folder")

    # Create a summary dataframe for visualization
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

        # Check if the DataFrame is not empty before proceeding
        if not summary_df.empty:
            # Create comparison visualizations
            create_train_test_visualizations(
                summary_df,
                f"pics_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}/train_test_comparison"
            )

            # Print summary table
            print("\nTrain vs Test Performance Summary:")
            print(summary_df.sort_values('Test Silhouette', ascending=False).to_string(index=False))

            # Find the most stable feature set (smallest absolute difference)
            most_stable = summary_df.loc[summary_df['Difference'].abs().idxmin()]['Feature Set']
            print(f"\nMost stable feature set: {most_stable}")

            # Find the best performing feature set on test data
            best_test = summary_df.loc[summary_df['Test Silhouette'].idxmax()]['Feature Set']
            print(f"Best feature set on test data: {best_test}")
        else:
            print("\nNo valid results were found for comparison. Check the logs for errors in feature set analysis.")
    else:
        print("\nNo test/train summary data available. Check the logs for errors in feature set analysis.")

