#!/usr/bin/env python3
"""
Main entry point for diabetes clustering analysis.
"""

import os
import logging
import numpy as np
from datetime import datetime

from utils.logging_utils import setup_logging
from utils.data_utils import load_dataset, get_column_types, preprocess_data
from analysis.dimension_reduction import perform_pca_analysis, create_dimension_reduction_visualizations
from analysis.clustering import grid_search_clustering_parameters, compare_clustering_algorithms, analyze_cluster_characteristics
from analysis.anomaly_detection import perform_anomaly_detection
from analysis.feature_importance import analyze_feature_importance
from analysis.comparative import run_comparative_analysis
from config import feature_sets

def main(data_path="diabetes_dataset.csv", output_dir="pics", selected_features=None):
    """
    Main function to run the entire analysis pipeline.

    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    output_dir : str
        Directory to save output files
    selected_features : list or None
        List of feature names to include in the analysis. If None, all features are used.
    """
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting diabetes clustering analysis")

    if selected_features:
        logging.info(f"Using selected features: {selected_features}")
    else:
        logging.info("Using all available features")

    # Set seed for reproducibility
    np.random.seed(42)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load and prepare dataset
        df = load_dataset(data_path)
        categorical_cols, numerical_cols = get_column_types(df, selected_features)

        # 2. Preprocess data
        X_processed, preprocessor = preprocess_data(df, categorical_cols, numerical_cols)

        # 3. PCA Analysis
        pca_result = perform_pca_analysis(X_processed, output_dir)

        # 4. Create dimension reduction visualizations
        dim_red_result = create_dimension_reduction_visualizations(X_processed, output_dir)

        # 5. Grid search for optimal clustering parameters
        n_components_list = [pca_result['n_components_95']] + list(range(5, min(30, X_processed.shape[1]), 5))
        k_range = range(2, 11)
        grid_search_result = grid_search_clustering_parameters(X_processed, n_components_list, k_range, output_dir)

        # 6. Compare different clustering algorithms
        clustering_result = compare_clustering_algorithms(
            X_processed,
            grid_search_result['optimal_n_component'],
            grid_search_result['optimal_k'],
            output_dir
        )

        # 7. Perform anomaly detection
        anomaly_result = perform_anomaly_detection(
            clustering_result['X_pca_optimal'],
            grid_search_result['optimal_k'],
            output_dir
        )

        # 8. Analyze cluster characteristics
        cluster_analysis = analyze_cluster_characteristics(
            df,
            clustering_result['best_algorithm_labels'],
            numerical_cols,
            categorical_cols,
            output_dir
        )

        # 9. Analyze feature importance for clustering
        feature_importance = analyze_feature_importance(
            df,
            clustering_result['best_algorithm_labels'],
            numerical_cols,
            categorical_cols,
            preprocessor,
            output_dir
        )

        # 10. Optimize UMAP parameters for visualization
        from analysis.dimension_reduction import optimize_umap_parameters
        umap_result = optimize_umap_parameters(X_processed, clustering_result['best_algorithm'], output_dir)

        # 11. Final evaluation and summary
        from analysis.clustering import final_evaluation, generate_cluster_profiles
        eval_result = final_evaluation(pca_result, clustering_result, umap_result, output_dir)

        # 12. Generate final cluster profiles
        # Determine which labels to use for the final profiles (best method)
        if eval_result['best_method'] == 'UMAP + Best Algorithm':
            final_labels = umap_result['best_umap_labels']
        else:
            final_labels = clustering_result['best_algorithm_labels']

        profile_result = generate_cluster_profiles(
            df,
            final_labels,
            numerical_cols,
            categorical_cols,
            output_dir
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
            'feature_importance': feature_importance
        }

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Create the pics directory if it doesn't exist
    log_file = setup_logging()
    if not os.path.exists("pics"):
        os.makedirs("pics")
        print("Created 'pics' directory")
    else:
        print("'pics' directory already exists")

    # Run the main analysis function
    # This will save all visualizations to the 'pics' folder by default
    try:
        results = run_comparative_analysis(
            data_path="diabetes_dataset.csv",
            feature_sets=feature_sets
        )
        print("Analysis completed successfully!")
        print(f"All visualizations saved to the 'pics' folder")
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())