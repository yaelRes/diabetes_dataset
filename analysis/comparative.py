"""
Comparative analysis for diabetes clustering with different feature sets.
"""

import os
import logging
import traceback
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix

# Remove this import to break the circular reference
# from main import main
from utils.data_utils import load_dataset
from visualization.clustering_viz import plot_contingency_heatmap
import pandas as pd


def compare_clusterings(df, labels1, labels2, comparison_name="Comparison", output_dir="output"):
    """
    Compare two different cluster assignments and visualize their relationship.

    Args:
        df (pandas.DataFrame): The dataset
        labels1 (numpy.ndarray): First set of cluster labels
        labels2 (numpy.ndarray): Second set of cluster labels
        comparison_name (str): Name for the comparison (e.g. "All Features vs Selected Features")
        output_dir (str): Directory to save the visualizations

    Returns:
        dict: Dictionary containing comparison metrics
    """
    logging.info(f"Comparing clusterings: {comparison_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Compute agreement metrics
    ari = adjusted_rand_score(labels1, labels2)
    ami = adjusted_mutual_info_score(labels1, labels2)
    contingency = confusion_matrix(labels1, labels2)

    logging.info(f"Adjusted Rand Index: {ari:.4f}")
    logging.info(f"Adjusted Mutual Information: {ami:.4f}")

    # Visualize the contingency table as a heatmap
    labels1_name, labels2_name = comparison_name.split(" vs ")
    plot_contingency_heatmap(contingency, labels1_name, labels2_name, ari, ami, output_dir)

    return {
        'ari': ari,
        'ami': ami,
        'contingency': contingency
    }


def run_comparative_analysis(data_path="diabetes_dataset.csv", feature_sets=None, test_size=0.2, sample_ratio=1.0):
    """
    Run comparative analysis using different feature sets.

    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    feature_sets : dict
        Dictionary of feature sets to compare
    test_size : float
        Proportion of the dataset to include in the test split
    sample_ratio : float
        Proportion of the dataset to use (for faster runs), default is 1.0 (all data)

    Returns:
    --------
    dict : Results and metrics from the comparative analysis
    """
    import logging
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import silhouette_score

    from utils.logging_utils import setup_logging
    # Import main function here instead of at module level
    from main import main

    # Setup logging
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
            result = main(
                data_path=data_path,
                output_dir=feature_set_dir,
                selected_features=selected_features,
                test_size=test_size,
                sample_ratio=sample_ratio  # Pass the sample ratio to main
            )

            comparative_results[feature_set_name] = result

            # Extract key metrics for comparison
            if 'eval_result' in result and 'metrics_df' in result['eval_result']:
                # Get the best method's silhouette score
                best_method = result['eval_result']['best_method']
                best_method_idx = result['eval_result']['metrics_df']['Method'].tolist().index(best_method)
                train_silhouette = result['eval_result']['metrics_df']['Silhouette Score'].iloc[best_method_idx]

                # For test set performance
                test_silhouette = result['test_evaluation'].get('test_silhouette', 0)

                # Calculate difference (for stability assessment)
                difference = test_silhouette - train_silhouette

                test_train_summary[feature_set_name] = {
                    'train_silhouette': train_silhouette,
                    'test_silhouette': test_silhouette,
                    'difference': difference,
                    'best_method': best_method
                }

                logging.info(f"Feature set: {feature_set_name}")
                logging.info(f"  Best method: {best_method}")
                logging.info(f"  Train silhouette: {train_silhouette:.4f}")
                logging.info(f"  Test silhouette: {test_silhouette:.4f}")
                logging.info(f"  Difference: {difference:.4f}")

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

    return {
        'comparative_results': comparative_results,
        'test_train_summary': test_train_summary,
        'output_dir': comparative_dir
    }