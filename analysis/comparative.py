"""
Comparative analysis for diabetes clustering with different feature sets.
"""

import os
import logging
import traceback
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix

from utils.data_utils import load_dataset
from visualization.clustering_viz import plot_contingency_heatmap


def compare_clusterings(df, labels1, labels2, comparison_name="Comparison", output_dir="pics"):
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


def run_comparative_analysis(data_path, feature_sets):
    """
    Run multiple analyses with different feature sets and compare the results.

    Args:
        data_path (str): Path to the dataset CSV file
        feature_sets (dict): Dictionary where keys are names of feature sets and values are lists of features

    Returns:
        dict: Dictionary containing results of all analyses and comparisons
    """
    # Import here to avoid circular imports
    from main import main
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"pics_comparative_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    logging.info(f"Starting comparative analysis with {len(feature_sets)} feature sets")

    # Dictionary to store results for each feature set
    all_results = {}

    # Run analysis for each feature set
    for name, features in feature_sets.items():
        output_dir = os.path.join(base_output_dir, name.replace(" ", "_"))
        logging.info(f"Running analysis for feature set: {name}")

        try:
            results = main(
                data_path=data_path,
                output_dir=output_dir,
                selected_features=features
            )
            all_results[name] = results
            logging.info(f"Analysis for {name} completed successfully")
        except Exception as e:
            logging.error(f"Error during analysis for {name}: {e}")
            logging.error(traceback.format_exc())

    # Compare the clusterings if we have multiple feature sets
    if len(feature_sets) > 1:
        comparison_dir = os.path.join(base_output_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        # Load the dataset once for the comparisons
        df = load_dataset(data_path)

        # Compare all pairs of feature sets
        comparisons = {}
        feature_set_names = list(feature_sets.keys())

        for i in range(len(feature_set_names)):
            for j in range(i + 1, len(feature_set_names)):
                name1 = feature_set_names[i]
                name2 = feature_set_names[j]

                # Get labels from the best algorithm for each analysis
                if name1 in all_results and name2 in all_results:
                    labels1 = all_results[name1]['clustering_result']['best_algorithm_labels']
                    labels2 = all_results[name2]['clustering_result']['best_algorithm_labels']

                    comparison_name = f"{name1} vs {name2}"
                    comparisons[comparison_name] = compare_clusterings(
                        df,
                        labels1,
                        labels2,
                        comparison_name=comparison_name,
                        output_dir=comparison_dir
                    )

                    logging.info(f"Comparison between {name1} and {name2} completed")

        all_results['comparisons'] = comparisons

    logging.info("Comparative analysis completed!")
    return all_results
