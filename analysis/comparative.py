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


def run_comparative_analysis(data_path, feature_sets, test_size=0.2):
    """
    Run multiple analyses with different feature sets and compare the results.

    Args:
        data_path (str): Path to the dataset CSV file
        feature_sets (dict): Dictionary where keys are names of feature sets and values are lists of features
        test_size (float): Proportion of the dataset to include in the test split

    Returns:
        dict: Dictionary containing results of all analyses and comparisons
    """
    # Import here to avoid circular imports
    from main import main

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"pics_comparative_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    logging.info(f"Starting comparative analysis with {len(feature_sets)} feature sets and test size {test_size}")

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
                selected_features=features,
                test_size=test_size
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

        # Compare all pairs of feature sets (both training and test results)
        comparisons = {'train': {}, 'test': {}}
        feature_set_names = list(feature_sets.keys())

        for i in range(len(feature_set_names)):
            for j in range(i + 1, len(feature_set_names)):
                name1 = feature_set_names[i]
                name2 = feature_set_names[j]

                # Get labels from the best algorithm for each analysis (training)
                if name1 in all_results and name2 in all_results:
                    # Training set comparisons
                    labels1_train = all_results[name1]['clustering_result']['best_algorithm_labels']
                    labels2_train = all_results[name2]['clustering_result']['best_algorithm_labels']

                    # We need to use only the training portion of the original dataframe
                    # This is an approximation as we don't have the exact same train/test split
                    train_size = 1 - test_size
                    train_samples = int(len(df) * train_size)
                    df_train_approx = df.iloc[:train_samples]

                    comparison_name = f"{name1} vs {name2} (Training)"
                    comparisons['train'][comparison_name] = compare_clusterings(
                        df_train_approx,
                        labels1_train,
                        labels2_train,
                        comparison_name=comparison_name,
                        output_dir=os.path.join(comparison_dir, "train")
                    )

                    logging.info(f"Training comparison between {name1} and {name2} completed")

                    # Test set comparisons
                    if 'test_evaluation' in all_results[name1] and 'test_evaluation' in all_results[name2]:
                        labels1_test = all_results[name1]['test_evaluation']['test_labels']
                        labels2_test = all_results[name2]['test_evaluation']['test_labels']

                        # Use the approximate test portion of the dataset
                        df_test_approx = df.iloc[train_samples:]

                        comparison_name = f"{name1} vs {name2} (Test)"
                        comparisons['test'][comparison_name] = compare_clusterings(
                            df_test_approx,
                            labels1_test,
                            labels2_test,
                            comparison_name=comparison_name,
                            output_dir=os.path.join(comparison_dir, "test")
                        )

                        logging.info(f"Test comparison between {name1} and {name2} completed")

        all_results['comparisons'] = comparisons

        # Generate summary of test vs training performance
        test_train_summary = {}
        for name in feature_set_names:
            if name in all_results and 'test_evaluation' in all_results[name]:
                test_train_summary[name] = {
                    'train_silhouette': all_results[name]['clustering_result']['best_algorithm_silhouette'],
                    'test_silhouette': all_results[name]['test_evaluation']['test_silhouette'],
                    'difference': all_results[name]['test_evaluation']['test_silhouette'] -
                                  all_results[name]['clustering_result']['best_algorithm_silhouette']
                }

        all_results['test_train_summary'] = test_train_summary

        # Create a summary table and save it
        summary_df = pd.DataFrame({
            'Feature Set': list(test_train_summary.keys()),
            'Train Silhouette': [test_train_summary[name]['train_silhouette'] for name in test_train_summary],
            'Test Silhouette': [test_train_summary[name]['test_silhouette'] for name in test_train_summary],
            'Difference': [test_train_summary[name]['difference'] for name in test_train_summary]
        })

        # Sort by test silhouette score (descending)
        summary_df = summary_df.sort_values('Test Silhouette', ascending=False)
        summary_df.to_csv(os.path.join(comparison_dir, 'test_train_summary.csv'), index=False)

        logging.info("Test vs training performance summary generated")

    logging.info("Comparative analysis completed!")
    return all_results
