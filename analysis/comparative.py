import logging
import os
from datetime import datetime
from main import main
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix

from visualization.clustering_viz import plot_contingency_heatmap


def compare_clustering(df, labels1, labels2, comparison_name="comparison", output_dir="output"):
    logging.info(f"comparing clustering: {comparison_name}")
    os.makedirs(output_dir, exist_ok=True)

    ari = adjusted_rand_score(labels1, labels2)
    ami = adjusted_mutual_info_score(labels1, labels2)
    contingency = confusion_matrix(labels1, labels2)

    logging.info(f"adjusted rand index: {ari:.4f}")
    logging.info(f"adjusted mutual information: {ami:.4f}")

    labels1_name, labels2_name = comparison_name.split(" vs ")
    plot_contingency_heatmap(contingency, labels1_name, labels2_name, ari, ami, output_dir)


def run_comparative_analysis(data_path="diabetes_dataset.csv", feature_sets=None, test_size=0.2, sample_ratio=1.0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparative_dir = f"pics_comparative_{timestamp}"
    os.makedirs(comparative_dir, exist_ok=True)

    logging.info(f"starting comparative analysis with {len(feature_sets)} feature sets")

    if sample_ratio < 1.0:
        logging.info(f"Using {sample_ratio * 100:.1f}% of the dataset for faster processing")

    comparative_results = {}
    test_train_summary = {}

    for feature_set_name, selected_features in feature_sets.items():
        logging.info(f"analyzing feature set: {feature_set_name}")

        feature_set_dir = os.path.join(comparative_dir, feature_set_name.replace(" ", "_"))
        os.makedirs(feature_set_dir, exist_ok=True)

        try:
            result = main(
                data_path=data_path,
                output_dir=feature_set_dir,
                selected_features=selected_features,
                test_size=test_size,
                sample_ratio=sample_ratio  # Pass the sample ratio to main
            )

            comparative_results[feature_set_name] = result

            if 'eval_result' in result and 'metrics_df' in result['eval_result']:
                best_method = result['eval_result']['best_method']
                best_method_idx = result['eval_result']['metrics_df']['Method'].tolist().index(best_method)
                train_silhouette = result['eval_result']['metrics_df']['Silhouette Score'].iloc[best_method_idx]
                test_silhouette = result['test_evaluation'].get('test_silhouette', 0)

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
