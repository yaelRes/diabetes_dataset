# Diabetes Clustering Analysis with Train/Test Split

This repository contains code for performing unsupervised clustering analysis on diabetes data, with an added train/test validation approach to assess the stability and generalizability of the clustering models.

## Overview

The project provides a comprehensive pipeline for:

1. Loading and preprocessing diabetes dataset
2. Splitting data into training and test sets
3. Performing dimensionality reduction (PCA, t-SNE, UMAP)
4. Finding optimal clustering parameters
5. Comparing different clustering algorithms (K-means, Hierarchical, DBSCAN, GMM)
6. Evaluating models on test data to assess stability
7. Analyzing cluster characteristics and feature importance
8. Visualizing results with comparisons between training and test performance

## New Train/Test Split Feature

The new train/test split functionality allows you to:

- Train clustering models on a subset of your data
- Evaluate how well the clusters generalize to unseen data
- Identify which feature sets produce the most stable clusters
- Determine whether your clustering solution is likely to be meaningful for new patients

## Files Structure

- `main.py`: Main entry point for the analysis
- `config.py`: Configuration file with feature sets
- `data_utils.py`: Data loading and preprocessing utilities
- `dimension_reduction.py`: Dimensionality reduction analysis
- `clustering.py`: Clustering algorithms and evaluation
- `anomaly_detection.py`: Anomaly detection methods
- `feature_importance.py`: Feature importance analysis
- `comparative.py`: Comparative analysis with different feature sets
- `visualization/`: Visualization modules
  - `clustering_viz.py`: Visualization for clustering results
  - `dimension_reduction.py`: Visualization for dimension reduction
  - `feature_viz.py`: Visualization for feature importance
  - `comparison_viz.py`: Visualization for train/test comparisons

## Usage

### Basic Usage

```bash
python main.py --data_path your_dataset.csv --test_size 0.2
```

### Command Line Arguments

- `--data_path`: Path to the dataset CSV file (default: "diabetes_dataset.csv")
- `--test_size`: Proportion of the dataset to include in the test split (default: 0.2)

### Customizing Feature Sets

You can customize the feature sets in `config.py`. The default configuration includes:

- All Features
- Top 10 Important Features
- Metabolic Features Only
- Clinical Measurements
- Lifestyle Factors
- Genetic and Demographics
- Minimalist Set

### Output

The analysis generates:
- Visualizations for dimension reduction, clustering, and feature importance
- Cluster profiles and characteristics
- Performance metrics for both training and test sets
- Stability metrics to identify the most reliable feature sets
- Comparative analysis across different feature sets

## Understanding the Results

### Train/Test Performance Metrics

- **Training Silhouette Score**: How well-separated the clusters are in the training set
- **Test Silhouette Score**: How well the clustering generalizes to the test set
- **Difference**: Test score minus train score (negative values indicate potential overfitting)
- **Stability**: A measure of how consistent the clustering performance is between training and test sets

### Interpreting Stability

- **High stability + high test score**: Your clustering solution is robust and likely to generalize well
- **Low stability (large difference)**: Your clustering might be overfitting to the training data
- **Negative difference**: Test performance is worse than training (common in overfitting)
- **Positive difference**: Test performance is better than training (uncommon, may indicate lucky split)

## Visualization Examples

The updated code generates several new visualizations in the `train_test_comparison` directory:

1. `train_test_comparison.png`: Bar chart comparing train vs test scores for each feature set
2. `score_differences.png`: Horizontal bar chart showing test-train differences
3. `stability_metrics.png`: Bar chart showing stability scores for each feature set
4. `train_test_scatter.png`: Scatter plot showing the relationship between train and test scores

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- umap-learn

## Note on Clustering Evaluation

Since clustering is an unsupervised learning task, the train/test evaluation is focused on assessing the stability of cluster structures rather than prediction accuracy. A good clustering solution should:

1. Have high silhouette scores in both training and test sets
2. Show minimal difference between training and test performance
3. Maintain consistent cluster characteristics between sets

## License

[MIT License](LICENSE)
