"""
Feature importance analysis for diabetes clustering.

This module contains functions for analyzing the importance of features
in determining cluster separation and boundaries.
"""

import os
import numpy as np
import pandas as pd
import logging
from scipy import stats

from utils.caching import cache_result
from visualization.feature_viz import (
    plot_feature_importance_f_values,
    plot_feature_importance_chi2,
    plot_top_features_pairplot
)


@cache_result()
def analyze_feature_importance(df, best_algorithm_labels, numerical_cols, categorical_cols, preprocessor,
                               output_dir="pics"):
    """
    Analyze the importance of features for cluster separation.

    Args:
        df (pandas.DataFrame): The dataset
        best_algorithm_labels (numpy.ndarray): Best clustering labels
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        preprocessor (ColumnTransformer): Preprocessor used to transform the data
        output_dir (str): Directory to save visualizations

    Returns:
        dict: Dictionary containing feature importance results
    """
    logging.info("Analyzing feature importance for clustering...")
    os.makedirs(output_dir, exist_ok=True)

    # Create a dataframe with the original data and cluster labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = best_algorithm_labels

    # 1. ANOVA F-value for numerical features (how well each feature separates clusters)
    f_values = {}
    p_values = {}

    for col in numerical_cols:
        # Calculate f_value and p_value using one-way ANOVA
        groups = [df_with_clusters[df_with_clusters['cluster'] == c][col].values for c in np.unique(best_algorithm_labels)]
        f_val, p_val = stats.f_oneway(*groups)
        f_values[col] = f_val
        p_values[col] = p_val

    # Sort features by F-value (higher is better)
    sorted_features = sorted(f_values.items(), key=lambda x: x[1], reverse=True)

    # Plot F-values
    features = [x[0] for x in sorted_features]
    f_vals = [x[1] for x in sorted_features]
    plot_feature_importance_f_values(features, f_vals, p_values, output_dir)

    # 2. Chi-square for categorical features
    chi2_values = {}
    chi2_p_values = {}
    sorted_cat_features = []

    for col in categorical_cols:
        contingency_table = pd.crosstab(df_with_clusters['cluster'], df_with_clusters[col])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        chi2_values[col] = chi2
        chi2_p_values[col] = p

    # Sort categorical features by chi-square value (higher is better)
    if categorical_cols:
        sorted_cat_features = sorted(chi2_values.items(), key=lambda x: x[1], reverse=True)
        cat_features = [x[0] for x in sorted_cat_features]
        chi2_vals = [x[1] for x in sorted_cat_features]

        # Plot chi-square values
        plot_feature_importance_chi2(cat_features, chi2_vals, chi2_p_values, output_dir)

    # 3. Create a summary dataframe for all features
    importance_df = pd.DataFrame({
        'Feature': features,
        'F_Value': f_vals,
        'p_value': [p_values[feat] for feat in features],
        'Significance': ['***' if p_values[feat] < 0.001 else
                         '**' if p_values[feat] < 0.01 else
                         '*' if p_values[feat] < 0.05 else
                         '' for feat in features]
    })

    # Add categorical features if they exist
    if categorical_cols:
        cat_importance_df = pd.DataFrame({
            'Feature': cat_features,
            'Chi2_Value': chi2_vals,
            'p_value': [chi2_p_values[feat] for feat in cat_features],
            'Significance': ['***' if chi2_p_values[feat] < 0.001 else
                             '**' if chi2_p_values[feat] < 0.01 else
                             '*' if chi2_p_values[feat] < 0.05 else
                             '' for feat in cat_features]
        })
        importance_df = pd.concat([importance_df, cat_importance_df], ignore_index=True)

    # Save to CSV
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # 4. Visualize cluster separation for top features
    # Get top 3 numerical features based on F-value
    top_numerical = features[:min(3, len(features))]

    if len(top_numerical) > 1:
        # Create pairwise scatter plots for top features
        plot_top_features_pairplot(df_with_clusters, top_numerical, output_dir)

    return {
        'numerical_importance': {
            'f_values': f_values,
            'p_values': p_values,
            'sorted_features': sorted_features
        },
        'categorical_importance': {
            'chi2_values': chi2_values,
            'chi2_p_values': chi2_p_values,
            'sorted_features': sorted_cat_features if categorical_cols else []
        },
        'importance_df': importance_df,
        'top_numerical_features': top_numerical
    }