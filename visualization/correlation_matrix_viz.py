"""
Additional visualizations for correlation analysis and risk assessment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_detailed_correlation_matrix(df, numerical_cols, output_dir="output"):
    """Plot a detailed correlation matrix with significance markers.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset
        numerical_cols (list): List of numerical column names
        output_dir (str): Directory to save output files
    """
    # Calculate correlation matrix and p-values
    from scipy.stats import pearsonr
    
    # Create empty matrices for correlations and p-values
    n = len(numerical_cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    # Fill matrices
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i != j:
                corr, p = pearsonr(df[col1].dropna(), df[col2].dropna())
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p
            else:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
    
    # Create DataFrame from matrices
    corr_df = pd.DataFrame(corr_matrix, index=numerical_cols, columns=numerical_cols)
    p_df = pd.DataFrame(p_matrix, index=numerical_cols, columns=numerical_cols)
    
    # Determine significance
    def annotate_heatmap(val, p_val):
        """Add significance stars to correlation values."""
        stars = ''
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        
        if stars:
            return f'{val:.2f}{stars}'
        else:
            return f'{val:.2f}'
    
    # Create annotations
    annot_matrix = np.empty_like(corr_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            annot_matrix[i, j] = annotate_heatmap(corr_matrix[i, j], p_matrix[i, j])
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot
    plt.figure(figsize=(14, 12))
    
    # Generate heatmap with correlation coefficients and significance markers
    ax = sns.heatmap(corr_df, mask=mask, annot=annot_matrix, fmt='', cmap='coolwarm',
                    cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5,
                    vmin=-1, vmax=1)
    
    plt.title('Detailed Correlation Matrix of Numerical Features\n'
              '*p<0.05, **p<0.01, ***p<0.001', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_risk_score(df, df_with_clusters, risk_factors, output_dir="output"):
    """Create and visualize diabetes risk score.
    
    Args:
        df (pandas.DataFrame): Original DataFrame
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        risk_factors (list): List of risk factor column names
        output_dir (str): Directory to save output files
        
    Returns:
        pandas.DataFrame: DataFrame with added risk scores
    """
    # Standardize risk factors
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy to avoid modifying the original
    result_df = df_with_clusters.copy()
    
    # Check if risk factors exist in the dataframe
    available_factors = [col for col in risk_factors if col in df.columns]
    
    if not available_factors:
        print("No risk factors found in the dataset. Cannot create risk score.")
        return result_df
    
    # Standardize risk factors
    scaler = StandardScaler()
    risk_data = df[available_factors].copy()
    
    # Handle missing values
    risk_data = risk_data.fillna(risk_data.mean())
    
    # Standardize the data
    scaled_data = scaler.fit_transform(risk_data)
    
    # Create a simple weighted sum for risk score
    # Higher values = higher risk
    risk_score = np.sum(scaled_data, axis=1)
    
    # Add to dataframe
    result_df['diabetes_risk_score'] = risk_score
    
    # Create metabolic health score (inverse of risk score)
    # Higher values = better health
    health_score = -risk_score
    result_df['metabolic_health_score'] = health_score
    
    # Visualize risk score distribution by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='diabetes_risk_score', data=result_df)
    plt.title('Diabetes Risk Score by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Risk Score (higher = higher risk)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'risk_score_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create risk assessment plot
    from visualization.umap_heatmap_viz import plot_risk_assessment
    plot_risk_assessment(result_df, 'diabetes_risk_score', 'metabolic_health_score', output_dir)
    
    return result_df


def create_multivariate_analysis(df_with_clusters, numerical_cols, output_dir="output"):
    """Create multivariate analysis of numerical features.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        numerical_cols (list): List of numerical column names
        output_dir (str): Directory to save output files
    """
    # Select a subset of columns for the analysis (to keep the plot readable)
    if len(numerical_cols) > 7:
        # Select the most informative columns
        selected_cols = numerical_cols[:7]
    else:
        selected_cols = numerical_cols
    
    # Create a parallel coordinates plot
    plt.figure(figsize=(15, 8))
    
    # Create a copy of the data for plotting
    plot_df = df_with_clusters[selected_cols + ['cluster']].copy()
    
    # Normalize the data for better visualization
    for col in selected_cols:
        plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())
    
    # Create a colormap for the clusters
    from matplotlib.colors import ListedColormap
    n_clusters = len(plot_df['cluster'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    cmap = ListedColormap(colors)
    
    # Plot the parallel coordinates
    pd.plotting.parallel_coordinates(
        plot_df, 'cluster', color=colors, 
        alpha=0.5, axvlines=True
    )
    
    plt.title('Parallel Coordinates Plot of Normalized Features by Cluster')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_feature_importance_for_clusters(df_with_clusters, numerical_cols, categorical_cols, output_dir="output"):
    """Calculate and visualize feature importance for separating clusters.
    
    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster labels
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        output_dir (str): Directory to save output files
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Prepare features and target
    X = df_with_clusters[numerical_cols + categorical_cols].copy()
    y = df_with_clusters['cluster']
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Create and fit model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    model.fit(X, y)
    
    # Get feature importances
    feature_names = (numerical_cols + 
                    [f"{col}_{val}" for col in categorical_cols 
                     for val in model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_[categorical_cols.index(col)]])
    
    # Extract feature importances 
    importances = model.named_steps['classifier'].feature_importances_
    
    # Match importances to feature names
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Features for Cluster Separation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_for_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df
