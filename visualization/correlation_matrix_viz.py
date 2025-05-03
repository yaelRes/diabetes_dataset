import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from visualization.umap_heatmap_viz import plot_risk_assessment
from matplotlib.colors import ListedColormap


def plot_detailed_correlation_matrix(df, numerical_cols, output_dir="output"):
    n = len(numerical_cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i != j:
                corr, p = pearsonr(df[col1].dropna(), df[col2].dropna())
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p
            else:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0

    corr_df = pd.DataFrame(corr_matrix, index=numerical_cols, columns=numerical_cols)
    p_df = pd.DataFrame(p_matrix, index=numerical_cols, columns=numerical_cols)

    def annotate_heatmap(val, p_val):
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

    annot_matrix = np.empty_like(corr_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            annot_matrix[i, j] = annotate_heatmap(corr_matrix[i, j], p_matrix[i, j])

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(corr_df, mask=mask, annot=annot_matrix, fmt='', cmap='coolwarm',
                     cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5,
                     vmin=-1, vmax=1)

    plt.title('Detailed Correlation Matrix of Numerical Features\n'
              '*p<0.05, **p<0.01, ***p<0.001', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_risk_score(df, df_with_clusters, risk_factors, output_dir="output"):
    result_df = df_with_clusters.copy()

    available_factors = [col for col in risk_factors if col in df.columns]

    if not available_factors:
        print("No risk factors found in the dataset. Cannot create risk score.")
        return result_df

    scaler = StandardScaler()
    risk_data = df[available_factors].copy()

    risk_data = risk_data.fillna(risk_data.mean())

    scaled_data = scaler.fit_transform(risk_data)

    risk_score = np.sum(scaled_data, axis=1)

    result_df['diabetes_risk_score'] = risk_score

    health_score = -risk_score
    result_df['metabolic_health_score'] = health_score

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='diabetes_risk_score', data=result_df)
    plt.title('Diabetes Risk Score by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Risk Score (higher = higher risk)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'risk_score_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plot_risk_assessment(result_df, 'diabetes_risk_score', 'metabolic_health_score', output_dir)

    return result_df


def create_multivariate_analysis(df_with_clusters, numerical_cols, output_dir="output"):
    if len(numerical_cols) > 7:
        selected_cols = numerical_cols[:7]
    else:
        selected_cols = numerical_cols

    plt.figure(figsize=(15, 8))

    plot_df = df_with_clusters[selected_cols + ['cluster']].copy()

    for col in selected_cols:
        plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())

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

    X = df_with_clusters[numerical_cols + categorical_cols].copy()
    y = df_with_clusters['cluster']

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

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    model.fit(X, y)

    feature_names = (numerical_cols +
                     [f"{col}_{val}" for col in categorical_cols
                      for val in
                      model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_[
                          categorical_cols.index(col)]])

    importances = model.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    })

    importance_df = importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Features for Cluster Separation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_for_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df
