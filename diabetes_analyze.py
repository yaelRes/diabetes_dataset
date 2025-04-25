import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (16, 10)

# Load the dataset
data = pd.read_csv("diabetes_dataset.csv")

# Data preprocessing
# Drop the first unnamed index column if present
if '' in data.columns:
    data = data.drop('', axis=1)
elif 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Separate numerical and categorical features
categorical_cols = ['Sex', 'Ethnicity', 'Physical_Activity_Level',
                    'Alcohol_Consumption', 'Smoking_Status']
numerical_cols = [col for col in data.columns if col not in categorical_cols and
                  col not in ['Family_History_of_Diabetes', 'Previous_Gestational_Diabetes']]
binary_cols = ['Family_History_of_Diabetes', 'Previous_Gestational_Diabetes']

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('bin', 'passthrough', binary_cols)
    ])

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Apply preprocessing
X_processed = preprocessing_pipeline.fit_transform(data)


def evaluate_clustering(X, labels, method_name):
    """Evaluate clustering performance using silhouette score"""
    if len(np.unique(labels)) > 1:  # Ensure we have more than one cluster
        silhouette = silhouette_score(X, labels)
        print(f"{method_name} - Silhouette Score: {silhouette:.4f}")
        return silhouette
    else:
        print(f"{method_name} - Only one cluster found, cannot calculate silhouette score")
        return -1


def plot_clusters_2d(X_2d, labels, method_name, title):
    """Plot clusters in 2D space"""
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis',
                          alpha=0.7, s=40, edgecolors='w', linewidth=0.5)

    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title}\n{method_name}", fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{method_name.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()


def run_dimensionality_reduction(X, methods):
    """Apply various dimensionality reduction methods"""
    results = {}

    for method_name, method in methods.items():
        print(f"Applying {method_name}...")
        X_reduced = method.fit_transform(X)
        results[method_name] = X_reduced

    return results


def run_clustering(X, methods):
    """Apply various clustering methods"""
    results = {}

    for method_name, method in methods.items():
        print(f"Applying {method_name}...")
        labels = method.fit_predict(X)
        silhouette = evaluate_clustering(X, labels, method_name)
        results[method_name] = {
            'labels': labels,
            'silhouette': silhouette,
            'n_clusters': len(np.unique(labels))
        }

    return results


# Define dimensionality reduction methods
dim_reduction_methods = {
    'PCA': PCA(n_components=2),
    'UMAP': umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42),
    't-SNE': TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
}

# Define clustering methods
clustering_methods = {
    'K-Means (k=4)': KMeans(n_clusters=4, random_state=42),
    'K-Means (k=5)': KMeans(n_clusters=5, random_state=42),
    'K-Means (k=6)': KMeans(n_clusters=6, random_state=42),
    'Agglomerative (k=5)': AgglomerativeClustering(n_clusters=5),
    'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
}

# Run dimensionality reduction
reduced_data = run_dimensionality_reduction(X_processed, dim_reduction_methods)

# Run clustering on the original high-dimensional data
clustering_results = run_clustering(X_processed, clustering_methods)

# Visualize clusters with different dimensionality reduction techniques
print("\nVisualizing clusters with different dimensionality reduction techniques...")

# Find the best clustering method based on silhouette score
best_method = max(clustering_results.items(), key=lambda x: x[1]['silhouette'])[0]
best_labels = clustering_results[best_method]['labels']
print(
    f"\nBest clustering method: {best_method} with silhouette score: {clustering_results[best_method]['silhouette']:.4f}")

# Visualize the best clustering method with different dimensionality reduction techniques
for dr_name, X_reduced in reduced_data.items():
    plot_clusters_2d(
        X_reduced,
        best_labels,
        f"{best_method}",
        f"Diabetes Dataset Clusters Visualized with {dr_name}"
    )

# Analyze the best clustering
best_cluster_labels = best_labels
data['Cluster'] = best_cluster_labels

# Analyze the characteristics of each cluster
cluster_summary = data.groupby('Cluster').mean()
print("\nCluster Characteristics (mean values):")
print(cluster_summary)


# Create a function to visualize feature distributions by cluster
def plot_feature_distributions(data, feature_list, n_cols=3):
    n_features = len(feature_list)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(feature_list):
        if i < len(axes):
            sns.boxplot(x='Cluster', y=feature, data=data, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature} by Cluster')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('feature_distributions_by_cluster.png', dpi=300)
    plt.close()


# Select the most important features for visualization
important_features = [
    'Age', 'BMI', 'Fasting_Blood_Glucose', 'HbA1c',
    'Waist_Circumference', 'Blood_Pressure_Systolic',
    'Cholesterol_Total', 'Cholesterol_HDL', 'Cholesterol_LDL'
]

# Plot feature distributions by cluster
plot_feature_distributions(data, important_features)

# Create a PCA-based feature importance plot
pca = PCA()
pca.fit(X_processed)

# Create a dataframe with PCA components and their explained variance
pca_df = pd.DataFrame({
    'Variance Explained': pca.explained_variance_ratio_,
    'Principal Component': [f'PC{i + 1}' for i in range(len(pca.explained_variance_ratio_))]
})

# Plot the explained variance of PCA components
plt.figure(figsize=(12, 6))
sns.barplot(x='Principal Component', y='Variance Explained', data=pca_df.head(10))
plt.title('Explained Variance by Principal Components')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pca_explained_variance.png', dpi=300)
plt.close()

# Generate a heatmap of cluster centers for the most important features
# First, we need to get the cluster centers
if 'K-Means' in best_method:
    # For K-Means, we can directly get the cluster centers
    kmeans = clustering_methods[best_method]
    cluster_centers = kmeans.cluster_centers_

    # Process feature names for the heatmap
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Get one-hot encoded column names
            encoder = trans
            for i, col in enumerate(cols):
                cats = encoder.categories_[i][1:]  # Skip the first category (dropped)
                feature_names.extend([f"{col}_{cat}" for cat in cats])
        elif name == 'bin':
            feature_names.extend(cols)

    # Create a DataFrame with cluster centers
    centers_df = pd.DataFrame(cluster_centers[:, :len(numerical_cols)],
                              columns=numerical_cols)

    # Normalize the centers for heatmap visualization
    centers_norm = (centers_df - centers_df.mean()) / centers_df.std()

    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(centers_norm, annot=True, cmap='coolwarm', linewidths=.5,
                yticklabels=[f'Cluster {i}' for i in range(centers_norm.shape[0])])
    plt.title('Normalized Cluster Centers for Numerical Features')
    plt.tight_layout()
    plt.savefig('cluster_centers_heatmap.png', dpi=300)
    plt.close()

print("\nClustering analysis complete. Output files saved.")