import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
import kagglehub
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("diabetes_risk_log.log"),
        logging.StreamHandler()  # logging.infos to console
    ]
)

# Set seed for reproducibility
np.random.seed(42)

# Download dataset if not already downloaded
path = kagglehub.dataset_download("marshalpatel3558/diabetes-prediction-dataset")
logging.info(f"diabetes_dataset.csv downloaded to {path}")

df = pd.read_csv("diabetes_dataset.csv")
df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')
logging.info(f"dataset shape {df.shape}")

# Display basic information
logging.info("\nDataset Overview:")
logging.info(df.columns)
logging.info(df.head())
logging.info("\nBasic Statistics:")
logging.info(df.describe())
logging.info("\nMissing Values:")
logging.info(df.isnull().sum())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove the unnamed index column if it exists
if '' in numerical_cols:
    numerical_cols.remove('')
    df = df.drop('', axis=1, errors='ignore')
elif 'Unnamed: 0' in numerical_cols:
    numerical_cols.remove('Unnamed: 0')
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')

logging.info(f"\nCategorical columns: {categorical_cols}")
logging.info(f"Numerical columns: {numerical_cols}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(df)
logging.info(f"\nProcessed data shape: {X_processed.shape}")


# Function to plot clusters
def plot_clusters(X_2d, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


#------------------------------------PCA stuff-----------------------------------------------------

# I use all features
pca_full = PCA()
pca_full.fit(X_processed)
logging.info(f"pca all components, explained variance ratio: {pca_full.explained_variance_ratio_}")

plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axhline(y=0.99, color='g', linestyle='--')
plt.grid(True)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('{PCA Explained Variance by Components')
plt.show()
"""
The `explained_variance_ratio_` in PCA (Principal Component Analysis) 
tells you **how much variance each principal component (PC) explains out of the total variance in your dataset**.

So, in your case:

- The first principal component explains about **5.9%** of the total variance (`0.059`).
- The second explains about **5.8%**, and so on.

This is useful to understand **how many components you need to keep most of the information** in your data.

---
Your top components:
- PC0: 5.90%
- PC1: 5.85%
- PC2: 5.75%
- PC3: 5.72%
- PC4: 5.65%
- ...
- PC14: 5.19%
Up to PC14, each contributes roughly equally, meaning the dataset is relatively **spread out across many dimensions**.
Then there's a **sharp drop**:
- PC15: 2.77%
- PC16: 2.45%
- PC17: 1.87%
- ...
- PC24: ~0%
- PC25–PC28: exactly 0%
---
### Interpretation of near-zero or zero values:
- Components 24–28 explain **no variance**. That means they’re **useless** in terms of information —
 your data lives in a lower-dimensional space (i.e., rank-deficient).
- Possibly, your data has only ~24 meaningful dimensions, 
and the rest are either linear combinations of others or completely redundant
 (e.g., due to collinearity or preprocessing like one-hot encoding).
---
### What to do with this:
- You can **plot the cumulative explained variance** (scree plot) to decide how many components to keep.
- Often, we retain enough PCs to explain **~90–95%** of the variance.
- It also helps with dimensionality reduction before clustering or visualization (e.g., t-SNE, UMAP).
"""

#----------------------------------------PCA  TSNE UMAP 2d to see the distrebiution-------------------------------------

# I do pca with 2 just to see the distribution
pca_components = 2
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_processed)
logging.info(f"pca n_components={pca_components} explained variance ratio: {pca.explained_variance_ratio_}")

# t-SNE
logging.info("\nPerforming t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_processed)

# UMAP
logging.info("\nPerforming UMAP...")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_processed)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# PCA plot
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
axes[0].set_title(f'PCA (2D)\nPC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE plot
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
axes[1].set_title('t-SNE (2D)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

# UMAP plot
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6)
axes[2].set_title('UMAP (2D)')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.suptitle("Dimensionality Reduction Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#----------------------------------------------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize variables to track results
silhouette_scores = []
n_components_values = []
k_values = []
max_silhouette_score = 0
optimal_n_component = 0
optimal_k = 0
optimal_pca = None

# Define parameter ranges
n_components_list = [0.95] + list(range(15, 25))  # PCA components
k_range = range(2, 8)  # K-means clusters

# Run grid search over PCA components and number of clusters
for n_component in n_components_list:
    for k in k_range:
        pca = PCA(n_components=n_component)
        pca_n = pca.fit_transform(X_processed)

        if isinstance(n_component, float):
            logging.info(f"reduced to {pca_n.shape[1]} components preserving {n_component * 100:.0f}% of variance")
        else:
            logging.info(f"reduced to {pca_n.shape[1]} components")

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = kmeans.fit_predict(pca_n)

        # Calculate silhouette score
        score = silhouette_score(pca_n, labels)

        # Store results
        silhouette_scores.append(score)
        n_components_values.append(n_component)
        k_values.append(k)

        # Track optimal parameters
        if score > max_silhouette_score:
            max_silhouette_score = score
            optimal_n_component = n_component
            optimal_k = k
            optimal_pca = pca_n

        logging.info(f"PCA n_component={n_component}, K={k}, Silhouette Score: {score:.4f}")

# Create DataFrame for easier manipulation
results_df = pd.DataFrame({
    'n_component': n_components_values,
    'k': k_values,
    'score': silhouette_scores
})

# Log optimal parameters
logging.info(f"Optimal parameters: PCA n_component={optimal_n_component}, K={optimal_k}")
logging.info(f"Best silhouette score: {max_silhouette_score:.4f}")

# Create heatmap visualization
plt.figure(figsize=(12, 8))

# Create pivot table for heatmap
heatmap_data = results_df.pivot_table(
    index='n_component',
    columns='k',
    values='score'
)

# Format labels for n_components
formatted_labels = []
for n in heatmap_data.index:
    if isinstance(n, float):
        formatted_labels.append(f"{n:.2f}")
    else:
        formatted_labels.append(f"{n}")

# Create heatmap
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                 linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

# Highlight optimal parameters
optimal_row = list(heatmap_data.index).index(optimal_n_component)
optimal_col = list(heatmap_data.columns).index(optimal_k)
ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title('Silhouette Scores for Different PCA Components and K-Means Clusters')
plt.ylabel('PCA Components')
plt.xlabel('Number of Clusters (k)')
plt.tight_layout()
plt.savefig('pca_kmeans_heatmap.png', dpi=300)
plt.show()

# Print the best parameter combination
print(f"Best parameters: PCA components={optimal_n_component}, K={optimal_k}")
print(f"Best silhouette score: {max_silhouette_score:.4f}")
# silhouette_scores = []
# bar_labels = []
# max_silhouette_score = 0
# optimal_n_component = 0
# optimal_k = 0
# k_range = range(2, 8)
# optimal_pca = None
# for n_component in [0.95, *range(15, 25)]:
#     for k in k_range:
#         pca_n = PCA(n_components=n_component)  # Keep components that explain 95% of variance
#         pca_n = pca_n.fit_transform(X_processed)
#         logging.info(f"reduced to {pca_n.shape[1]} components if less than 1 it preserve value percent of the data")
#         # Try KMeans with better initialization and more iterations
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
#         labels = kmeans.fit_predict(pca_n)
#         score = silhouette_score(pca_n, labels)
#         silhouette_scores.append(score)
#         bar_labels.append(f"n={n_component},k={k}")
#         if score > max_silhouette_score:
#             max_silhouette_score = score
#             optimal_n_component = n_component
#             optimal_k = k
#             optimal_pca = pca_n
#         logging.info(f"pca n_component={n_component} K={k}, Silhouette Score: {score:.4f}")
#
# plt.figure(figsize=(10, 6))
# plt.bar(bar_labels, silhouette_scores, color='skyblue', edgecolor='black')
# plt.xlabel('Number of n_component=(n) Clusters=(k)')
# plt.ylabel('Silhouette Score')
# plt.title(f'Improved Silhouette Score vs. Number of KMeans Clusters on pca_k')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

from sklearn.cluster import AgglomerativeClustering

# Try hierarchical clustering with various linkage methods
linkage_methods = ['ward', 'complete', 'average']
silhouette_scores_hierarchical = []

for method in linkage_methods:
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    hierarchical_labels = hierarchical.fit_predict(optimal_pca)
    score = silhouette_score(optimal_pca, hierarchical_labels)
    silhouette_scores_hierarchical.append(score)
    logging.info(f"Hierarchical with {method} linkage: Silhouette Score: {score:.4f} pca{optimal_k}")

# Use the best method
best_method = linkage_methods[np.argmax(silhouette_scores_hierarchical)]
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage=best_method)
hierarchical_labels = hierarchical.fit_predict(optimal_pca)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(linkage_methods, silhouette_scores_hierarchical, color='skyblue', edgecolor='black')
plt.xlabel('Linkage Method')
plt.ylabel('Silhouette Score')
plt.title(f'Silhouette Score vs Linkage Method\n'
          f'AgglomerativeClustering on PCA n_component={optimal_n_component}, n_clusters={optimal_k}')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Try UMAP with optimized parameters before clustering
# UMAP often produces better embeddings for clustering than t-SNE or PCA
import umap.umap_ as umap

# For UMAP heatmap visualization
# Initialize variables for storing results
umap_results = []
n_neighbors_options = [5, 15, 30, 50]
min_dist_options = [0.0, 0.1, 0.25, 0.5]

# Run grid search over UMAP parameters
for n_components in range(2, 6):
    for n_clusters in range(2, 6):
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                # Create UMAP embedding
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    n_components=n_components, random_state=42)
                X_umap = reducer.fit_transform(X_processed)

                # Cluster the UMAP embedding
                kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                labels_umap = kmeans_umap.fit_predict(X_umap)

                # Calculate silhouette score
                score = silhouette_score(X_umap, labels_umap)

                # Store results
                umap_results.append({
                    'n_components': n_components,
                    'n_clusters': n_clusters,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'score': score
                })

                logging.info(
                    f"UMAP(n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}) + KMeans(k={n_clusters}): Silhouette Score: {score:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(umap_results)

# Find the overall best parameters
best_idx = results_df['score'].idxmax()
best_score = results_df.loc[best_idx, 'score']
optimal_n_components = results_df.loc[best_idx, 'n_components']
optimal_n_clusters = results_df.loc[best_idx, 'n_clusters']
optimal_n_neighbors = results_df.loc[best_idx, 'n_neighbors']
optimal_min_dist = results_df.loc[best_idx, 'min_dist']

logging.info(
    f"Best UMAP parameters: n_components={optimal_n_components}, n_clusters={optimal_n_clusters}, n_neighbors={optimal_n_neighbors}, min_dist={optimal_min_dist}")
logging.info(f"Best silhouette score: {best_score:.4f}")

# Create two different heatmaps:

# 1. Heatmap for n_neighbors vs min_dist (at optimal n_components and n_clusters)
filtered_df = results_df[
    (results_df['n_components'] == optimal_n_components) &
    (results_df['n_clusters'] == optimal_n_clusters)
    ]

heatmap_data = filtered_df.pivot_table(
    index='n_neighbors',
    columns='min_dist',
    values='score'
)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                 linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

# Highlight optimal parameters
best_row_idx = list(heatmap_data.index).index(optimal_n_neighbors)
best_col_idx = list(heatmap_data.columns).index(optimal_min_dist)
ax.add_patch(plt.Rectangle((best_col_idx, best_row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title(f'UMAP Silhouette Scores\n(n_components={optimal_n_components}, n_clusters={optimal_n_clusters})')
plt.ylabel('n_neighbors')
plt.xlabel('min_dist')
plt.tight_layout()
plt.savefig('umap_neighbors_mindist_heatmap.png', dpi=300)
plt.show()

# 2. Heatmap for n_components vs n_clusters (at optimal n_neighbors and min_dist)
filtered_df = results_df[
    (results_df['n_neighbors'] == optimal_n_neighbors) &
    (results_df['min_dist'] == optimal_min_dist)
    ]

heatmap_data = filtered_df.pivot_table(
    index='n_components',
    columns='n_clusters',
    values='score'
)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                 linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

# Highlight optimal parameters
best_row_idx = list(heatmap_data.index).index(optimal_n_components)
best_col_idx = list(heatmap_data.columns).index(optimal_n_clusters)
ax.add_patch(plt.Rectangle((best_col_idx, best_row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title(f'UMAP Silhouette Scores\n(n_neighbors={optimal_n_neighbors}, min_dist={optimal_min_dist})')
plt.ylabel('n_components')
plt.xlabel('n_clusters')
plt.tight_layout()
plt.savefig('umap_components_clusters_heatmap.png', dpi=300)
plt.show()



# Try different n_neighbors values for UMAP
# n_neighbors_options = [5, 15, 30, 50]
# min_dist_options = [0.0, 0.1, 0.25, 0.5]
# best_score = -1
# best_umap_params = None
# best_labels = None
# best_n_clusters = 0
# best_n_components = 0
# for n_components in range(2, 6):
#     for n_clusters in range(2, 6):
#         for n_neighbors in n_neighbors_options:
#             for min_dist in min_dist_options:
#                 # Create UMAP embedding
#                 reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
#                                     n_components=n_components, random_state=42)
#                 X_umap_optimized = reducer.fit_transform(X_processed)
#
#                 # Cluster the UMAP embedding
#                 kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
#                 labels_umap = kmeans_umap.fit_predict(X_umap_optimized)
#
#                 # Calculate silhouette score
#                 score = silhouette_score(X_umap_optimized, labels_umap, random_state=42)
#
#                 logging.info(
#                     f"UMAP(n_neighbors={n_neighbors}, min_dist={min_dist},n_components={n_components}) + KMeans: Silhouette Score: {score:.4f}")
#
#                 if score > best_score:
#                     best_score = score
#                     best_umap_params = (n_neighbors, min_dist)
#                     best_labels = labels_umap
#                     best_X_umap = X_umap_optimized
#                     best_n_clusters = n_clusters
#                     best_n_components = n_components
#
# logging.info(f"Best UMAP parameters: n_neighbors={best_umap_params[0]}, min_dist={best_umap_params[1]}"
#              f" best_n_clusters={best_n_clusters} n_components={best_n_components}")
# logging.info(f"Best silhouette score: {best_score:.4f}")
#
# # Visualize the best clustering results
# plt.figure(figsize=(12, 10))
# plt.scatter(best_X_umap[:, 0], best_X_umap[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
# plt.colorbar(label='Cluster')
# plt.title(f'UMAP + KMeans Clustering (Silhouette Score: {best_score:.4f})')
# plt.tight_layout()
# plt.show()

raise Exception()


# 6. If all else fails, try Gaussian Mixture Models with different covariance types
from sklearn.mixture import GaussianMixture

covariance_types = ['full', 'tied', 'diag', 'spherical']
silhouette_scores_gmm = []

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=optimal_k, covariance_type=cov_type, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca_optimal)
    score = silhouette_score(X_pca_optimal, gmm_labels)
    silhouette_scores_gmm.append(score)
    logging.info(f"GMM with {cov_type} covariance: Silhouette Score: {score:.4f}")

# Use the best covariance type
best_cov_type = covariance_types[np.argmax(silhouette_scores_gmm)]
gmm = GaussianMixture(n_components=optimal_k, covariance_type=best_cov_type, random_state=42)
gmm_labels = gmm.fit_predict(X_pca_optimal)

# Final evaluation: compare all methods and use the best one
methods = ['KMeans', 'Hierarchical', 'UMAP+KMeans', 'GMM']
scores = [
    silhouette_score(X_pca_optimal, kmeans_labels),
    silhouette_score(X_pca_optimal, hierarchical_labels),
    best_score,
    silhouette_score(X_pca_optimal, gmm_labels)
]

best_method_idx = np.argmax(scores)
logging.info(f"Best clustering method: {methods[best_method_idx]} with score {scores[best_method_idx]:.4f}")

# Use the best method's labels for subsequent analysis
if methods[best_method_idx] == 'KMeans':
    best_labels_final = kmeans_labels
elif methods[best_method_idx] == 'Hierarchical':
    best_labels_final = hierarchical_labels
elif methods[best_method_idx] == 'UMAP+KMeans':
    best_labels_final = best_labels
else:  # GMM
    best_labels_final = gmm_labels

# Update the clusters in your dataframe
df_with_clusters = df.copy()
df_with_clusters['Cluster'] = best_labels_final

# 3. Feature Analysis and Cluster Interpretation

# Add cluster labels to original dataframe
df_with_clusters = df.copy()
df_with_clusters['KMeans_Cluster'] = kmeans_labels

# Analyze numerical features across clusters
plt.figure(figsize=(15, 12))
numerical_features_to_plot = ['Age', 'BMI', 'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic',
                              'Cholesterol_Total']
for i, feature in enumerate(numerical_features_to_plot):
    if feature in df_with_clusters.columns:
        plt.subplot(3, 2, i + 1)
        sns.boxplot(x='KMeans_Cluster', y=feature, data=df_with_clusters)
        plt.title(f'{feature} by Cluster')
plt.tight_layout()
plt.show()

# Analyze categorical features across clusters
plt.figure(figsize=(15, 12))
categorical_features_to_plot = ['Sex', 'Ethnicity', 'Physical_Activity_Level', 'Smoking_Status',
                                'Family_History_of_Diabetes']
for i, feature in enumerate(categorical_features_to_plot):
    if feature in df_with_clusters.columns:
        if i < 6:  # Limit to 6 plots
            plt.subplot(3, 2, i + 1)
            cluster_feature_counts = pd.crosstab(df_with_clusters['KMeans_Cluster'], df_with_clusters[feature],
                                                 normalize='index')
            cluster_feature_counts.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'{feature} Distribution by Cluster')
            plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Diabetes Risk Analysis

# Calculate mean values of key health indicators by cluster
cluster_stats = df_with_clusters.groupby('KMeans_Cluster').agg({
    'Age': 'mean',
    'BMI': 'mean',
    'Fasting_Blood_Glucose': 'mean',
    'HbA1c': 'mean',
    'Blood_Pressure_Systolic': 'mean',
    'Cholesterol_Total': 'mean',
    'Family_History_of_Diabetes': 'mean'
}).reset_index()

logging.info("\nCluster Statistics:")
logging.info(cluster_stats)

# Identify high-risk and low-risk clusters based on diabetes indicators
# Higher values in HbA1c and Fasting_Blood_Glucose indicate higher diabetes risk
cluster_stats['Diabetes_Risk_Score'] = (
                                               (cluster_stats['HbA1c'] - cluster_stats['HbA1c'].min()) / (
                                               cluster_stats['HbA1c'].max() - cluster_stats['HbA1c'].min()) +
                                               (cluster_stats['Fasting_Blood_Glucose'] - cluster_stats[
                                                   'Fasting_Blood_Glucose'].min()) /
                                               (cluster_stats['Fasting_Blood_Glucose'].max() - cluster_stats[
                                                   'Fasting_Blood_Glucose'].min())
                                       ) / 2

logging.info("\nCluster Diabetes Risk Analysis:")
logging.info(
    cluster_stats[['KMeans_Cluster', 'Diabetes_Risk_Score']].sort_values('Diabetes_Risk_Score', ascending=False))

# Visualize diabetes risk by cluster
plt.figure(figsize=(12, 6))
sns.barplot(x='KMeans_Cluster', y='Diabetes_Risk_Score', data=cluster_stats)
plt.title('Diabetes Risk Score by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Risk Score (Higher = Higher Risk)')
plt.grid(axis='y')
plt.show()

# 5. Correlation Analysis
plt.figure(figsize=(14, 12))
correlation_matrix = df_with_clusters.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# 6. Summary and Conclusions
logging.info("\n=== Unsupervised Learning Analysis Summary ===")
logging.info(f"Dataset: Diabetes Prediction Dataset with {df.shape[0]} records")
logging.info(f"Optimal number of clusters identified: {optimal_k}")

# Describe the clusters
high_risk_cluster = cluster_stats.loc[cluster_stats['Diabetes_Risk_Score'].idxmax()]['KMeans_Cluster']
low_risk_cluster = cluster_stats.loc[cluster_stats['Diabetes_Risk_Score'].idxmin()]['KMeans_Cluster']

logging.info(f"\nHigh-risk cluster (Cluster {high_risk_cluster}):")
for col in ['Age', 'BMI', 'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic']:
    high_risk_value = cluster_stats.loc[cluster_stats['KMeans_Cluster'] == high_risk_cluster, col].values[0]
    logging.info(f"- Average {col}: {high_risk_value:.2f}")

logging.info(f"\nLow-risk cluster (Cluster {low_risk_cluster}):")
for col in ['Age', 'BMI', 'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic']:
    low_risk_value = cluster_stats.loc[cluster_stats['KMeans_Cluster'] == low_risk_cluster, col].values[0]
    logging.info(f"- Average {col}: {low_risk_value:.2f}")

# logging.info("\nKey findings:")
# logging.info("1. The dataset can be effectively segmented into distinct risk groups")
# logging.info("2. HbA1c and Fasting Blood Glucose are strong indicators for cluster separation")
# logging.info("3. Multiple patient profiles with different risk factors were identified")
# logging.info("4. There are clear correlations between certain health metrics and diabetes risk")

# Add after the correlation matrix section

# 5a. Advanced Feature Relationship Analysis
logging.info("\nAnalyzing key health marker relationships...")

# Create pairplot for key diabetes indicators
plt.figure(figsize=(16, 14))
sns.pairplot(df_with_clusters,
             vars=['HbA1c', 'Fasting_Blood_Glucose', 'BMI', 'Age', 'Waist_Circumference'],
             hue='KMeans_Cluster',
             diag_kind='kde',
             plot_kws={'alpha': 0.6})
plt.suptitle('Relationships Between Key Diabetes Markers by Cluster', y=1.02)
plt.show()

# Add lifestyle correlation analysis
lifestyle_cols = ['Physical_Activity_Level', 'Dietary_Intake_Calories', 'Alcohol_Consumption', 'Smoking_Status']
health_cols = ['HbA1c', 'Fasting_Blood_Glucose', 'BMI', 'Blood_Pressure_Systolic', 'Cholesterol_Total']

# Convert categorical lifestyle variables to numeric for correlation analysis
lifestyle_mapping = {
    'Physical_Activity_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Alcohol_Consumption': {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    'Smoking_Status': {'Never': 0, 'Former': 1, 'Current': 2}
}

# Create a copy for encoding
df_encoded = df_with_clusters.copy()
for col, mapping in lifestyle_mapping.items():
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map(mapping)

# Calculate lifestyle impact on health markers
lifestyle_health_corr = df_encoded[lifestyle_cols + health_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(lifestyle_health_corr.iloc[:len(lifestyle_cols), len(lifestyle_cols):],
            annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation between Lifestyle Factors and Health Markers')
plt.show()

# Ethnicity analysis
if 'Ethnicity' in df_with_clusters.columns:
    plt.figure(figsize=(14, 8))
    ethnic_hba1c = df_with_clusters.groupby('Ethnicity')[['HbA1c', 'Fasting_Blood_Glucose', 'BMI']].mean().sort_values(
        'HbA1c')
    ethnic_hba1c.plot(kind='bar')
    plt.title('Average Diabetes Markers by Ethnicity')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Gender differences analysis
if 'Sex' in df_with_clusters.columns:
    plt.figure(figsize=(12, 8))
    sex_indicators = df_with_clusters.groupby('Sex')[['HbA1c', 'Fasting_Blood_Glucose', 'BMI',
                                                      'Blood_Pressure_Systolic', 'Cholesterol_Total']].mean()
    sex_indicators.plot(kind='bar')
    plt.title('Gender Differences in Key Health Indicators')
    plt.ylabel('Average Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Age group analysis
# First create Age_Group
df_with_clusters['Age_Group'] = pd.cut(df_with_clusters['Age'],
                                       bins=[0, 30, 45, 60, 75, 100],
                                       labels=['<30', '30-45', '45-60', '60-75', '75+'])

df_with_clusters['Diabetes_Risk_Score'] = (
                                                  (df_with_clusters['HbA1c'] - df_with_clusters['HbA1c'].min()) /
                                                  (df_with_clusters['HbA1c'].max() - df_with_clusters['HbA1c'].min()) +
                                                  (df_with_clusters['Fasting_Blood_Glucose'] - df_with_clusters[
                                                      'Fasting_Blood_Glucose'].min()) /
                                                  (df_with_clusters['Fasting_Blood_Glucose'].max() - df_with_clusters[
                                                      'Fasting_Blood_Glucose'].min())
                                          ) / 2

# Then do the groupby with only columns that exist
age_analysis = df_with_clusters.groupby('Age_Group', observed=True)[
    ['HbA1c', 'Fasting_Blood_Glucose', 'Diabetes_Risk_Score']].mean()

# Display the results
plt.figure(figsize=(12, 6))
age_analysis.plot(kind='bar')
plt.title('Diabetes Markers by Age Group')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylabel('Value')
plt.show()
# Calculate composite metabolic health score
df_with_clusters['Metabolic_Health_Score'] = (
                                                     (df_with_clusters['BMI'] - df_with_clusters['BMI'].min()) / (
                                                     df_with_clusters['BMI'].max() - df_with_clusters['BMI'].min()) +
                                                     (df_with_clusters['Blood_Pressure_Systolic'] - df_with_clusters[
                                                         'Blood_Pressure_Systolic'].min()) /
                                                     (df_with_clusters['Blood_Pressure_Systolic'].max() -
                                                      df_with_clusters['Blood_Pressure_Systolic'].min()) +
                                                     (df_with_clusters['Cholesterol_Total'] - df_with_clusters[
                                                         'Cholesterol_Total'].min()) /
                                                     (df_with_clusters['Cholesterol_Total'].max() - df_with_clusters[
                                                         'Cholesterol_Total'].min())
                                             ) / 3

# Add to cluster statistics
cluster_stats['Metabolic_Health_Score'] = df_with_clusters.groupby('KMeans_Cluster')[
    'Metabolic_Health_Score'].mean().values

# Add visualization comparing diabetes risk vs metabolic health by cluster
plt.figure(figsize=(10, 8))
plt.scatter(cluster_stats['Diabetes_Risk_Score'],
            cluster_stats['Metabolic_Health_Score'],
            s=200, alpha=0.7)

# Label each point with its cluster number
for i, row in cluster_stats.iterrows():
    plt.annotate(f"Cluster {int(row['KMeans_Cluster'])}",
                 (row['Diabetes_Risk_Score'], row['Metabolic_Health_Score']),
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Diabetes Risk Score')
plt.ylabel('Metabolic Health Score')
plt.title('Metabolic Health vs Diabetes Risk by Cluster')
plt.grid(True, alpha=0.3)
plt.show()
