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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
import umap.umap_ as umap
import logging
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("diabetes_clustering_analysis.log"),
        logging.StreamHandler()
    ]
)

# Set seed for reproducibility
np.random.seed(42)

# Load dataset
# Try to use the path from your original code
try:
    df = pd.read_csv("diabetes_dataset.csv")
    logging.info(f"Dataset loaded successfully from diabetes_dataset.csv")
except:
    try:
        # Alternate path mentioned in your code
        df = pd.read_csv("diabetes_prediction_dataset")
        logging.info(f"Dataset loaded successfully from diabetes_prediction_dataset")
    except Exception as e:
        logging.error(f"Error loading the dataset: {e}")
        # If you can't load the dataset, insert the correct path here
        raise

# Basic dataset information
logging.info(f"Dataset shape: {df.shape}")
logging.info(f"Columns: {df.columns.tolist()}")
logging.info(f"Sample data:\n{df.head()}")
logging.info(f"Missing values:\n{df.isnull().sum()}")

# None is atype in
if 'Alcohol_Consumption' in df.columns:
    df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove any unnamed index columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in numerical_cols:
        numerical_cols.remove('Unnamed: 0')

logging.info(f"Categorical columns: {categorical_cols}")
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
logging.info(f"Processed data shape: {X_processed.shape}")


# Function to plot clusters
def plot_clusters(X_2d, labels, title, filename=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 0], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# Function to plot silhouette scores
def plot_silhouette_heatmap(results_df, optimal_n_component, optimal_k, title, filename=None):
    plt.figure(figsize=(12, 8))

    # Create pivot table for heatmap
    heatmap_data = results_df.pivot_table(
        index='n_component',
        columns='k',
        values='score'
    )

    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                     linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    # Highlight optimal parameters if they exist in the heatmap
    if optimal_n_component in heatmap_data.index and optimal_k in heatmap_data.columns:
        optimal_row = list(heatmap_data.index).index(optimal_n_component)
        optimal_col = list(heatmap_data.columns).index(optimal_k)
        ax.add_patch(plt.Rectangle((optimal_col, optimal_row), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(title)
    plt.ylabel('PCA Components')
    plt.xlabel('Number of Clusters (k)')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# 1. DIMENSION REDUCTION & VISUALIZATION
# =============================================================================

# PCA analysis
logging.info("Performing PCA analysis...")

# Analyze all components first
pca_full = PCA()
pca_full.fit(X_processed)
explained_variance = pca_full.explained_variance_ratio_

# Plot variance explained
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% explained variance')
plt.grid(True)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance by Components')
plt.legend()
plt.savefig('pca_explained_variance.png', dpi=300, bbox_inches='tight')
plt.show()

# Find number of components for 95% explained variance
n_components_95 = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
logging.info(f"Number of components needed for 95% variance: {n_components_95}")

# Create 2D projections for visualization
# PCA (2D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_processed)
logging.info(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_}")

# t-SNE
logging.info("Performing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_processed)

# UMAP
logging.info("Performing UMAP...")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_processed)

# Visualize the 2D projections
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# PCA plot
axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6)
axes[0].set_title(
    f'PCA (2D)\nPC1: {pca_2d.explained_variance_ratio_[0]:.2%}, PC2: {pca_2d.explained_variance_ratio_[1]:.2%}')
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
plt.savefig('dimension_reduction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 2. GRID SEARCH FOR OPTIMAL CLUSTERING PARAMETERS
# =============================================================================

# Initialize variables to track results
silhouette_scores = []
n_components_values = []
k_values = []
max_silhouette_score = 0
optimal_n_component = 0
optimal_k = 0
optimal_pca = None

# Define parameter ranges
n_components_list = [n_components_95] + list(range(5, min(30, X_processed.shape[1]), 5))  # PCA components
k_range = range(2, 11)  # K-means clusters

# Run grid search over PCA components and number of clusters
results = []
for n_component in n_components_list:
    if isinstance(n_component, float) and n_component < 1:
        # For percentage of variance explained
        pca = PCA(n_components=n_component)
    else:
        # For specific number of components
        pca = PCA(n_components=min(n_component, X_processed.shape[1] - 1))

    X_pca = pca.fit_transform(X_processed)

    if isinstance(n_component, float):
        actual_components = X_pca.shape[1]
        logging.info(f"Using {actual_components} components to preserve {n_component * 100:.0f}% of variance")
    else:
        actual_components = n_component
        logging.info(f"Using {actual_components} components")

    for k in k_range:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_pca)

        # Calculate silhouette score
        score = silhouette_score(X_pca, labels)

        # Store results
        results.append({
            'n_component': actual_components if not isinstance(n_component, float) else n_component,
            'k': k,
            'score': score,
            'labels': labels,
            'pca_data': X_pca
        })

        # Track optimal parameters
        if score > max_silhouette_score:
            max_silhouette_score = score
            optimal_n_component = actual_components if not isinstance(n_component, float) else n_component
            optimal_k = k
            optimal_pca = X_pca

        logging.info(f"PCA n_component={n_component}, K={k}, Silhouette Score: {score:.4f}")

# Create DataFrame for easier manipulation
results_df = pd.DataFrame([(r['n_component'], r['k'], r['score']) for r in results],
                          columns=['n_component', 'k', 'score'])

# Log optimal parameters
logging.info(f"Optimal parameters: PCA n_component={optimal_n_component}, K={optimal_k}")
logging.info(f"Best silhouette score: {max_silhouette_score:.4f}")

# Create heatmap visualization
plot_silhouette_heatmap(
    results_df,
    optimal_n_component,
    optimal_k,
    'Silhouette Scores for Different PCA Components and K-Means Clusters',
    'pca_kmeans_heatmap.png'
)

# Get the best result and its data
best_result = [r for r in results if (
    r['n_component'] == optimal_n_component if not isinstance(optimal_n_component, float)
    else r['n_component'] == optimal_n_component) and r['k'] == optimal_k][0]

best_pca_data = best_result['pca_data']
best_labels = best_result['labels']

# =============================================================================
# 3. COMPARE DIFFERENT CLUSTERING ALGORITHMS
# =============================================================================

# Now that we have the optimal number of components, compare different clustering algorithms
logging.info("Comparing different clustering algorithms...")

# Use the optimal PCA n_components
pca_optimal = PCA(n_components=optimal_n_component if not isinstance(optimal_n_component, float) else None)
if isinstance(optimal_n_component, float):
    pca_optimal.set_params(n_components=optimal_n_component)

X_pca_optimal = pca_optimal.fit_transform(X_processed)

# 1. K-means (already done)
kmeans_labels = best_labels
kmeans_silhouette = max_silhouette_score

# 2. Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_pca_optimal)
hierarchical_silhouette = silhouette_score(X_pca_optimal, hierarchical_labels)
logging.info(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.4f}")

# 3. DBSCAN
# For DBSCAN, we need to find a good eps value
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=min(10, len(X_pca_optimal) - 1))
nn.fit(X_pca_optimal)
distances, indices = nn.kneighbors(X_pca_optimal)
distances = np.sort(distances[:, -1])

# Plot the k-distance graph to find the elbow
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('k-th nearest neighbor distance')
plt.title('K-distance Graph for DBSCAN eps Parameter Selection')
plt.grid(True)
plt.savefig('dbscan_kdistance_graph.png', dpi=300, bbox_inches='tight')
plt.show()

# Find the elbow point (you may need to adjust this heuristic)
knee_point = np.diff(np.diff(distances))
elbow_index = np.argmax(knee_point) + 1
eps_value = distances[elbow_index]
logging.info(f"Selected DBSCAN eps value: {eps_value:.4f}")

# Run DBSCAN with the selected eps
dbscan = DBSCAN(eps=eps_value, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca_optimal)

# Handle the case if DBSCAN returns mostly noise (-1)
if len(np.unique(dbscan_labels)) <= 1 or -1 in np.unique(dbscan_labels):
    # Adjust eps to get more clusters
    eps_attempts = [eps_value * factor for factor in [0.5, 0.75, 1.25, 1.5, 2]]
    best_eps = eps_value
    best_n_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
    best_noise_ratio = np.sum(dbscan_labels == -1) / len(dbscan_labels)

    for eps in eps_attempts:
        temp_dbscan = DBSCAN(eps=eps, min_samples=5)
        temp_labels = temp_dbscan.fit_predict(X_pca_optimal)
        n_clusters = len(np.unique(temp_labels[temp_labels != -1]))
        noise_ratio = np.sum(temp_labels == -1) / len(temp_labels)

        logging.info(f"DBSCAN with eps={eps:.4f}: {n_clusters} clusters, {noise_ratio:.2%} noise")

        # Better result has more clusters and less noise
        if n_clusters > best_n_clusters and noise_ratio < 0.5:
            best_eps = eps
            best_n_clusters = n_clusters
            best_noise_ratio = noise_ratio
            dbscan_labels = temp_labels

    logging.info(f"Selected better DBSCAN eps value: {best_eps:.4f}")

# Calculate DBSCAN silhouette (if there are multiple clusters and not all noise)
if len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels):
    dbscan_silhouette = silhouette_score(X_pca_optimal, dbscan_labels)
elif len(np.unique(dbscan_labels)) > 1:
    # Calculate silhouette only on non-noise points
    non_noise = dbscan_labels != -1
    if np.sum(non_noise) > 1 and len(np.unique(dbscan_labels[non_noise])) > 1:
        dbscan_silhouette = silhouette_score(X_pca_optimal[non_noise], dbscan_labels[non_noise])
    else:
        dbscan_silhouette = 0
else:
    dbscan_silhouette = 0
logging.info(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")

# 4. Gaussian Mixture Model
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_pca_optimal)
gmm_silhouette = silhouette_score(X_pca_optimal, gmm_labels)
logging.info(f"GMM Silhouette Score: {gmm_silhouette:.4f}")

# Compare silhouette scores
algorithms = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
silhouette_scores = [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette, gmm_silhouette]

plt.figure(figsize=(10, 6))
bars = plt.bar(algorithms, silhouette_scores, color=['blue', 'red', 'green', 'purple'])
plt.xlabel('Clustering Algorithm')
plt.ylabel('Silhouette Score')
plt.title('Comparison of Clustering Algorithms')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the scores as text above the bars
for bar, score in zip(bars, silhouette_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom')

plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Pick the best algorithm based on silhouette score
best_algorithm_index = np.argmax(silhouette_scores)
best_algorithm = algorithms[best_algorithm_index]
best_algorithm_silhouette = silhouette_scores[best_algorithm_index]
logging.info(f"Best clustering algorithm: {best_algorithm} with silhouette score {best_algorithm_silhouette:.4f}")

# Get the labels from the best algorithm
if best_algorithm == 'K-means':
    best_algorithm_labels = kmeans_labels
elif best_algorithm == 'Hierarchical':
    best_algorithm_labels = hierarchical_labels
elif best_algorithm == 'DBSCAN':
    best_algorithm_labels = dbscan_labels
else:  # GMM
    best_algorithm_labels = gmm_labels

# Visualize the best clustering result on the 2D PCA plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca_optimal[:, 0], X_pca_optimal[:, 1], c=best_algorithm_labels,
                      cmap='viridis', alpha=0.7, s=10)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Best Clustering Result: {best_algorithm} (Silhouette Score: {best_algorithm_silhouette:.4f})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('best_clustering_result.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. ANOMALY DETECTION
# =============================================================================

logging.info("Performing anomaly detection...")

# 1. K-means distance-based anomaly detection
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_pca_optimal)
distances = kmeans.transform(X_pca_optimal).min(axis=1)

# Define anomalies as points with distance > mean + 3*std
anomaly_threshold_kmeans = distances.mean() + 3 * distances.std()
anomalies_kmeans = distances > anomaly_threshold_kmeans
anomaly_ratio_kmeans = anomalies_kmeans.mean()
logging.info(f"K-means anomalies: {anomaly_ratio_kmeans:.2%} of data")

# 2. GMM log-likelihood anomaly detection
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm.fit(X_pca_optimal)
log_likelihood = gmm.score_samples(X_pca_optimal)

# Define anomalies as points with log likelihood < mean - 3*std
anomaly_threshold_gmm = log_likelihood.mean() - 3 * log_likelihood.std()
anomalies_gmm = log_likelihood < anomaly_threshold_gmm
anomaly_ratio_gmm = anomalies_gmm.mean()
logging.info(f"GMM anomalies: {anomaly_ratio_gmm:.2%} of data")

# 3. One-Class SVM anomaly detection
svm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')  # nu is approximately the proportion of outliers
svm.fit(X_pca_optimal)
svm_scores = svm.decision_function(X_pca_optimal)
anomalies_svm = svm.predict(X_pca_optimal) == -1
anomaly_ratio_svm = anomalies_svm.mean()
logging.info(f"One-Class SVM anomalies: {anomaly_ratio_svm:.2%} of data")

# Plot histograms of anomaly scores
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# K-means distance histogram
axes[0].hist(distances, bins=50, alpha=0.7, color='blue')
axes[0].axvline(x=anomaly_threshold_kmeans, color='red', linestyle='--',
                label=f'Threshold: {anomaly_threshold_kmeans:.2f}')
axes[0].set_title(f'K-means Distance Scores\n({anomaly_ratio_kmeans:.2%} anomalies)')
axes[0].set_xlabel('Distance to Nearest Centroid')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# GMM log-likelihood histogram
axes[1].hist(log_likelihood, bins=50, alpha=0.7, color='green')
axes[1].axvline(x=anomaly_threshold_gmm, color='red', linestyle='--',
                label=f'Threshold: {anomaly_threshold_gmm:.2f}')
axes[1].set_title(f'GMM Log-Likelihood Scores\n({anomaly_ratio_gmm:.2%} anomalies)')
axes[1].set_xlabel('Log-Likelihood')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# One-Class SVM decision function histogram
axes[2].hist(svm_scores, bins=50, alpha=0.7, color='purple')
axes[2].axvline(x=0, color='red', linestyle='--', label='Threshold: 0')
axes[2].set_title(f'One-Class SVM Decision Scores\n({anomaly_ratio_svm:.2%} anomalies)')
axes[2].set_xlabel('Decision Function Value')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig('anomaly_detection_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize anomalies in 2D PCA space
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# K-means anomalies
axes[0].scatter(X_pca_optimal[~anomalies_kmeans, 0], X_pca_optimal[~anomalies_kmeans, 1],
                c='blue', alpha=0.5, s=10, label='Normal')
axes[0].scatter(X_pca_optimal[anomalies_kmeans, 0], X_pca_optimal[anomalies_kmeans, 1],
                c='red', alpha=0.9, s=20, label='Anomaly')
axes[0].set_title(f'K-means Anomalies ({anomaly_ratio_kmeans:.2%})')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend()

# GMM anomalies
axes[1].scatter(X_pca_optimal[~anomalies_gmm, 0], X_pca_optimal[~anomalies_gmm, 1],
                c='blue', alpha=0.5, s=10, label='Normal')
axes[1].scatter(X_pca_optimal[anomalies_gmm, 0], X_pca_optimal[anomalies_gmm, 1],
                c='red', alpha=0.9, s=20, label='Anomaly')
axes[1].set_title(f'GMM Anomalies ({anomaly_ratio_gmm:.2%})')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend()

# One-Class SVM anomalies
axes[2].scatter(X_pca_optimal[~anomalies_svm, 0], X_pca_optimal[~anomalies_svm, 1],
                c='blue', alpha=0.5, s=10, label='Normal')
axes[2].scatter(X_pca_optimal[anomalies_svm, 0], X_pca_optimal[anomalies_svm, 1],
                c='red', alpha=0.9, s=20, label='Anomaly')
axes[2].set_title(f'One-Class SVM Anomalies ({anomaly_ratio_svm:.2%})')
axes[2].set_xlabel('Principal Component 1')
axes[2].set_ylabel('Principal Component 2')
axes[2].legend()

plt.tight_layout()
plt.savefig('anomaly_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. IDENTIFY CLUSTER CHARACTERISTICS
# =============================================================================

logging.info("Analyzing cluster characteristics...")

# Create a dataframe with the original data and cluster labels
df_with_clusters = df.copy()
df_with_clusters['cluster'] = best_algorithm_labels

# Analyze cluster characteristics for numerical columns
cluster_stats = {}
for column in numerical_cols:
    cluster_stats[column] = df_with_clusters.groupby('cluster')[column].agg(['mean', 'std', 'min', 'max'])

    # Plot distribution by cluster
    plt.figure(figsize=(12, 6))
    for cluster in np.unique(best_algorithm_labels):
        sns.kdeplot(df_with_clusters[df_with_clusters['cluster'] == cluster][column],
                    label=f'Cluster {cluster}')

    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'cluster_distribution_{column}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Display summary statistics for each cluster
for column in numerical_cols:
    logging.info(f"\nCluster statistics for {column}:")
    logging.info(cluster_stats[column])

# For categorical columns, calculate proportions in each cluster
for column in categorical_cols:
    logging.info(f"\nCluster proportions for {column}:")
    proportions = df_with_clusters.groupby('cluster')[column].value_counts(normalize=True).unstack().fillna(0)
    logging.info(proportions)

    # Plot proportions
    proportions.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Proportion of {column} Categories by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title=column)
    plt.tight_layout()
    plt.savefig(f'cluster_proportions_{column}.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 6. UMAP OPTIMIZATION FOR VISUALIZATION
# =============================================================================

logging.info("Optimizing UMAP parameters for visualization...")

# Define UMAP parameter ranges
n_neighbors_options = [5, 15, 30, 50]
min_dist_options = [0.0, 0.1, 0.25, 0.5]
n_components_options = range(2, 6)  # Try 2 to 5 components
n_clusters_options = range(2, 6)  # Number of clusters to try
umap_results = []

# Find optimal UMAP parameters with different component counts
print("Starting UMAP grid search over n_neighbors, min_dist, n_components, and n_clusters...")
print(f"Testing n_neighbors options: {n_neighbors_options}")
print(f"Testing min_dist options: {min_dist_options}")
print(f"Testing n_components options: {list(n_components_options)}")
print(f"Testing n_clusters options: {list(n_clusters_options)}")

# Create a progress counter
total_iterations = len(n_neighbors_options) * len(min_dist_options) * len(n_components_options) * len(
    n_clusters_options)
progress_count = 0

for n_components in n_components_options:
    for n_clusters in n_clusters_options:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                progress_count += 1
                print(f"Progress: {progress_count}/{total_iterations} "
                      f"(n_components={n_components}, n_clusters={n_clusters}, "
                      f"n_neighbors={n_neighbors}, min_dist={min_dist:.2f})")

                # Create UMAP embedding
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    n_components=n_components, random_state=42)
                X_umap = reducer.fit_transform(X_processed)

                # Apply the best clustering algorithm to the UMAP embedding
                if best_algorithm == 'K-means':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                elif best_algorithm == 'Hierarchical':
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                elif best_algorithm == 'DBSCAN':
                    model = DBSCAN(eps=eps_value, min_samples=5)
                else:  # GMM
                    model = GaussianMixture(n_components=n_clusters, random_state=42)

                umap_labels = model.fit_predict(X_umap)

                # Calculate silhouette score
                if len(np.unique(umap_labels)) > 1:  # Only calculate if there are multiple clusters
                    # For n_components > 3, calculate silhouette directly
                    score = silhouette_score(X_umap, umap_labels)
                else:
                    score = 0

                umap_results.append({
                    'n_components': n_components,
                    'n_clusters': n_clusters,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'score': score,
                    'embedding': X_umap,
                    'labels': umap_labels
                })

                logging.info(
                    f"UMAP(n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}) + Clusters(k={n_clusters}): Silhouette Score: {score:.4f}")

# Find the best UMAP parameters
best_umap_result = max(umap_results, key=lambda x: x['score'])
best_n_components = best_umap_result['n_components']
best_n_clusters = best_umap_result['n_clusters']
best_n_neighbors = best_umap_result['n_neighbors']
best_min_dist = best_umap_result['min_dist']
best_umap_score = best_umap_result['score']
best_umap_embedding = best_umap_result['embedding']
best_umap_labels = best_umap_result['labels']

logging.info(
    f"Best UMAP parameters: n_components={best_n_components}, n_clusters={best_n_clusters}, n_neighbors={best_n_neighbors}, min_dist={best_min_dist}")
logging.info(f"Best UMAP silhouette score: {best_umap_score:.4f}")

# Create comprehensive plots for UMAP results
print("Creating UMAP parameter heatmaps and visualizations...")

# Create a DataFrame from the results
umap_df = pd.DataFrame([{
    'n_components': r['n_components'],
    'n_clusters': r['n_clusters'],
    'n_neighbors': r['n_neighbors'],
    'min_dist': r['min_dist'],
    'score': r['score']
} for r in umap_results])

# 1. Create heatmaps for various parameter combinations

# For each n_components value, create a series of heatmaps for different n_clusters
for n_comp in n_components_options:
    for n_clust in n_clusters_options:
        # Filter data for this n_components and n_clusters value
        filtered_df = umap_df[(umap_df['n_components'] == n_comp) & (umap_df['n_clusters'] == n_clust)]

        if len(filtered_df) > 0:
            # Create pivot table for heatmap
            heatmap_data = filtered_df.pivot_table(
                index='n_neighbors',
                columns='min_dist',
                values='score'
            )

            plt.figure(figsize=(10, 7))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                        linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

            plt.title(f'UMAP Parameter Optimization\n(n_components={n_comp}, n_clusters={n_clust})')
            plt.ylabel('n_neighbors')
            plt.xlabel('min_dist')
            plt.tight_layout()
            plt.savefig(f'umap_heatmap_ncomp_{n_comp}_nclust_{n_clust}.png', dpi=300, bbox_inches='tight')
            plt.show()

# 2. For each n_components and n_clusters combination, find best results
best_configs = []
for n_comp in n_components_options:
    for n_clust in n_clusters_options:
        # Filter results for this n_components and n_clusters
        comp_clust_results = [r for r in umap_results if
                              r['n_components'] == n_comp and
                              r['n_clusters'] == n_clust]

        if comp_clust_results:
            # Find best result for this combination
            best_result = max(comp_clust_results, key=lambda x: x['score'])
            best_configs.append({
                'n_components': n_comp,
                'n_clusters': n_clust,
                'n_neighbors': best_result['n_neighbors'],
                'min_dist': best_result['min_dist'],
                'score': best_result['score']
            })

# Create a heatmap of n_components vs n_clusters
if best_configs:
    best_configs_df = pd.DataFrame(best_configs)
    heatmap_component_clusters = best_configs_df.pivot_table(
        index='n_components',
        columns='n_clusters',
        values='score'
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_component_clusters, annot=True, fmt='.3f', cmap='viridis',
                linewidths=.5, cbar_kws={'label': 'Silhouette Score'})

    # Highlight the best parameters
    if best_n_components in heatmap_component_clusters.index and best_n_clusters in heatmap_component_clusters.columns:
        best_row = list(heatmap_component_clusters.index).index(best_n_components)
        best_col = list(heatmap_component_clusters.columns).index(best_n_clusters)
        plt.gca().add_patch(plt.Rectangle((best_col, best_row), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title('Best Silhouette Scores by Components and Clusters')
    plt.ylabel('Number of UMAP Components')
    plt.xlabel('Number of Clusters')
    plt.tight_layout()
    plt.savefig('umap_components_vs_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. Create comparison visualizations for n_components vs score and n_clusters vs score
# First, group by n_components and find max score for each
component_scores = umap_df.groupby('n_components')['score'].max().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(component_scores['n_components'], component_scores['score'], color='skyblue')
plt.xlabel('Number of UMAP Components')
plt.ylabel('Best Silhouette Score')
plt.title('Best Silhouette Score by Number of UMAP Components')
plt.xticks(component_scores['n_components'])
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels above bars
for i, row in component_scores.iterrows():
    plt.text(row['n_components'], row['score'] + 0.01, f'{row["score"]:.4f}', ha='center')

plt.tight_layout()
plt.savefig('umap_components_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Next, group by n_clusters and find max score for each
cluster_scores = umap_df.groupby('n_clusters')['score'].max().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(cluster_scores['n_clusters'], cluster_scores['score'], color='lightgreen')
plt.xlabel('Number of Clusters')
plt.ylabel('Best Silhouette Score')
plt.title('Best Silhouette Score by Number of Clusters')
plt.xticks(cluster_scores['n_clusters'])
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels above bars
for i, row in cluster_scores.iterrows():
    plt.text(row['n_clusters'], row['score'] + 0.01, f'{row["score"]:.4f}', ha='center')

plt.tight_layout()
plt.savefig('umap_clusters_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. For 2D and 3D visualizations of the best result
if best_n_components in [2, 3]:
    # Visualize the best UMAP embedding with clusters
    if best_n_components == 2:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(best_umap_embedding[:, 0], best_umap_embedding[:, 1],
                              c=best_umap_labels, cmap='viridis', alpha=0.7, s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.title(
            f'Best UMAP Visualization (2D)\n'
            f'n_components={best_n_components}, n_clusters={best_n_clusters}, '
            f'n_neighbors={best_n_neighbors}, min_dist={best_min_dist:.2f}, '
            f'Silhouette={best_umap_score:.4f}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('best_umap_clustering_2d.png', dpi=300, bbox_inches='tight')
        plt.show()

    elif best_n_components == 3:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            best_umap_embedding[:, 0],
            best_umap_embedding[:, 1],
            best_umap_embedding[:, 2],
            c=best_umap_labels, cmap='viridis', alpha=0.7, s=10
        )
        plt.colorbar(scatter, label='Cluster')
        ax.set_title(
            f'Best UMAP Visualization (3D)\n'
            f'n_components={best_n_components}, n_clusters={best_n_clusters}, '
            f'n_neighbors={best_n_neighbors}, min_dist={best_min_dist:.2f}, '
            f'Silhouette={best_umap_score:.4f}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        plt.tight_layout()
        plt.savefig('best_umap_clustering_3d.png', dpi=300, bbox_inches='tight')
        plt.show()

# Create interactive plots to explore parameter combinations (if needed)
# For comprehensive exploration, we'll also create a pairwise visualization
# of how different parameter combinations affect the silhouette score

# Create a grid of scatterplots for parameter pairs
param_pairs = [
    ('n_components', 'n_clusters'),
    ('n_components', 'n_neighbors'),
    ('n_components', 'min_dist'),
    ('n_clusters', 'n_neighbors'),
    ('n_clusters', 'min_dist'),
    ('n_neighbors', 'min_dist')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (param1, param2) in enumerate(param_pairs):
    # Create scatter plot
    scatter = axes[i].scatter(
        umap_df[param1],
        umap_df[param2],
        c=umap_df['score'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )

    # Add labels and title
    axes[i].set_xlabel(param1)
    axes[i].set_ylabel(param2)
    axes[i].set_title(f'{param1} vs {param2}')

    # Add grid
    axes[i].grid(True, linestyle='--', alpha=0.5)

    # Mark the best parameter combination
    best_x = best_umap_result[param1]
    best_y = best_umap_result[param2]
    axes[i].scatter([best_x], [best_y], color='red', s=100, edgecolor='black', zorder=10)
    axes[i].annotate('Best', (best_x, best_y), xytext=(10, 10),
                     textcoords='offset points', color='red',
                     fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

# Add a colorbar
cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', pad=0.01)
cbar.set_label('Silhouette Score')

plt.suptitle('UMAP Parameter Pair Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('umap_parameter_pairs.png', dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# 7. FINAL EVALUATION AND SUMMARY
# =============================================================================

logging.info("Generating final evaluation and summary...")

# 1. Compare silhouette scores across all methods
all_methods = {
    'PCA + K-means': kmeans_silhouette,
    'PCA + Hierarchical': hierarchical_silhouette,
    'PCA + DBSCAN': dbscan_silhouette,
    'PCA + GMM': gmm_silhouette,
    'UMAP + Best Algorithm': best_umap_score
}

plt.figure(figsize=(12, 6))
methods = list(all_methods.keys())
scores = list(all_methods.values())
bars = plt.bar(methods, scores, color=['blue', 'red', 'green', 'purple', 'orange'])
plt.xlabel('Method')
plt.ylabel('Silhouette Score')
plt.title('Comparison of All Clustering Methods')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the scores as text above the bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('all_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Determine the overall best method
best_method = max(all_methods.items(), key=lambda x: x[1])
logging.info(f"Overall best method: {best_method[0]} with silhouette score {best_method[1]:.4f}")

# 3. Generate cluster profiles for the best method
if best_method[0] == 'UMAP + Best Algorithm':
    final_labels = best_umap_labels
    X_final = best_umap_embedding
else:
    final_labels = best_algorithm_labels
    X_final = X_pca_optimal

# Create a dataframe with the original data and final cluster labels
final_df = df.copy()
final_df['cluster'] = final_labels

# Summary statistics for each cluster
logging.info("\n===== FINAL CLUSTER PROFILES =====")
for cluster in np.unique(final_labels):
    cluster_data = final_df[final_df['cluster'] == cluster]
    logging.info(f"\nCluster {cluster} ({len(cluster_data)} samples, {len(cluster_data) / len(final_df):.2%} of data):")

    # Numerical features
    for col in numerical_cols:
        logging.info(f"{col}: mean={cluster_data[col].mean():.2f}, std={cluster_data[col].std():.2f}")

    # Top 3 categories for each categorical feature
    for col in categorical_cols:
        top_categories = cluster_data[col].value_counts(normalize=True).nlargest(3)
        logging.info(f"{col} top categories: {', '.join([f'{cat}: {val:.2%}' for cat, val in top_categories.items()])}")

# Save results to file
final_df.to_csv('diabetes_clustering_results.csv', index=False)
logging.info("Results saved to diabetes_clustering_results.csv")

# Final visualization of clusters with their profiles
plt.figure(figsize=(14, 12))

# Create scatter plot
scatter = plt.scatter(X_final[:, 0], X_final[:, 1], c=final_labels, cmap='viridis', alpha=0.7, s=10)
plt.colorbar(scatter, label='Cluster')

# Add cluster annotations
for cluster in np.unique(final_labels):
    # Find the centroid of this cluster
    mask = final_labels == cluster
    centroid_x = X_final[mask, 0].mean()
    centroid_y = X_final[mask, 1].mean()

    # Create annotation text
    cluster_size = np.sum(mask)
    cluster_pct = cluster_size / len(final_labels) * 100
    annotation = f"Cluster {cluster}\n{cluster_size} samples ({cluster_pct:.1f}%)"

    # Add annotation
    plt.annotate(annotation, (centroid_x, centroid_y),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                 ha='center', va='center', fontsize=9)

plt.title(f'Final Clustering Result: {best_method[0]} (Silhouette Score: {best_method[1]:.4f})')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('final_clustering_result.png', dpi=300, bbox_inches='tight')
plt.show()

logging.info("Analysis complete!")