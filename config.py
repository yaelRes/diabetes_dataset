"""
Configuration file for diabetes clustering analysis.
Contains feature sets and other configuration parameters.
"""

# Define different feature sets for comparative analysis
feature_sets = {
    "All Features": None,  # None means use all features

    "Top 10 Important Features": [
        "Fasting_Blood_Glucose",
        "HbA1c",
        "BMI",
        "Age",
        "Waist_Circumference",
        "Blood_Pressure_Systolic",
        "Cholesterol_HDL",
        "Family_History_of_Diabetes",
        "Physical_Activity_Level",
        "Cholesterol_LDL"
    ],

    # "Metabolic Features Only": [
    #     "Fasting_Blood_Glucose",
    #     "HbA1c",
    #     "BMI",
    #     "Cholesterol_HDL",
    #     "Cholesterol_LDL",
    #     "Triglycerides",
    #     "Insulin_Level",
    #     "HOMA_IR",`
    #     "C_Reactive_Protein"
    # ],
    #
    # "Clinical Measurements": [
    #     "HbA1c",
    #     "Age",
    #     "Weight",
    #     "Height",
    #     "BMI",
    #     "Waist_Circumference",
    #     "Blood_Pressure_Systolic",
    #     "Blood_Pressure_Diastolic"
    # ],
    #
    # "Lifestyle Factors": [
    #     "HbA1c",
    #     "Physical_Activity_Level",
    #     "Smoking_Status",
    #     "Alcohol_Consumption",
    #     "Diet_Quality",
    #     "Sleep_Duration",
    #     "Stress_Level"
    # ],
    #
    # "Genetic and Demographics": [
    #     "HbA1c",
    #     "Age",
    #     "Gender",
    #     "Ethnicity",
    #     "Family_History_of_Diabetes"
    # ],
    #
    # "Minimalist Set": [
    #     "Fasting_Blood_Glucose",
    #     "HbA1c",
    #     "BMI",
    #     "Age"
    # ]
}

# Configuration for PCA analysis
PCA_CONFIG = {
    "n_components_range": range(2, 30, 2),  # Range of PCA components to try
    "variance_threshold_95": 0.95,          # Threshold for 95% variance explained
    "variance_threshold_99": 0.99           # Threshold for 99% variance explained
}

# Configuration for clustering algorithms
CLUSTERING_CONFIG = {
    "k_range": range(2, 11),                # Range of cluster counts to try
    "dbscan_eps_factors": [0.5, 0.75, 1.25, 1.5, 2],  # Factors to adjust DBSCAN eps
    "random_state": 42                      # Random seed for reproducibility
}

TSNE_CONFIG = {
    "n_components_options": range(2, 6),     # Range of t-SNE components to try
    "perplexity_options": [5, 15, 30, 50, 100],  # Values for perplexity parameter
    "n_clusters_options": range(5, 6),
    "random_state": 42                       # Random seed for reproducibility
}

TSNE_CONFIG = {
    "n_components_options": range(2, 3),     # Range of t-SNE components to try
    "perplexity_options": [15],  # Values for perplexity parameter
    "n_clusters_options": range(5, 6),
    "random_state": 42                       # Random seed for reproducibility
}

# Configuration for UMAP optimization
UMAP_CONFIG = {
    "n_neighbors_options": [5, 15, 30, 50], # Values for n_neighbors parameter
    "min_dist_options": [0.0, 0.1, 0.25, 0.5],  # Values for min_dist parameter
    "n_components_options": range(2, 6),    # Range of UMAP components to try
    "n_clusters_options": range(2, 6),
    "random_state": 42                      # Random seed for reproducibility
}

UMAP_CONFIG = {
    "n_neighbors_options": [50], # Values for n_neighbors parameter
    "min_dist_options": [0.5],  # Values for min_dist parameter
    "n_components_options": range(5, 6),    # Range of UMAP components to try
    "n_clusters_options": range(5, 6),
    "random_state": 42                      # Random seed for reproducibility
}

PCA_CONFIG = {
    "n_components_range": [2],  # Range of PCA components to try
    "variance_threshold_95": 0.95,          # Threshold for 95% variance explained
    "variance_threshold_99": 0.99           # Threshold for 99% variance explained
}

# Configuration for clustering algorithms
CLUSTERING_CONFIG = {
    "k_range": [2],                # Range of cluster counts to try
    "dbscan_eps_factors": [2],  # Factors to adjust DBSCAN eps
    "random_state": 42                      # Random seed for reproducibility
}
