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

    "Metabolic Features Only": [
        "Fasting_Blood_Glucose",
        "HbA1c",
        "BMI",
        "Cholesterol_HDL",
        "Cholesterol_LDL",
        "Triglycerides",
        "Insulin_Level",
        "HOMA_IR",
        "C_Reactive_Protein"
    ],

    "Clinical Measurements": [
        "HbA1c",
        "Age",
        "Weight",
        "Height",
        "BMI",
        "Waist_Circumference",
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic"
    ],

    "Lifestyle Factors": [
        "HbA1c",
        "Physical_Activity_Level",
        "Smoking_Status",
        "Alcohol_Consumption",
        "Diet_Quality",
        "Sleep_Duration",
        "Stress_Level"
    ],

    "Genetic and Demographics": [
        "HbA1c",
        "Age",
        "Gender",
        "Ethnicity",
        "Family_History_of_Diabetes"
    ],

    "Minimalist Set": [
        "Fasting_Blood_Glucose",
        "HbA1c",
        "BMI",
        "Age"
    ]
}

PCA_CONFIG = {
    "n_components_range": [None, 0.95] + list(range(2, 26, 2)),
    "random_state": 42
}

TSNE_CONFIG = {
    "n_components_options": range(2, 4),
    "perplexity_options": [5, 15, 30, 50, 100],
    "random_state": 42
}

CLUSTERING_CONFIG = {
    "k_range": range(2, 5),
    "hdbscan_min_samples": range(5, 106, 10),
    "random_state": 42
}

# TSNE_CONFIG = {
#     "n_components_options": range(2, 3),
#     "perplexity_options": [15],
#     "n_clusters_options": range(5, 6),
#     "random_state": 42
# }

# Configuration for UMAP optimization
UMAP_CONFIG = {
    "n_neighbors_options": [5, 15, 30, 50],
    "min_dist_options": [0.0, 0.1, 0.25, 0.5],
    "n_components_options": range(2, 6),
    "n_clusters_options": range(2, 6),
    "random_state": 42
}

# UMAP_CONFIG = {
#     "n_neighbors_options": [15, 30],
#     "min_dist_options": [0.0, 0.1],
#     "n_components_options": range(2, 6),
#     "n_clusters_options": range(2, 5),
#     "random_state": 42
# }

#
# PCA_CONFIG = {
#     "n_components_range": [2],
#     "variance_threshold_95": 0.95,
#     "variance_threshold_99": 0.99
# }
#
# # Configuration for clustering algorithms
# CLUSTERING_CONFIG = {
#     "k_range": [2],
#     "dbscan_eps_factors": [2],
#     "random_state": 42
# }
