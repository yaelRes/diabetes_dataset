"""
Enhanced data preprocessing utilities for diabetes clustering analysis with weighted markers.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to apply weights to specific features
class WeightedScaler(BaseEstimator, TransformerMixin):
    """Scaler that applies additional weight to specific features."""
    
    def __init__(self, weight=3.0):
        self.weight = weight
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
        
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled * self.weight
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def preprocess_data_weighted(df, categorical_cols, numerical_cols, 
                             diabetes_markers=['HbA1c', 'Fasting_Blood_Glucose'], 
                             weight=3.0):
    """Preprocess data using pipeline with weighted diabetes markers.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns
        numerical_cols (list): List of numerical columns
        diabetes_markers (list): List of diabetes diagnostic markers to emphasize
        weight (float): Weight to apply to diabetes markers (default: 3.0)
        
    Returns:
        tuple: (preprocessed_data_matrix, preprocessor_object)
    """
    # Separate diabetes markers from other numerical columns
    other_numerical = [col for col in numerical_cols if col not in diabetes_markers]
    
    # Create transformers list
    transformers = []
    
    # Add weighted transformer for diabetes markers if they exist in the dataset
    diabetes_markers_present = [col for col in diabetes_markers if col in df.columns]
    if diabetes_markers_present:
        transformers.append(
            ('diabetes', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('weighted_scaler', WeightedScaler(weight=weight))
            ]), diabetes_markers_present)
        )
        logging.info(f"Applying {weight}x weight to diabetes markers: {diabetes_markers_present}")
    
    # Add standard transformer for other numerical columns
    if other_numerical:
        transformers.append(
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), other_numerical)
        )
    
    # Add transformer for categorical columns if they exist
    if categorical_cols:
        transformers.append(
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        )
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(df[numerical_cols + categorical_cols])
    logging.info(f"Processed data shape: {X_processed.shape}")

    return X_processed, preprocessor


def create_diabetes_features(df):
    """Create additional features based on clinical diabetes criteria.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Enhanced dataframe with new features
    """
    df_enhanced = df.copy()
    
    # Add clinical threshold features if columns exist
    if 'HbA1c' in df.columns:
        df_enhanced['is_diabetic_HbA1c'] = (df['HbA1c'] > 6.5).astype(float)
    
    if 'Fasting_Blood_Glucose' in df.columns:
        df_enhanced['is_diabetic_FBG'] = (df['Fasting_Blood_Glucose'] > 126).astype(float)
    
    # Create combined score if both features exist
    if 'HbA1c' in df.columns and 'Fasting_Blood_Glucose' in df.columns:
        df_enhanced['diabetes_score'] = df_enhanced.get('is_diabetic_HbA1c', 0) + df_enhanced.get('is_diabetic_FBG', 0)
        logging.info("Added diabetes diagnostic features based on clinical thresholds")
    
    return df_enhanced


# Example usage in main.py:
"""
# Create enhanced features
df_enhanced = create_diabetes_features(df)

# Get column types with the new features
categorical_cols, numerical_cols = get_column_types(df_enhanced, selected_features)

# Use weighted preprocessing
X_processed, preprocessor = preprocess_data_weighted(
    df_enhanced, 
    categorical_cols, 
    numerical_cols,
    diabetes_markers=['HbA1c', 'Fasting_Blood_Glucose', 'is_diabetic_HbA1c', 
                      'is_diabetic_FBG', 'diabetes_score'],
    weight=3.0
)
"""
