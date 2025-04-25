"""
Data loading and preprocessing utilities for diabetes clustering analysis.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from utils.caching import cache_result
from sklearn.model_selection import train_test_split


@cache_result()
def load_dataset(file_path):
    """Load and prepare dataset.
    
    Args:
        file_path (str): Path to the CSV dataset
        
    Returns:
        pandas.DataFrame: Loaded and preprocessed dataframe
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
    except Exception as e:
        logging.error(f"Error loading the dataset: {e}")
        raise

    # Basic dataset information
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Sample data:\n{df.head()}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")

    # Fill None in Alcohol_Consumption if column exists
    if 'Alcohol_Consumption' in df.columns:
        df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')

    # Remove any unnamed index columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df


def get_column_types(df, selected_features=None):
    """Identify categorical and numerical columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        selected_features (list, optional): List of features to include
        
    Returns:
        tuple: (categorical_columns, numerical_columns)
    """
    # Filter columns by selected_features if provided
    if selected_features is not None:
        # Ensure all selected features exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        if len(available_features) < len(selected_features):
            missing = set(selected_features) - set(available_features)
            logging.warning(f"Some selected features are not in the dataset: {missing}")

        df = df[available_features]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove any unnamed index columns from numerical columns
    if 'Unnamed: 0' in numerical_cols:
        numerical_cols.remove('Unnamed: 0')

    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    return categorical_cols, numerical_cols


@cache_result()
def preprocess_data(df, categorical_cols, numerical_cols):
    """Preprocess data using pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns
        numerical_cols (list): List of numerical columns
        
    Returns:
        tuple: (preprocessed_data_matrix, preprocessor_object)
    """
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
        ] if categorical_cols else [
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ]  # Handle the case when there are no categorical columns
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(df[numerical_cols + categorical_cols])
    logging.info(f"Processed data shape: {X_processed.shape}")

    return X_processed, preprocessor


def split_train_test(df, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets.

    Args:
        df (pandas.DataFrame): Input dataframe
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (df_train, df_test)
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test