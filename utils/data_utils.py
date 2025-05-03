import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"dataset loaded successfully from {file_path}")
    except Exception as e:
        logging.error(f"error loading the dataset: {e}")
        raise

    if 'Alcohol_Consumption' in df.columns:
        df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Sample data:\n{df.head()}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df.drop('Family_History_of_Diabetes',axis=1)
    df.drop('Previous_Gestational_Diabetes',axis=1)

    return df


def get_column_types(df, selected_features=None):
    if selected_features is not None:
        selected_column_features = [f for f in selected_features if f in df.columns]
        if len(selected_column_features) < len(selected_features):
            missing = set(selected_features) - set(selected_column_features)
            logging.warning(f"this selected features are not in the dataset: {missing}")

        df = df[selected_column_features]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Unnamed: 0' in numerical_cols:
        numerical_cols.remove('Unnamed: 0')

    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    return categorical_cols, numerical_cols


def preprocess_data(df, categorical_cols, numerical_cols):
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
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test
