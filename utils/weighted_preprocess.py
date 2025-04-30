import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class WeightedScaler(BaseEstimator, TransformerMixin):
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


def preprocess_data(df, categorical_cols, numerical_cols,
                    diabetes_markers=None,
                    weight=None):
    other_numerical = [col for col in numerical_cols if col not in diabetes_markers]
    transformers = []
    diabetes_markers_present = [col for col in diabetes_markers if col in df.columns]
    if diabetes_markers_present is not None and weight is not None:
        transformers.append(
            ('diabetes', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('weighted_scaler', WeightedScaler(weight=weight))
            ]), diabetes_markers_present)
        )
        logging.info(f"Applying {weight}x weight to diabetes markers: {diabetes_markers_present}")

    if other_numerical:
        transformers.append(
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), other_numerical)
        )

    if categorical_cols:
        transformers.append(
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    X_processed = preprocessor.fit_transform(df[numerical_cols + categorical_cols])
    logging.info(f"processed data shape: {X_processed.shape}")

    return X_processed, preprocessor


def create_diabetes_features(df):
    df_enhanced = df.copy()
    min_hba1c = 6.5
    min_fasting_blood_glucose = 126

    if 'HbA1c' in df.columns:
        df_enhanced['is_diabetic_HbA1c'] = (df['HbA1c'] >min_hba1c).astype(float)

    if 'Fasting_Blood_Glucose' in df.columns:
        df_enhanced['is_diabetic_FBG'] = (df['Fasting_Blood_Glucose'] > min_fasting_blood_glucose).astype(float)

    if 'HbA1c' in df.columns and 'Fasting_Blood_Glucose' in df.columns:
        df_enhanced['diabetes_score'] = df_enhanced.get('is_diabetic_HbA1c', 0) + df_enhanced.get('is_diabetic_FBG', 0)
        logging.info("Added diabetes diagnostic features based on clinical thresholds")

    return df_enhanced

