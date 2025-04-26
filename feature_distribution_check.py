import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import our data utils and new distribution checking functions
from utils.data_utils import load_dataset, get_column_types



def check_feature_distribution(df_train, df_test, categorical_features=None):
    """
    Analyze the distribution of features across train and test sets.

    Args:
        df_train (pandas.DataFrame): Training dataframe
        df_test (pandas.DataFrame): Test dataframe
        categorical_features (list, optional): List of categorical features to specifically analyze.
                                              If None, will detect categorical features automatically.

    Returns:
        dict: Dictionary with distribution metrics for each feature
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import chi2_contingency, ks_2samp

    results = {}

    # If categorical features not specified, detect them
    if categorical_features is None:
        categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Add all remaining columns as numerical
    numerical_features = [col for col in df_train.columns if col not in categorical_features]

    # Check categorical features distribution
    print("Analyzing categorical features:")
    for feature in categorical_features:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue

        # Get value counts and percentages
        train_counts = df_train[feature].value_counts(normalize=True).sort_index()
        test_counts = df_test[feature].value_counts(normalize=True).sort_index()

        # Create a complete index with all unique values from both sets
        all_values = sorted(set(train_counts.index) | set(test_counts.index))

        # Reindex with all values, filling NaN with 0
        train_counts = train_counts.reindex(all_values, fill_value=0)
        test_counts = test_counts.reindex(all_values, fill_value=0)

        # Calculate chi-square test for independence
        try:
            # Create contingency table
            obs = pd.DataFrame({
                'train': train_counts * len(df_train),
                'test': test_counts * len(df_test)
            })

            # Run chi-square test
            chi2, p_value, _, _ = chi2_contingency(obs)

            results[feature] = {
                'type': 'categorical',
                'chi2': chi2,
                'p_value': p_value,
                'train_distribution': train_counts.to_dict(),
                'test_distribution': test_counts.to_dict(),
                'missing_in_train': df_train[feature].isna().mean(),
                'missing_in_test': df_test[feature].isna().mean(),
                'unique_values_train': len(df_train[feature].unique()),
                'unique_values_test': len(df_test[feature].unique()),
                'values_only_in_train': set(df_train[feature].unique()) - set(df_test[feature].unique()),
                'values_only_in_test': set(df_test[feature].unique()) - set(df_train[feature].unique())
            }

            print(f"  {feature}: chi2={chi2:.2f}, p-value={p_value:.4f}")

            # Print warning if p-value is low (distributions are significantly different)
            if p_value < 0.05:
                print(
                    f"    WARNING: Distributions of {feature} in train and test are significantly different (p={p_value:.4f})")

            # Print warning if there are categories only in one set
            values_only_in_train = set(df_train[feature].unique()) - set(df_test[feature].unique())
            values_only_in_test = set(df_test[feature].unique()) - set(df_train[feature].unique())

            if values_only_in_train:
                if None in values_only_in_train:
                    values_only_in_train.remove(None)
                if len(values_only_in_train) > 0:
                    print(f"    WARNING: Values in train but not in test: {values_only_in_train}")

            if values_only_in_test:
                if None in values_only_in_test:
                    values_only_in_test.remove(None)
                if len(values_only_in_test) > 0:
                    print(f"    WARNING: Values in test but not in train: {values_only_in_test}")

        except Exception as e:
            print(f"  Error analyzing {feature}: {e}")

    # Check numerical features distribution
    print("\nAnalyzing numerical features:")
    for feature in numerical_features:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue

        # Filter out NaN values for KS test
        train_values = df_train[feature].dropna()
        test_values = df_test[feature].dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            print(f"  {feature}: Insufficient non-NaN values for analysis")
            continue

        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(train_values, test_values)

        results[feature] = {
            'type': 'numerical',
            'ks_stat': ks_stat,
            'p_value': p_value,
            'train_mean': df_train[feature].mean(),
            'test_mean': df_test[feature].mean(),
            'train_std': df_train[feature].std(),
            'test_std': df_test[feature].std(),
            'train_min': df_train[feature].min(),
            'test_min': df_test[feature].min(),
            'train_max': df_train[feature].max(),
            'test_max': df_test[feature].max(),
            'missing_in_train': df_train[feature].isna().mean(),
            'missing_in_test': df_test[feature].isna().mean()
        }

        print(f"  {feature}: KS={ks_stat:.2f}, p-value={p_value:.4f}")

        # Print warning if p-value is low (distributions are significantly different)
        if p_value < 0.05:
            print(
                f"    WARNING: Distributions of {feature} in train and test are significantly different (p={p_value:.4f})")

    return results


def visualize_categorical_distributions(df_train, df_test, categorical_features, max_features=10):
    """
    Create visualizations of categorical feature distributions in train vs test sets.

    Args:
        df_train (pandas.DataFrame): Training dataframe
        df_test (pandas.DataFrame): Test dataframe
        categorical_features (list): List of categorical features to visualize
        max_features (int): Maximum number of features to visualize

    Returns:
        None: Displays the visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import pandas as pd

    # Limit to max_features
    if len(categorical_features) > max_features:
        print(f"Limiting visualization to {max_features} features")
        categorical_features = categorical_features[:max_features]

    # Calculate grid dimensions
    n_features = len(categorical_features)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Convert axes to a flattened array for easier indexing
    if n_features == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Plot each feature
    for i, feature in enumerate(categorical_features):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        # Get value counts
        train_counts = df_train[feature].value_counts(normalize=True).sort_index()
        test_counts = df_test[feature].value_counts(normalize=True).sort_index()

        # Create a DataFrame for plotting
        all_values = sorted(set(train_counts.index) | set(test_counts.index))
        df_plot = pd.DataFrame(index=all_values)
        df_plot['Train'] = df_plot.index.map(train_counts).fillna(0)
        df_plot['Test'] = df_plot.index.map(test_counts).fillna(0)

        # Plot
        df_plot.plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {feature}')
        ax.set_ylabel('Proportion')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend()

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_numerical_distributions(df_train, df_test, numerical_features, max_features=10):
    """
    Create visualizations of numerical feature distributions in train vs test sets.

    Args:
        df_train (pandas.DataFrame): Training dataframe
        df_test (pandas.DataFrame): Test dataframe
        numerical_features (list): List of numerical features to visualize
        max_features (int): Maximum number of features to visualize

    Returns:
        None: Displays the visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import numpy as np

    # Limit to max_features
    if len(numerical_features) > max_features:
        print(f"Limiting visualization to {max_features} features")
        numerical_features = numerical_features[:max_features]

    # Calculate grid dimensions
    n_features = len(numerical_features)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Convert axes to a flattened array for easier indexing
    if n_features == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Plot each feature
    for i, feature in enumerate(numerical_features):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        # Plot density
        sns.kdeplot(df_train[feature].dropna(), ax=ax, label='Train')
        sns.kdeplot(df_test[feature].dropna(), ax=ax, label='Test')

        ax.set_title(f'Distribution of {feature}')
        ax.set_ylabel('Density')
        ax.legend()

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()

# Load the dataset
file_path = "diabetes_dataset.csv"  # Replace with your actual file path
df = load_dataset(file_path)

# Get column types
categorical_cols, numerical_cols = get_column_types(df)

# Option 1: Simple train/test split using scikit-learn
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Option 2: Using your existing split_train_test function from data_utils
# from data_utils import split_train_test
# df_train, df_test = split_train_test(df)

# Check feature distributions
results = check_feature_distribution(df_train, df_test, categorical_features=categorical_cols)

# Visualize categorical feature distributions
print("\nVisualizing categorical feature distributions:")
visualize_categorical_distributions(df_train, df_test, categorical_cols)

# Visualize numerical feature distributions
print("\nVisualizing numerical feature distributions:")
visualize_numerical_distributions(df_train, df_test, numerical_cols)

# If you find issues with distribution, consider stratified sampling
# Example of stratified sampling on an important categorical feature:
if 'Diabetes_Status' in df.columns:
    print("\nRe-splitting using stratified sampling on Diabetes_Status:")
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Diabetes_Status']
    )

    # Re-check to verify improved distribution
    print("\nAfter stratification:")
    results_after = check_feature_distribution(df_train, df_test, categorical_features=categorical_cols)

    # Compare specific feature before and after stratification
    print("\nComparison of Diabetes_Status distribution before and after stratification:")
    before = pd.DataFrame({
        'train': results['Diabetes_Status']['train_distribution'],
        'test': results['Diabetes_Status']['test_distribution']
    })

    after = pd.DataFrame({
        'train': results_after['Diabetes_Status']['train_distribution'],
        'test': results_after['Diabetes_Status']['test_distribution']
    })

    print("\nBefore stratification:")
    print(before)
    print("\nAfter stratification:")
    print(after)

# Handle rare categories in enumerated features
# Option 1: Group rare categories
for cat_feature in categorical_cols:
    # Calculate frequency
    value_counts = df[cat_feature].value_counts(normalize=True)

    # Find rare categories (e.g., less than 1% of data)
    rare_categories = value_counts[value_counts < 0.01].index.tolist()

    if rare_categories:
        print(f"\nFeature '{cat_feature}' has {len(rare_categories)} rare categories")
        print(f"Grouping rare categories in '{cat_feature}'")

        # Create a copy of the feature
        df[f'{cat_feature}_grouped'] = df[cat_feature].copy()

        # Replace rare categories with 'Other'
        df.loc[df[f'{cat_feature}_grouped'].isin(rare_categories), f'{cat_feature}_grouped'] = 'Other'

        # Split again with the new grouped feature
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        # Check distribution of the new grouped feature
        print(f"Distribution after grouping rare categories in '{cat_feature}_grouped':")
        check_feature_distribution(df_train, df_test, categorical_features=[f'{cat_feature}_grouped'])

# Option 2: Cross-validation with stratification on multiple features
# For complex cases with multiple important categorical features
from sklearn.model_selection import StratifiedKFold
import itertools


def stratify_on_multiple_features(df, categorical_features, n_splits=5):
    """Create a stratification column combining multiple categorical features"""
    # Create a combined feature for stratification
    df_temp = df.copy()

    # Select a subset of categorical features if there are too many
    if len(categorical_features) > 3:
        print("Using only the first 3 categorical features for stratification to avoid too many combinations")
        strat_features = categorical_features[:3]
    else:
        strat_features = categorical_features

    # Create stratification column
    df_temp['strat_col'] = df_temp[strat_features].astype(str).agg('_'.join, axis=1)

    # Initialize KFold with stratification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate indices
    train_indices = []
    test_indices = []

    for train_idx, test_idx in skf.split(df_temp, df_temp['strat_col']):
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    return train_indices, test_indices


# Example usage of multi-feature stratification
if len(categorical_cols) >= 2:
    print("\nPerforming cross-validation with stratification on multiple features:")
    features_to_stratify = categorical_cols[:2]  # Use first two categorical features
    print(f"Stratifying on: {features_to_stratify}")

    train_indices, test_indices = stratify_on_multiple_features(df, features_to_stratify, n_splits=5)

    # Check the first fold
    print("\nChecking distribution in first fold:")
    fold_train = df.iloc[train_indices[0]]
    fold_test = df.iloc[test_indices[0]]

    check_feature_distribution(fold_train, fold_test, categorical_features=categorical_cols)

