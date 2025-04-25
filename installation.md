# Installation Guide

This guide will help you implement the train/test split functionality in your diabetes clustering analysis project.

## Prerequisites

Make sure you have all required packages installed:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn umap-learn
```

## Implementation Steps

Follow these steps to update your codebase with train/test split functionality:

### 1. Add Train/Test Split Function to data_utils.py

Add the following function to your `data_utils.py` file:

```python
from sklearn.model_selection import train_test_split

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
```

### 2. Create Test Evaluation Module

Create a new file called `test_evaluation.py` in your project directory with the content provided in the "Test Set Evaluation Function" code.

### 3. Create Comparison Visualization Module

Create a new file called `comparison_viz.py` in your `visualization` directory with the content provided in the "Comparison Visualization Module" code.

### 4. Update Main Function

Replace your `main` function in `main.py` with the updated version provided in the "Modified Main Function with Train/Test Split" code.

### 5. Update Comparative Analysis Function

Replace your `run_comparative_analysis` function in `comparative.py` with the updated version provided in the "Updated Comparative Analysis Function" code.

### 6. Update Main Execution Block

Replace the `if __name__ == "__main__":` block in your `main.py` file with the updated version provided in the "Updated Main Execution Block" code.

## Verifying Your Implementation

After implementing these changes, you can run your analysis with:

```bash
python main.py --data_path your_dataset.csv --test_size 0.2
```

You should see:
1. The analysis running with train/test split
2. New output directories for train and test results
3. Additional visualizations comparing train vs test performance
4. A summary in the console showing the stability of different feature sets

## Troubleshooting

If you encounter issues:

1. Check that all required imports are present at the top of each file
2. Verify that directory paths are created properly (e.g., `train_dir` and `test_dir`)
3. Ensure that all functions reference the correct parameter names

## Additional Resources

For a complete understanding of the train/test split implementation, refer to:

- The README.md file for an overview of the functionality
- The scikit-learn documentation on [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- The sample code snippets provided for each component
