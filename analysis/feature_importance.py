import os
import numpy as np
import pandas as pd
import logging
from scipy import stats
from visualization.feature_viz import plot_feature_importance_f_values, plot_feature_importance_chi2, plot_top_features_pairplot

def diabetes_feature_importance(diabetes_df, cluster_labels, num_cols, cat_cols, diabetes_out="output"):
    logging.info("feature importance for clustering")
    os.makedirs(diabetes_out, exist_ok=True)
    df = diabetes_df.copy()
    df['cluster'] = cluster_labels
    f_vals, p_vals = {}, {}
    for col in num_cols:
        groups = [df[df['cluster'] == c][col].values for c in np.unique(cluster_labels)]
        f, p = stats.f_oneway(*groups)
        f_vals[col] = f
        p_vals[col] = p
    sorted_num = sorted(f_vals.items(), key=lambda x: x[1], reverse=True)
    num_features = [x[0] for x in sorted_num]
    num_fvals = [x[1] for x in sorted_num]
    plot_feature_importance_f_values(num_features, num_fvals, p_vals, diabetes_out)
    chi2_vals, chi2_pvals = {}, {}
    for col in cat_cols:
        tab = pd.crosstab(df['cluster'], df[col])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        chi2_vals[col] = chi2
        chi2_pvals[col] = p
    if cat_cols:
        sorted_cat = sorted(chi2_vals.items(), key=lambda x: x[1], reverse=True)
        cat_features = [x[0] for x in sorted_cat]
        cat_chi2 = [x[1] for x in sorted_cat]
        plot_feature_importance_chi2(cat_features, cat_chi2, chi2_pvals, diabetes_out)
    imp_df = pd.DataFrame({
        'Feature': num_features,
        'F_Value': num_fvals,
        'p_value': [p_vals[f] for f in num_features],
        'Significance': ['***' if p_vals[f] < 0.001 else '**' if p_vals[f] < 0.01 else '*' if p_vals[f] < 0.05 else '' for f in num_features]
    })
    if cat_cols:
        cat_imp_df = pd.DataFrame({
            'Feature': cat_features,
            'Chi2_Value': cat_chi2,
            'p_value': [chi2_pvals[f] for f in cat_features],
            'Significance': ['***' if chi2_pvals[f] < 0.001 else '**' if chi2_pvals[f] < 0.01 else '*' if chi2_pvals[f] < 0.05 else '' for f in cat_features]
        })
        imp_df = pd.concat([imp_df, cat_imp_df], ignore_index=True)
    imp_df.to_csv(os.path.join(diabetes_out, 'feature_importance.csv'), index=False)
    top_num = num_features[:min(3, len(num_features))]
    if len(top_num) > 1:
        plot_top_features_pairplot(df, top_num, diabetes_out)
    return {
        'numerical_importance': {'f_values': f_vals, 'p_values': p_vals, 'sorted_features': sorted_num},
        'categorical_importance': {'chi2_values': chi2_vals, 'chi2_p_values': chi2_pvals, 'sorted_features': sorted_cat if cat_cols else []},
        'importance_df': imp_df,
        'top_numerical_features': top_num
    }