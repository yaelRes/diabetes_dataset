import logging
import os
from datetime import datetime
from main import run_analysis
from visualization.comparison_viz import create_comparative_visualizations

def diabetes_comparative_analysis(data_path="diabetes_dataset.csv", diabetes_feature_sets=None, test_size=0.2, sample_ratio=1.0):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    diabetes_dir = f"output_comparative_{ts}"
    os.makedirs(diabetes_dir, exist_ok=True)
    logging.info(f"starting comparative analysis with {len(diabetes_feature_sets)} feature sets")
    if sample_ratio < 1.0:
        logging.info(f"Using {sample_ratio * 100:.1f}% of the dataset for faster processing")
    results = {}
    for name, feats in diabetes_feature_sets.items():
        logging.info(f"analyzing feature set: {name}")
        feat_dir = os.path.join(diabetes_dir, name.replace(" ", "_"))
        os.makedirs(feat_dir, exist_ok=True)
        res = run_analysis(
            data_path=data_path,
            output_dir=feat_dir,
            selected_features=feats,
            test_size=test_size,
            sample_ratio=sample_ratio
        )
        results[name] = res
    try:
        dfs = {}
        for name, res in results.items():
            if 'eval_result' in res and 'metrics_df' in res['eval_result']:
                dfs[name] = res['eval_result']['metrics_df']
        if dfs:
            create_comparative_visualizations(dfs, diabetes_dir)
    except:
        pass
    return results
