from pathlib import Path
from typing import Final

import scipy
import shap
import joblib
import numpy as np
import pandas as pd
import os
import anndata as ad

# Example configuration - adjust to your actual data
model_dir = "path/to/your/stratified_models"
results_dir = "shap_results_stratified"
os.makedirs(results_dir, exist_ok=True)

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/LR2/"
CLASSIFICATION_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/"

def get_stratum_background(ppmi_subset, min_samples=3):
    X = ppmi_subset.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    n_total = X.shape[0]

    # Handle extremely small strata
    if n_total <= min_samples:
        print(f"Warning: Only {n_total} samples in stratum. Using all as background.")
        return X  # Return all samples as background

    # For moderately small strata (4-20 samples)
    if n_total <= 20:
        # Use 30% of data for background, minimum 2 samples
        n_background = max(2, int(n_total * 0.3))
        background_idx = np.random.choice(n_total, n_background, replace=False)
        return X[background_idx]

    # For larger small strata (21-50 samples)
    n_background = min(10, n_total // 3)
    background_idx = np.random.choice(n_total, n_background, replace=False)
    return X[background_idx]


def analyze_stratum_model(model_path, ppmi_subset, X_explain):
    """
    SHAP analysis with proper feature name handling
    - Works with your filtered features
    - Maintains scikit-learn feature name validation
    - Preserves all existing functionality
    """
    # 1. Load model and get pipeline
    model_dict = joblib.load(model_path)
    pipeline = model_dict['model']

    # 2. Get background data (using your function)
    background_data = get_stratum_background(ppmi_subset)

    # 3. Convert to pandas DataFrame with feature names
    feature_names = model_dict['features']  # Using the features saved with your model

    def ensure_dataframe(data, features):
        """Convert array to DataFrame with proper feature names"""
        if isinstance(data, pd.DataFrame):
            return data[features]  # Ensure correct column order
        return pd.DataFrame(np.asarray(data), columns=features)

    background_df = ensure_dataframe(background_data, feature_names)
    explain_df = ensure_dataframe(X_explain, feature_names)

    # 4. Create prediction function that maintains DataFrame structure
    def pipeline_predict(X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=feature_names)
        return pipeline.predict_proba(X)

    # 5. Initialize explainer
    explainer = shap.LinearExplainer(
        pipeline_predict,
        background_df,
        link='logit'
    )

    # 6. Calculate SHAP values
    explanation_sample = explain_df.iloc[:100] if len(explain_df) > 100 else explain_df
    shap_values = explainer.shap_values(explanation_sample)

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_names': feature_names,
        'explained_samples': explanation_sample.values  # Return numpy array for plotting
    }


def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL']#['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['50-70']#['30-50', '50-70', '70-80', '>80']
    genders = ['Male']#['Male', 'Female']

    for gender in genders:
        for age_group in age_groups:
            for visit in visits:
                stratified_model = Path(PATH) / f"model_{gender}_{age_group}_{visit}.joblib"
                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])) &
                        (ppmi_ad.obs['Visit'] == visit))

                common_genes = pd.read_csv(CLASSIFICATION_PATH + f"common_genes_{gender}_{age_group}_{visit}.csv", index_col=0)
                ppmi_ad = ppmi_ad[:, ppmi_ad.var.index.isin(common_genes.index)]
                ppmi_ad_subset = ppmi_ad[mask]
                X_explain = ppmi_ad_subset.X[:100]
                # shap_results = run_stratum_shap_analysis(ppmi_ad_subset, stratified_model)
                results = analyze_stratum_model(
                    model_path=stratified_model,
                    ppmi_subset=ppmi_ad_subset,
                    X_explain=X_explain
                )
                # Check prediction distribution
                # print("Predicted probabilities (first 10):", pipeline.predict_proba(explain_df)[:10])

                # Check SHAP value shape
                print("SHAP values shape:", np.array(results['shap_values']).shape)

                # Check if SHAP values are near-zero
                print("Mean absolute SHAP value:", np.mean(np.abs(results['shap_values'])))

                # Check explained sample variance
                print("Explained sample variance (mean std per gene):",
                      np.mean(np.std(results['explained_samples'], axis=0)))

                shap.summary_plot(
                    results['shap_values'],
                    results['explained_samples'],
                    feature_names=results['feature_names'],
                    plot_type="dot",
                )
                # sample_stratum = list(shap_results.keys())[0]
                # print(f"Analysis complete for {sample_stratum}")
                # print(f"SHAP values shape: {shap_results[sample_stratum]['shap_values'].shape}")
                # print(f"Features: {shap_results[sample_stratum]['feature_names'][:5]}...")

if __name__ == '__main__':
    main()