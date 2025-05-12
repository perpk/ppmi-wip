from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/LR2/"

def extract_logreg_feature_importance(model_path):
    """
    Extracts and sorts feature importance from logistic regression model
    Returns DataFrame with features and their coefficients
    """
    # Load your saved model
    model_dict = joblib.load(model_path)
    pipeline = model_dict['model']

    # Get the logistic regression estimator from the pipeline
    logreg = pipeline.named_steps['lr']  # Adjust key if different

    # Get feature names (use your original features or pipeline-transformed names)
    try:
        feature_names = model_dict['features']  # From your saved dictionary
    except:
        feature_names = pipeline[:-1].get_feature_names_out()

    # Create DataFrame of coefficients
    if logreg.coef_.shape[0] == 1:  # Binary classification
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': logreg.coef_[0],
            'abs_coefficient': np.abs(logreg.coef_[0])
        })
    else:  # Multiclass classification
        importance_df = pd.DataFrame(
            logreg.coef_,
            columns=feature_names,
            index=[f"Class_{i}" for i in range(logreg.coef_.shape[0])]
        ).T.reset_index()
        importance_df = importance_df.rename(columns={'index': 'feature'})

    # Sort by absolute coefficient value (most important features first)
    if 'abs_coefficient' in importance_df.columns:
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

    return importance_df


def analyze_feature_importance(importance_df, path, top_n=20):
    # Display top features
    print(f"\nTop {top_n} most important features:")
    print(importance_df.head(top_n))

    # Visualize (simple bar plot)
    if 'abs_coefficient' in importance_df.columns:
        # Binary classification visualization
        importance_df.head(top_n).plot.bar(
            x='feature',
            y='abs_coefficient',
            title='Feature Importance (Absolute Coefficients)',
            figsize=(12, 6)
        )
    else:
        # Multiclass visualization
        class_cols = [c for c in importance_df.columns if c.startswith('Class_')]
        importance_df.set_index('feature').head(top_n)[class_cols].plot.bar(
            title='Feature Importance by Class',
            figsize=(12, 6),
            width=0.8
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path)

    return importance_df

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Female']

    for gender in genders:
        for age_group in age_groups:
            for visit in visits:
                stratified_model = Path(PATH) / f"model_{gender}_{age_group}_{visit}.joblib"
                if not stratified_model.exists():
                    continue
                importance_df = extract_logreg_feature_importance(stratified_model)
                importance_df.to_csv(f"{PATH}/feature_importance_data_{gender}_{age_group}_{visit}.csv")
                analyze_feature_importance(importance_df, f"{PATH}/feature_importance_plot_{gender}_{age_group}_{visit}.png", top_n=30)
                print("debug")

if __name__ == '__main__':
    main()