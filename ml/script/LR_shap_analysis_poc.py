import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

def generate_shap_beeswarm_for_pipeline(model_path, symbol_mapping, results_file_path, stratum, top_n=30):
    # Load pipeline and data
    model_data = joblib.load(model_path)
    pipeline = model_data['model']
    X_test = model_data['X_test']
    features = model_data['features']

    # Convert to DataFrame if needed
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=features)

    # Extract the preprocessing steps and final estimator
    preprocessor = pipeline[:-1]  # All steps except the last
    lr_model = pipeline[-1]  # The Logistic Regression model

    # Transform the test data using preprocessing steps
    X_test_transformed = preprocessor[0].transform(X_test)

    # Get feature names after preprocessing
    try:
        # For column transformers with feature names
        transformed_features = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if transformer doesn't support feature names
        transformed_features = features  # Use original names

    # Create explainer for the logistic regression
    explainer = shap.LinearExplainer(
        lr_model,
        shap.sample(X_test_transformed, 100)  # Use sample for background
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_transformed)
    gene_symbols = symbol_mapping.loc[transformed_features]['gene_symbol'].to_list()
    # Create beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=gene_symbols,
        plot_type="dot",
        max_display=top_n,
        show=False
    )

    # Customize plot
    plt.title(f"Top {top_n} Gene Features by SHAP Value Impact\n(Logistic Regression Pipeline)",
              fontsize=14, pad=20)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.ylabel("Gene Features", fontsize=12)

    # Adjust color bar
    cb = plt.gcf().axes[-1]
    cb.set_position([0.92, 0.2, 0.02, 0.2])
    cb.set_ylabel("Normalized Expression", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{results_file_path}/shap_beeswarm_plot_{stratum}.png")

# Example usage
if __name__ == "__main__":
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']
    lr_model_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/LR/deg_classification"
    results_path = f"{lr_model_path}/results"

    # genders = ['Male', 'Female']
    # age_groups = ['30-50', '50-70', '70-80', '>80']
    # for gender in genders:
    #     for age_group in age_groups:
    lr_model_file = f"{lr_model_path}/model_LR_stratified_Male_30-50_LR_useSMOTE_False.joblib"
    generate_shap_beeswarm_for_pipeline(lr_model_file, symbol_ensembl_mapping, results_path, "Male_30-50_noSmote", top_n=30)