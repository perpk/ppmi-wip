import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad


def create_explainer(classifier, model, X_test_transformed, X_train_transformed):
    background = shap.sample(X_train_transformed, 100)  # Shared background

    if classifier == "LR":
        return shap.LinearExplainer(
            model,
            background,
            feature_perturbation="interventional"
        )
    elif classifier == "SVM":
        return shap.KernelExplainer(
            model.decision_function,
            background,
        )
    elif classifier == "RF":
        return shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
    elif classifier == "XGBOOST":
        return shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="probability"
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")


def generate_shap_beeswarm_for_pipeline(model_path, symbol_mapping, results_file_path, stratum, classifier, top_n=30):
    # Load pipeline and data
    model_data = joblib.load(model_path)
    pipeline = model_data['model']
    X_test = model_data['X_test']
    X_train = model_data['X_train']
    features = model_data['features']
    generate_shap_beeswarm(pipeline, X_test, X_train, features, symbol_mapping, results_file_path, stratum, classifier, top_n)

def generate_shap_beeswarm(pipeline, X_test, X_train, features, symbol_mapping, results_file_path, stratum, classifier, top_n=30):

    # Convert to DataFrame if needed
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=features)

    # Extract the preprocessing steps and final estimator
    preprocessor = pipeline[:-1]  # All steps except the last
    model = pipeline[-1]  # The Logistic Regression model

    # Transform the test data using preprocessing steps
    X_test_transformed = preprocessor[0].transform(X_test)
    X_train_transformed = preprocessor[0].transform(X_train)

    # Get feature names after preprocessing
    try:
        # For column transformers with feature names
        transformed_features = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if transformer doesn't support feature names
        transformed_features = features  # Use original names

    # Create explainer for the logistic regression
    explainer = create_explainer(classifier, model, X_test_transformed, X_train_transformed)

    # Calculate SHAP values
    if classifier == "LR":
        shap_values = explainer.shap_values(X_test_transformed)
    elif classifier == "SVM":
        shap_values = explainer.shap_values(X_test_transformed, nsamples=100) # For class 1
    elif classifier == "RF":
        shap_values = explainer(X_test_transformed).values[:, :, 1]
    elif classifier == "XGBOOST":
        shap_values = explainer(X_test_transformed)

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
    plt.title(f"Top {top_n} Gene Features by SHAP Value Impact\n({classifier} Pipeline)",
              fontsize=14, pad=20)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.ylabel("Gene Features", fontsize=12)

    # Adjust color bar
    cb = plt.gcf().axes[-1]
    cb.set_position([0.92, 0.2, 0.02, 0.2])
    cb.set_ylabel("Normalized Expression", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{results_file_path}/shap_beeswarm_plot_{stratum}.png")


def main():
    classifiers = ["SVM"] #["LR", "SVM", "RF", "XGBOOST"]
    genders = ['Male']#['Male', 'Female']
    age_groups = ['50-70']#['30-50', '50-70', '70-80']
    withSmote = False
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']

    for gender in genders:
        for age_group in age_groups:
            for classifier in classifiers:
                print(f"Processing SHAP Analysis - {classifier} for {gender} {age_group}")
                results_path = f"/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/{classifier}/deg_classification/shap"
                joblib_path = f"/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/{classifier}/deg_classification/no_smote"

                model_file = f"{joblib_path}/model_{classifier}_stratified_{gender}_{age_group}_{classifier}_useSMOTE_{withSmote}.joblib"
                generate_shap_beeswarm_for_pipeline(model_file, symbol_ensembl_mapping, results_path,
                                                    f"{gender}_{age_group}_useSMOTE_{withSmote}", classifier)


if __name__ == '__main__':
    main()
