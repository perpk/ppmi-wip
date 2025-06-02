from typing import Final
from common_ml import test_classifier, get_dynamic_stratified_kfold, run_10x_fold_validation, plot_results
from ML_training import train_classifier
import anndata as ad
import datetime
import pandas as pd
from shap_analysis_stratified import generate_shap_beeswarm

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/"
DEG_SOURCE_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/"

def main():
    genders = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80"]
    classifiers = ["XGBOOST"]#["LR", "SVM", "RF", "XGBOOST"]
    use_smote = False
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']

    for gender in genders:
        for age_group in age_groups:
            for classifier in classifiers:
                print(f"Running {gender} {age_group} {classifier}")

                stratum = f"stratified_{gender}_{age_group}_{classifier}_useSMOTE_{use_smote}"
                subpath = f"{PATH}{classifier}/deg_classification/no_smote/"
                result_file = f"{subpath}results_{stratum}.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Results for Age Group: {age_group}, Gender: {gender}\n\n")

                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])))

                ppmi_ad_subset = ppmi_ad[mask]
                degs = pd.read_csv(DEG_SOURCE_PATH + f"DEGs_stratified_consoVisits_{gender}_{age_group}.csv",
                                   index_col=0)
                sign_degs = degs[(degs['log2FoldChange'].abs() > 0.5) & (degs['padj'] < 0.05)]
                ppmi_ad_subset = ppmi_ad_subset[:, ppmi_ad_subset.var.index.isin(sign_degs.index.tolist())]
                print(f"Training start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                result = train_classifier(ppmi_ad_subset, stratum, classifier, subpath, use_smote)
                if result is None:
                    print(f"Failed to train {classifier} for {age_group} - skipping")
                    continue
                best_estimator, X_test, y_test, pipeline, X, y, X_train = result

                with open(result_file, 'a') as f:
                    f.write(f"Age Group: {age_group}\n")

                print(f"Training end at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                y_proba, y_pred = test_classifier(best_estimator, X_test, y_test, result_file)

                print(f"10xValidation start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                run_10x_fold_validation(pipeline, X, y, subpath, stratum, f"{gender}, {age_group}, {classifier} - SMOTE={use_smote}", groups=ppmi_ad_subset.obs['Patient'])
                print(f"10xValidation end at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                plot = plot_results(y_test, y_proba, y_pred, f"{gender}, {age_group}, {classifier} - SMOTE={use_smote}")
                plot.savefig(f"{subpath}results_{stratum}.png")
                plot.clf()
                plot.close()

                if classifier == "XGBOOST":
                    results_path = f"/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/{classifier}/deg_classification/shap"
                    generate_shap_beeswarm(pipeline, X_test, X_train, X.columns.tolist(), symbol_ensembl_mapping, results_path, f"{gender}_{age_group}_useSMOTE_{use_smote}", classifier)

if __name__ == '__main__':
    main()