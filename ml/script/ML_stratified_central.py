from typing import Final
from common_ml import test_classifier, get_dynamic_stratified_kfold, run_10x_fold_validation, plot_results
from ML_training import train_classifier
import anndata as ad

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/ml/classification/"
DEG_SOURCE_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/"

def main():
    genders = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    classifiers = ["XGBOOST"]#["LR", "SVM", "RF", "XGBOOST"]
    use_smote = False
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    for gender in genders:
        for age_group in age_groups:
            for classifier in classifiers:
                print(f"Running {gender} {age_group} {classifier}")

                stratum = f"stratified_{gender}_{age_group}_{classifier}_useSMOTE_{use_smote}"
                subpath = f"{PATH}{classifier}/deg_classification/"
                result_file = f"{subpath}results_{stratum}.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Results for Age Group: {age_group}, Gender: {gender}\n\n")

                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])))

                ppmi_ad_subset = ppmi_ad[mask]

                result = train_classifier(ppmi_ad_subset, stratum, classifier, subpath, use_smote)
                if result is None:
                    print(f"Failed to train {classifier} for {age_group} - skipping")
                    continue
                best_estimator, X_test, y_test, pipeline, X, y = result

                with open(result_file, 'a') as f:
                    f.write(f"Age Group: {age_group}\n")

                y_proba, y_pred = test_classifier(best_estimator, X_test, y_test, result_file)
                run_10x_fold_validation(pipeline, X, y, subpath, stratum, f"{gender}, {age_group}, {classifier} - SMOTE={use_smote}", groups=ppmi_ad_subset.obs['Patient'])
                plot = plot_results(y_test, y_proba, y_pred, f"{gender}, {age_group}, {classifier} - SMOTE={use_smote}")
                plot.savefig(f"{subpath}results_{stratum}.png")
                plot.clf()
                plot.close()

if __name__ == '__main__':
    main()