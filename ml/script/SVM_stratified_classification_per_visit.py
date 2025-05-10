import warnings
from typing import Final

import anndata as ad
from feature_importance_calcs import calculate_stratified_importances
from common_ml import test_classifier, run_10x_fold_validation, plot_results
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report)
from sklearn.exceptions import FitFailedWarning

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/SVM/"

def train_svm(anndata_obj_subset, stratum, test_size=0.2, random_state=42, min_samples=5,):
    print("Training SVM Model")
    X = pd.DataFrame(anndata_obj_subset.layers['counts_log2'], columns=anndata_obj_subset.var_names)
    y = (anndata_obj_subset.obs['Diagnosis'] == 'PD').astype(int)

    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")

    if min(class_counts) < 2:
        print(f"Warning: Insufficient samples in stratum {stratum} - skipping")
        return None

    n_test = max(1, int(len(X) * test_size))
    if n_test < 2:
        print("Warning: Too few samples in stratum - skipping")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    train_class_counts = y_train.value_counts()
    use_smote = min(train_class_counts) >= min_samples

    steps = [
        ('scaler', StandardScaler())
    ]
    if use_smote:
        steps.append(('smote', SMOTE(
            random_state=random_state,
            k_neighbors=min(3, min(train_class_counts) - 1))))
    else:
        print(f"Warning: Not using SMOTE for {stratum} - smallest class has {min(train_class_counts)} samples")

    steps.append(('svm', SVC(probability=True, random_state=42)))

    svm_pipeline = Pipeline(steps)

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto', 0.1],
        'svm__kernel': ['linear', 'rbf'],
        'smote__k_neighbors': [3, 5]
    }

    if use_smote:
        param_grid['smote__k_neighbors'] = [
            min(3, min(train_class_counts) - 1),
            min(5, min(train_class_counts) - 1),
        ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)

            grid_search = GridSearchCV(estimator=svm_pipeline, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1,
                                       verbose=1)
            grid_search.fit(X_train, y_train)
            best_svm = grid_search.best_estimator_
            best_svm.fit(X_train, y_train)
            model_path = os.path.join(PATH + f"model_{stratum}.joblib")
            joblib.dump({
                'model': best_svm,
                'X_test': X_test,
                'y_test': y_test,
                'features': X.columns.tolist()
            }, model_path)
    except Exception as e:
        print(f"Failed to train model for {stratum}: {str(e)}")
        return None

    return best_svm, X_test, y_test, svm_pipeline, X, y

def test_classifier(classifier, X_test, y_test, result_file):
    print("Testing classifier...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]
    classification_report_str = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print("Classification Report:")
    print(classification_report_str)
    print(f"\nROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

    # Append results to file
    with open(result_file, 'a') as f:
        f.write("Testing classifier results:\n")
        f.write("Classification Report:\n")
        f.write(classification_report_str + "\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n")
        f.write(f"PR-AUC: {pr_auc:.3f}\n")
        f.write("\n")
    return y_proba, y_pred

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Female']

    for gender in genders:
        for age_group in age_groups:
            result_file = PATH + f"results_{gender}_{age_group}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Results for Age Group: {age_group}, Gender: {gender}\n\n")

            for visit in visits:
                print(f"Visit: {visit}, Age Group: {age_group}, Gender: {gender}")
                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])) &
                        (ppmi_ad.obs['Visit'] == visit))
                ppmi_ad_subset = ppmi_ad[mask]
                common_genes = calculate_stratified_importances(ppmi_ad_subset, 'PD')
                ppmi_ad_subset = ppmi_ad_subset[:, ppmi_ad_subset.var.index.isin(common_genes)]
                result = train_svm(ppmi_ad_subset, f"{gender}_{age_group}_{visit}")
                if result is None:
                    print(f"Failed to train SVM for {visit} - skipping")
                    continue
                best_svm, X_test, y_test, svm_pipeline, X, y = result

                with open(result_file, 'a') as f:
                    f.write(f"Visit: {visit}\n")
                y_proba, y_pred = test_classifier(best_svm, X_test, y_test, result_file)
                run_10x_fold_validation(svm_pipeline, X, y, result_file)
                plot = plot_results(y_test, y_proba, y_pred)
                plot.savefig(PATH + f"results_{gender}_{age_group}_{visit}.png")
                plot.clf()
                plot.close()


if __name__ == '__main__':
    main()