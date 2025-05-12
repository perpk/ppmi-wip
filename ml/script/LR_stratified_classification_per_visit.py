import warnings

import joblib
import pandas as pd
import anndata as ad
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Final
import os
from sklearn.exceptions import FitFailedWarning


PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/LR2/"
CLASSIFICATION_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/"

from common_ml import test_classifier, get_dynamic_stratified_kfold, run_10x_fold_validation, plot_results

def train_logistic_regression(anndata_obj_subset, stratum, test_size=0.2, random_state=42, min_samples=5):
    print("Training logistic regression model...")
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

    steps.append(('lr', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    )))

    lr_pipeline = Pipeline(steps)

    param_grid = {
        'lr__C': np.logspace(-3, 3, 7),
        'lr__penalty': ['l1', 'l2'],
        'lr__solver': ['liblinear']
    }

    if use_smote:
        param_grid['smote__k_neighbors'] = [
            min(3, min(train_class_counts) - 1),
            min(5, min(train_class_counts) - 1),
        ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            grid_search = GridSearchCV(
                lr_pipeline,
                param_grid,
                cv=get_dynamic_stratified_kfold(y_train),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_lr = grid_search.best_estimator_
            best_lr.fit(X_train, y_train)

            model_path = os.path.join(PATH + f"model_{stratum}.joblib")
            joblib.dump({
                'model': best_lr,
                'X_test': X_test,
                'y_test': y_test,
                'features': X.columns.tolist()
            }, model_path)
    except Exception as e:
        print(f"Failed to train model for {stratum}: {str(e)}")
        return None

    return best_lr, X_test, y_test, lr_pipeline, X, y

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Male', 'Female']

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
                common_genes = pd.read_csv(CLASSIFICATION_PATH + f"common_genes_{gender}_{age_group}_{visit}.csv", index_col=0)
                ppmi_ad_subset = ppmi_ad_subset[:, ppmi_ad_subset.var.index.isin(common_genes.index.tolist())]
                result = train_logistic_regression(ppmi_ad_subset, f"{gender}_{age_group}_{visit}")
                if result is None:
                    print(f"Failed to train logistic regression for {visit} - skipping")
                    continue
                best_lr, X_test, y_test, lr_pipeline, X, y = result

                with open(result_file, 'a') as f:
                    f.write(f"Visit: {visit}\n")

                y_proba, y_pred = test_classifier(best_lr, X_test, y_test, result_file)
                run_10x_fold_validation(lr_pipeline, X, y, result_file)
                plot = plot_results(y_test, y_proba, y_pred)
                plot.savefig(PATH + f"results_{gender}_{age_group}_{visit}.png")
                plot.clf()
                plot.close()


if __name__ == '__main__':
    main()