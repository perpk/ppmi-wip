import warnings
from typing import Final

import anndata as ad
import pandas as pd
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
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.exceptions import FitFailedWarning
import xgboost as xgb
import numpy as np

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/XGBOOST2/"
CLASSIFICATION_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/"

def train_xgboost(anndata_obj_subset, stratum, test_size=0.2, random_state=42, min_samples=5):
    print("XGBoost Model Training")
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

    steps.append(('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='auc')))

    xgboost_pipeline = Pipeline(steps)

    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [4, 6],
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__subsample': [0.8, 0.9],
        'xgb__colsample_bytree': [0.8, 0.9]
    }

    if use_smote:
        param_grid['smote__k_neighbors'] = [
            min(3, min(train_class_counts) - 1),
            min(5, min(train_class_counts) - 1),
        ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            grid_search = GridSearchCV(estimator=xgboost_pipeline,
                                       param_grid=param_grid,
                                       scoring='roc_auc',
                                       cv=StratifiedKFold(3, shuffle=True, random_state=42),  # Fewer folds for speed
                                       n_jobs=-1,
                                       verbose=2)
            grid_search.fit(X_train, y_train)
            best_xgboost = grid_search.best_estimator_
            best_xgboost.fit(X_train, y_train)
            model_path = os.path.join(PATH + f"model_{stratum}.joblib")
            joblib.dump({
                'model': best_xgboost,
                'X_test': X_test,
                'y_test': y_test,
                'features': X.columns.tolist()
            }, model_path)
    except Exception as e:
        print(f"Failed to train model for {stratum}: {str(e)}")
        return None

    return best_xgboost, X_test, y_test, xgboost_pipeline, X, y

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
                common_genes = pd.read_csv(CLASSIFICATION_PATH + f"common_genes_{gender}_{age_group}_{visit}.csv",
                                           index_col=0)
                ppmi_ad_subset = ppmi_ad_subset[:, ppmi_ad_subset.var.index.isin(common_genes.index.tolist())]
                result = train_xgboost(ppmi_ad_subset, f"{gender}_{age_group}_{visit}")
                if result is None:
                    print(f"Failed to train XGBoost for {visit} - skipping")
                    continue
                best_xgboost, X_test, y_test, xgboost_pipeline, X, y = result
                with open(result_file, 'a') as f:
                    f.write(f"Visit: {visit}\n")
                y_proba, y_pred = test_classifier(best_xgboost, X_test, y_test, result_file)
                run_10x_fold_validation(xgboost_pipeline, X, y, result_file)
                plot = plot_results(y_test, y_proba, y_pred)
                plot.savefig(PATH + f"results_{gender}_{age_group}_{visit}.png")
                plot.clf()
                plot.close()

if __name__ == '__main__':
    main()
