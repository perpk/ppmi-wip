import warnings

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
from sklearn.exceptions import FitFailedWarning
from common_ml import  get_dynamic_stratified_kfold
from ML_classifiers import *

def create_param_grid(classifier, random_state, y=None):
    if classifier == 'LR':
        return param_grid_lr, create_logistic_regression_classifier(random_state)
    if classifier == 'SVM':
        return param_grid_svm, create_svm_classifier(random_state)
    if classifier == 'RF':
        return param_grid_rf, create_random_forest_classifier(random_state)
    if classifier == 'XGBOOST':
        return param_grid_xgboost, create_xgboost_classifier(random_state, y)
    raise ValueError(f"Invalid classifier: {classifier}")

def train_classifier(anndata_obj_subset, stratum, classifier, results_path, use_smote=True, test_size=0.2,
                     random_state=42, min_samples=5):
    print(f"Training {classifier} model...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
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

    train_idx, test_idx = next(splitter.split(X, y, groups=anndata_obj_subset.obs['Patient']))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_class_counts = y_train.value_counts()
    steps = [
        ('scaler', StandardScaler())
    ]

    if use_smote:
        # Re-evaluate whether to use smote based on the amount of samples available
        use_smote = min(train_class_counts) >= min_samples
        if use_smote:
            steps.append(('smote', SMOTE(
                random_state=random_state,
                k_neighbors=min(3, min(train_class_counts) - 1))))
        else:
            print(f"Warning: Not using SMOTE for {stratum} - smallest class has {min(train_class_counts)} samples")

    param_grid, classifier_instance = create_param_grid(classifier, random_state, y if classifier == 'XGBOOST' else None)

    steps.append(classifier_instance)

    pipeline = Pipeline(steps)

    if use_smote:
        param_grid['smote__k_neighbors'] = [
            min(3, min(train_class_counts) - 1),
            min(5, min(train_class_counts) - 1),
        ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=get_dynamic_stratified_kfold(y_train),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            best_estimator.fit(X_train, y_train)

            model_path = os.path.join(results_path + f"model_{classifier}_{stratum}.joblib")
            joblib.dump({
                'model': best_estimator,
                'X_test': X_test,
                'y_test': y_test,
                'X_train': X_train,
                'y_train': y_train,
                'features': X.columns.tolist()
            }, model_path)
    except Exception as e:
        print(f"Failed to train model for {stratum}: {str(e)}")
        return None

    return best_estimator, X_test, y_test, pipeline, X, y, X_train
