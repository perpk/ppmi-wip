import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

param_grid_lr = {
    'lr__C': np.logspace(-3, 3, 7),
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear']
}


def create_logistic_regression_classifier(random_state):
    return ('lr', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    ))


param_grid_svm = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto', 0.1],
    'svm__kernel': ['linear', 'rbf']
}


def create_svm_classifier(random_state):
    return 'svm', SVC(probability=True, random_state=random_state, class_weight='balanced', C=1.0)


param_grid_rf = {
    'rf__n_estimators': [200, 500],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2]
}


def create_random_forest_classifier(random_state):
    return ('rf', RandomForestClassifier(
        random_state=random_state, class_weight='balanced',
        n_jobs=-1))


def create_xgboost_classifier(random_state, y):
    return ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
        random_state=random_state,
        n_jobs=-1,
        eval_metric='auc'))


param_grid_xgboost = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [4, 6],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8, 0.9],
    'xgb__colsample_bytree': [0.8, 0.9]
}
