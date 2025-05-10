from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def get_dynamic_stratified_kfold(y, default_splits=10):
    unique, counts = np.unique(y, return_counts=True)
    min_splits = min(counts.min(), default_splits)
    n_splits = max(2, min_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def run_10x_fold_validation(lr_pipeline, X, y, result_file):
    print("Running 10x fold validation...")
    cv = get_dynamic_stratified_kfold(y)
    cv_scores = {'roc_auc': [], 'pr_auc': []}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lr_pipeline.fit(X_train, y_train)
        y_proba = lr_pipeline.predict_proba(X_val)[:, 1]

        cv_scores['roc_auc'].append(roc_auc_score(y_val, y_proba))
        cv_scores['pr_auc'].append(average_precision_score(y_val, y_proba))

    avg_roc_auc = np.mean(cv_scores['roc_auc'])
    std_roc_auc = np.std(cv_scores['roc_auc'])
    avg_pr_auc = np.mean(cv_scores['pr_auc'])
    std_pr_auc = np.std(cv_scores['pr_auc'])

    print("\nCross-validation results:")
    print(f"ROC-AUC: {avg_roc_auc:.3f} ± {std_roc_auc:.3f}")
    print(f"PR-AUC: {avg_pr_auc:.3f} ± {std_pr_auc:.3f}")
    # Append results to file
    with open(result_file, 'a') as f:
        f.write("10x fold validation results:\n")
        f.write(f"ROC-AUC: {avg_roc_auc:.3f} ± {std_roc_auc:.3f}\n")
        f.write(f"PR-AUC: {avg_pr_auc:.3f} ± {std_pr_auc:.3f}\n")
        f.write("\n")

def plot_results(y_test, y_proba, y_pred):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    fpr, tpr, roc_threshold = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.subplot(1, 3, 2)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=f'PR (AUC = {average_precision_score(y_test, y_proba):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'PD'],
                yticklabels=['Control', 'PD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    return plt