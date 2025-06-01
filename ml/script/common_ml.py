from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import (StratifiedKFold, GroupKFold)
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


def run_10x_fold_validation(pipeline, X, y, path, stratum, title, groups=None):
    print(f"Running 10x fold validation on {stratum}")

    # Initialize CV
    if groups is not None:
        cv = GroupKFold(n_splits=10)
        print("Using GroupKFold to prevent group leakage")
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=10)
        print("Using StratifiedKFold (no groups provided)")

    plt.title(title)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Metric storage
    mean_fpr = np.linspace(0, 1, 100)
    roc_tprs, roc_aucs = [], []
    mean_recall = np.linspace(0, 1, 100)
    pr_precisions, pr_aps = [], []
    baseline_pr = len(y[y == 1]) / len(y)

    # Cross-validation loop
    for fold, (train, test) in enumerate(cv.split(X, y, groups=groups if groups is not None else y)):
        if len(np.unique(y.iloc[test])) < 2:
            print(f"Fold {fold} skipped: Only one class present")
            continue

        pipeline.fit(X.iloc[train], y.iloc[train])
        y_proba = pipeline.predict_proba(X.iloc[test])[:, 1]

        if len(np.unique(y_proba)) == 1:
            print(f"Fold {fold} skipped: Constant predictions")
            continue

        # ROC Curve
        fpr, tpr, _ = roc_curve(y.iloc[test], y_proba)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_tprs.append(interp_tpr)
        ax1.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC = {roc_auc:.2f})')

        # PR Curve
        precision, recall, _ = precision_recall_curve(y.iloc[test], y_proba)
        ap = average_precision_score(y.iloc[test], y_proba)
        pr_aps.append(ap)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        pr_precisions.append(interp_precision)
        ax2.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold {fold} (AP = {ap:.2f})')

    # === ROC Plot ===
    ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Chance', alpha=0.8)
    mean_tpr = np.mean(roc_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std_roc_auc = np.std(roc_aucs)
    ax1.plot(mean_fpr, mean_tpr, 'b-',
             label=f'Mean ROC (AUC = {mean_roc_auc:.2f} ± {std_roc_auc:.2f})', lw=2)
    ax1.fill_between(mean_fpr,
                     np.maximum(mean_tpr - np.std(roc_tprs, axis=0), 0),
                     np.minimum(mean_tpr + np.std(roc_tprs, axis=0), 1),
                     color='grey', alpha=0.2)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            xlabel='False Positive Rate', ylabel='True Positive Rate',
            title='ROC Curves by Fold')
    ax1.legend(loc="lower right")

    # === PR Plot ===
    ax2.plot([0, 1], [baseline_pr, baseline_pr], 'r--', lw=2, label='Baseline', alpha=0.8)
    mean_precision = np.mean(pr_precisions, axis=0)
    mean_ap = np.mean(pr_aps)
    std_ap = np.std(pr_aps)
    ax2.plot(mean_recall, mean_precision, 'b-',
             label=f'Mean PR (AP = {mean_ap:.2f} ± {std_ap:.2f})', lw=2)
    ax2.fill_between(mean_recall,
                     np.maximum(mean_precision - np.std(pr_precisions, axis=0), 0),
                     np.minimum(mean_precision + np.std(pr_precisions, axis=0), 1),
                     color='grey', alpha=0.2)
    ax2.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
            xlabel='Recall', ylabel='Precision',
            title='Precision-Recall Curves by Fold')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{path}{stratum}_k_fold_validation.png")
    plt.close()

    print(f"Mean ROC AUC: {mean_roc_auc:.3f} ± {std_roc_auc:.3f}")
    print(f"Mean Average Precision: {mean_ap:.3f} ± {std_ap:.3f}")
    return (mean_roc_auc, std_roc_auc), (mean_ap, std_ap)


def __run_10x_fold_validation(lr_pipeline, X, y, result_file):
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

def plot_results(y_test, y_proba, y_pred, plot_title):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    fpr, tpr, roc_threshold = roc_curve(y_test, y_proba)
    plt.title(plot_title)
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