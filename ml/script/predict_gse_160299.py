import os
from typing import Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, classification_report)
import seaborn as sns
from pathlib import Path

PROJECT_ROOT :Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq"
REPORT_DIR :Final = f"{PROJECT_ROOT}/classifier_reports/classification/GSE160299/results"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_and_process_data(file_path):
    """
    Load and process the GSE dataset from local filesystem
    """
    print(f"Loading dataset from {file_path}...")

    # Handle both compressed and uncompressed files
    if file_path.endswith('.gz'):
        counts_df = pd.read_csv(file_path, sep='\t', compression='gzip')
    else:
        counts_df = pd.read_csv(file_path, sep='\t')

    # Set gene names as index (assuming first column is gene names)
    counts_df = counts_df.set_index(counts_df.columns[0])

    # Log2 transform (add 1 to avoid log(0))
    print("Applying log2 transformation...")
    log_counts = np.log2(counts_df + 1)

    return log_counts


def load_classifier(classifier_path):
    resolved_path = Path(classifier_path).expanduser().resolve()
    if not resolved_path.exists():
        return None
    classifier = load(resolved_path)
    return classifier


def generate_classification_report(y_true, y_pred, y_proba, class_labels, classifier_name):
    """
    Generate comprehensive evaluation report with plots
    """
    report = {
        'classification_report': classification_report(y_true, y_pred, target_names=class_labels, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'roc_curve': {},
        'pr_curve': {}
    }

    # ROC Curve (for binary classification)
    if len(class_labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        report['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {classifier_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(REPORT_DIR, f'{classifier_name}_roc_curve.png'))
        plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if len(class_labels) == 2 else y_proba,
                                                  pos_label=1)
    report['pr_curve'] = {'precision': precision, 'recall': recall}

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {classifier_name}')
    plt.savefig(os.path.join(REPORT_DIR, f'{classifier_name}_pr_curve.png'))
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(report['confusion_matrix'], annot=True, fmt='d',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(REPORT_DIR, f'{classifier_name}_confusion_matrix.png'))
    plt.close()

    # Save text report
    with open(os.path.join(REPORT_DIR, f'{classifier_name}_report.txt'), 'w') as f:
        f.write(f"Classifier: {classifier_name}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_labels))
        f.write(f"\nConfusion Matrix:\n{report['confusion_matrix']}\n")
        if len(class_labels) == 2:
            f.write(f"\nROC AUC: {roc_auc:.4f}\n")

    return report


def run_predictions(data, joblib, name, true_labels=None, class_labels=None):
    """
    Run predictions and generate evaluation reports
    """
    results = {}
    evaluate = true_labels is not None
    try:
        classifier = joblib['model']
        classifier_features = [feature.rsplit('.', 1)[0] for feature in joblib['features']]
        common_features = set(data.index) & set(classifier_features)
        if len(common_features) == 0:
            raise ValueError("No common features found between dataset and classifier")

        data_aligned = data[data.index.isin(common_features)]
        feature_order = {feat: idx for idx, feat in enumerate(classifier_features)}
        data_aligned = data_aligned.loc[[f for f in classifier_features if f in common_features]]
        X = data_aligned.T
        X_processed = normalize(X, norm='l2')

        # Make predictions
        predictions = classifier.predict(X_processed)
        proba = classifier.predict_proba(X_processed) if hasattr(classifier, 'predict_proba') else None

        results = {
            'predictions': predictions,
            'probabilities': proba,
            'features_used': list(common_features),
            'features_missing': len(classifier_features) - len(common_features)
        }

        # Generate reports if we have true labels
        if evaluate and proba is not None:
            print(f"Generating evaluation report for {name}...")
            report = generate_classification_report(
                true_labels, predictions, proba, class_labels, name
            )
            results['report'] = report

    except Exception as e:
        print(f"Error running {name}: {str(e)}")
        results[name] = None

    return results

def main():
    # Configuration - UPDATE THESE PATHS
    data_path = f"{PROJECT_ROOT}/GSE160299/GSE160299_Raw_gene_counts_matrix.txt"  # or .txt.gz

    classifiers_out_dirs = ["LR2", "SVM2", "RF2", "XGBOOST2"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    genders = ["Female", "Male"]
    visits = ["BL", "V02", "V04", "V06", "V08"]
    log_counts = load_and_process_data(data_path)
    print(f"Data shape: {log_counts.shape}")
    print("First few rows of data:")
    print(log_counts.head())

    for gender in genders:
        for age_group in age_groups:
            for visit in visits:
                for classifier_out_dir in classifiers_out_dirs:
                    print(f"Loading classifier {classifier_out_dir} for {gender} {age_group} {visit}...")
                    joblib = load_classifier(f"{PROJECT_ROOT}/classification/{classifier_out_dir}/model_{gender}_{age_group}_{visit}.joblib")
                    if joblib is None:
                        print(f"There is no classifier for {gender} {age_group} {visit}...")
                        continue

                    print(f"features = {len(joblib['features'])}")
                    true_labels = None
                    class_labels = None
                    # classifier_features = [feature.rsplit('.', 1)[0] for feature in classifier['features']]

                    # log_counts_filtered = log_counts[log_counts.index.isin(classifier_features)]

                    # prediction_results = run_predictions(
                    #     log_counts,
                    #     joblib,
                    #     name=f"{classifier_out_dir}_{gender}_{age_group}_{visit}",
                    #     true_labels=true_labels,
                    #     class_labels=class_labels
                    # )

                    # print("\nAnalysis complete. Reports saved to:", os.path.abspath(REPORT_DIR))

if __name__ == "__main__":
    main()