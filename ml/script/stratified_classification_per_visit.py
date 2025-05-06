import pandas as pd
import anndata as ad
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def filter_data_by_variation(anndata_object, anndata_layer, target_col, target_value, threshold=0.1):
    X = pd.DataFrame(anndata_object.layers[anndata_layer], columns=anndata_object.var_names)
    y = (anndata_object.obs[target_col] == target_value).astype(int)
    var_selector = VarianceThreshold(threshold=threshold)
    X_highvar = var_selector.fit_transform(X)
    selected_genes = X.columns[var_selector.get_support()]
    print(f"After variance threshold: {len(selected_genes)} genes remaining")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_highvar)
    return X_scaled, y, selected_genes


def anova_f(X_scaled, y, selected_genes, n_top_genes=20000):
    anova_scores, _ = f_classif(X_scaled, y)
    top_indices_anovaf = np.argsort(anova_scores)[-n_top_genes:][::-1]
    anova_selected_genes = selected_genes[top_indices_anovaf]
    anova_scores = anova_scores[top_indices_anovaf]
    return {'genes': anova_selected_genes, 'indices': top_indices_anovaf, 'scores': anova_scores}


def chi_square(X_scaled, y, selected_genes, n_top_genes=20000):
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_processed = discretizer.fit_transform(X_scaled)
    chi2_scores, _ = chi2(X_processed, y)
    top_indices_chi2 = np.argsort(chi2_scores)[-n_top_genes:][::-1]
    chi2_selected_genes = selected_genes[top_indices_chi2]
    chi2_scores = chi2_scores[top_indices_chi2]
    return {'genes': chi2_selected_genes, 'indices': top_indices_chi2, 'scores': chi2_scores}


def mutual_info(X_scaled, y, selected_genes, n_top_genes=20000):
    mi_scores = mutual_info_classif(X_scaled, y)
    top_indices_mi = np.argsort(mi_scores)[-n_top_genes:][::-1]
    mi_selected_genes = selected_genes[top_indices_mi]
    mi_scores = mi_scores[top_indices_mi]
    return {'genes': mi_selected_genes, 'indices': top_indices_mi, 'scores': mi_scores}


def fishers_score(X_scaled, y, selected_genes, n_top_genes=20000):
    fisher_scores= fisher_score.fisher_score(X_scaled, np.asarray(y))
    top_indices_fisher = np.argsort(fisher_scores)[-n_top_genes:][::-1]
    fisher_selected_genes = selected_genes[top_indices_fisher]
    fisher_scores = fisher_scores[top_indices_fisher]
    return {'genes': fisher_selected_genes, 'indices': top_indices_fisher, 'scores': fisher_scores}

def calculate_stratified_importances(anndata_obj_subset, case_diagnosis):
    X_scaled, y, selected_genes = filter_data_by_variation(anndata_obj_subset, 'counts_log2', 'Diagnosis',
                                                           case_diagnosis)
    print("1. Anova")
    anova_results = anova_f(X_scaled, y, selected_genes)
    print("2. Chi-squared")
    chi_square_results = chi_square(X_scaled, y, selected_genes)
    print("3. Mutual Information")
    mutual_info_results = mutual_info(X_scaled, y, selected_genes)
    print("4. Fisher Score")
    fishers_score_results = fishers_score(X_scaled, y, selected_genes)
    return set(anova_results['genes']) & set(chi_square_results['genes']) & set(mutual_info_results['genes']) & set(fishers_score_results['genes'])

def train_logistic_regression(anndata_obj_subset, test_size=0.2, random_state=42):
    print("Training logistic regression model...")
    X = pd.DataFrame(anndata_obj_subset.layers['counts_log2'], columns=anndata_obj_subset.var_names)
    y = (anndata_obj_subset.obs['Diagnosis'] == 'PD').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    param_grid = {
        'lr__C': np.logspace(-3, 3, 7),
        'lr__penalty': ['l1', 'l2'],
        'lr__solver': ['liblinear'],
        'smote__k_neighbors': [3, 5]
    }

    grid_search = GridSearchCV(
        lr_pipeline,
        param_grid,
        cv=StratifiedKFold(10),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_lr = grid_search.best_estimator_
    best_lr.fit(X_train, y_train)
    return best_lr, X_test, y_test, lr_pipeline, X, y


def test_classifier(best_lr, X_test, y_test, result_file):
    print("Testing classifier...")
    y_pred = best_lr.predict(X_test)
    y_proba = best_lr.predict_proba(X_test)[:, 1]
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


def run_10x_fold_validation(lr_pipeline, X, y, result_file):
    print("Running 10x fold validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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


def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['50-70']
    genders = ['Male']

    for gender in genders:
        for age_group in age_groups:

            result_file = f"results_{gender}_{age_group}.txt"
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
                best_lr, X_test, y_test, lr_pipeline, X, y = train_logistic_regression(ppmi_ad_subset)

                with open(result_file, 'a') as f:
                    f.write(f"Visit: {visit}\n")

                y_proba, y_pred = test_classifier(best_lr, X_test, y_test, result_file)
                run_10x_fold_validation(lr_pipeline, X, y, result_file)
                plot = plot_results(y_test, y_proba, y_pred)
                plot.savefig(f"results_{gender}_{age_group}_{visit}.png")
                plot.clf()
                plot.close()

                coefficients = pd.DataFrame({
                    'ensembl_id': common_genes,
                    'coefficient': lr_pipeline.named_steps['lr'].coef_[0],
                    'abs_coef': np.abs(lr_pipeline.named_steps['lr'].coef_[0])
                }).sort_values('abs_coef', ascending=False)




if __name__ == '__main__':
    main()