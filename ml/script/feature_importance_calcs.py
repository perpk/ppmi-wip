import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import numpy as np
from skfeature.function.similarity_based import fisher_score

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
    print(f"ANOVA F-Score => n_top_genes: {n_top_genes}")
    anova_scores, _ = f_classif(X_scaled, y)
    top_indices_anovaf = np.argsort(anova_scores)[-n_top_genes:][::-1]
    anova_selected_genes = selected_genes[top_indices_anovaf]
    anova_scores = anova_scores[top_indices_anovaf]
    return {'genes': anova_selected_genes, 'indices': top_indices_anovaf, 'scores': anova_scores}


def chi_square(X_scaled, y, selected_genes, n_top_genes=20000):
    print(f"Chi-Square => n_top_genes: {n_top_genes}")
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_processed = discretizer.fit_transform(X_scaled)
    chi2_scores, _ = chi2(X_processed, y)
    top_indices_chi2 = np.argsort(chi2_scores)[-n_top_genes:][::-1]
    chi2_selected_genes = selected_genes[top_indices_chi2]
    chi2_scores = chi2_scores[top_indices_chi2]
    return {'genes': chi2_selected_genes, 'indices': top_indices_chi2, 'scores': chi2_scores}


def mutual_info(X_scaled, y, selected_genes, n_top_genes=20000):
    print(f"Mutual Information => n_top_genes: {n_top_genes}")
    mi_scores = mutual_info_classif(X_scaled, y)
    top_indices_mi = np.argsort(mi_scores)[-n_top_genes:][::-1]
    mi_selected_genes = selected_genes[top_indices_mi]
    mi_scores = mi_scores[top_indices_mi]
    return {'genes': mi_selected_genes, 'indices': top_indices_mi, 'scores': mi_scores}


def fishers_score(X_scaled, y, selected_genes, n_top_genes=20000):
    print(f"Fisher's score => n_top_genes: {n_top_genes}")
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


def calculate_stratified_importances(anndata_obj_subset, case_diagnosis, n_top_genes=20000):
    X_scaled, y, selected_genes = filter_data_by_variation(anndata_obj_subset, 'counts_log2', 'Diagnosis',
                                                           case_diagnosis)
    print(f"n_top_genes: {n_top_genes}")
    print("1. Anova")
    anova_results = anova_f(X_scaled, y, selected_genes, n_top_genes=n_top_genes)
    print("2. Chi-squared")
    chi_square_results = chi_square(X_scaled, y, selected_genes, n_top_genes=n_top_genes)
    print("3. Mutual Information")
    mutual_info_results = mutual_info(X_scaled, y, selected_genes, n_top_genes=n_top_genes)
    print("4. Fisher Score")
    fishers_score_results = fishers_score(X_scaled, y, selected_genes, n_top_genes=n_top_genes)
    return set(anova_results['genes']) & set(chi_square_results['genes']) & set(mutual_info_results['genes']) & set(
        fishers_score_results['genes'])