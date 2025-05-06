import anndata as ad
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2
)
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from collections import defaultdict


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


def calculate_stratified_importances(gender, age_group, visit, diagnosis, case_diagnosis, anndata_obj):
    mask = ((anndata_obj.obs['Age_Group'] == age_group) &
            (anndata_obj.obs['Gender'] == gender) &
            (anndata_obj.obs['Diagnosis'].isin(diagnosis)) &
            (anndata_obj.obs['Visit'] == visit))
    anndata_obj_subset = anndata_obj[mask]
    X_scaled, y, selected_genes = filter_data_by_variation(anndata_obj_subset, 'counts_log2', 'Diagnosis',
                                                           case_diagnosis)
    anova_results = anova_f(X_scaled, y, selected_genes, 5000)
    chi_square_results = chi_square(X_scaled, y, selected_genes, 5000)
    mutual_info_results = mutual_info(X_scaled, y, selected_genes, 5000)
    fishers_score_results = fishers_score(X_scaled, y, selected_genes, 5000)
    return {'ANOVA': anova_results, 'Chi-squared': chi_square_results, 'Mutual Information': mutual_info_results,
            'Fisher Score': fishers_score_results}


def extract_common_genes_by_intersection(*genes):
    return pd.DataFrame({"Gene": list(set.intersection(*map(set, genes)))})


def extract_common_genes_by_borda_ranks(method_collection):
    for method in method_collection:
        # Sort genes by score (descending) and assign ranks
        sorted_indices = np.argsort(-np.array(method_collection[method]['scores']))
        method_collection[method]['rank'] = np.arange(1, len(sorted_indices) + 1)
        method_collection[method]['genes'] = [method_collection[method]['genes'][i] for i in sorted_indices]
    # Get all unique genes across methods
    gene_lists = [method_collection[method]['genes'] for method in method_collection]
    all_genes = list(set().union(*gene_lists))

    # Initialize Borda scores (default: high penalty for missing genes)
    max_rank = max(len(method['genes']) for method in method_collection.values()) + 1
    borda_scores = defaultdict(int)

    for gene in all_genes:
        for method in method_collection:
            if gene in method_collection[method]['genes']:
                borda_scores[gene] += method_collection[method]['genes'].index(gene) + 1  # Rank = position + 1
            else:
                borda_scores[gene] += max_rank  # Penalize missing genes
    ranked_genes = sorted(all_genes, key=lambda x: borda_scores[x])
    return pd.DataFrame({
        "Gene": ranked_genes,
        "Borda_Score": [borda_scores[gene] for gene in ranked_genes],
        "Rank": np.arange(1, len(ranked_genes) + 1)
    })


def main(subset=False):
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    if subset:
        ppmi_ad = ppmi_ad[:, :100]

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Male', 'Female']

    for gender in genders:
        for age_group in age_groups:
            consolidated_results = None
            for visit in visits:
                print(f"Calculating stratified importances for {gender} - age group {age_group} and visit {visit}")
                results = calculate_stratified_importances(gender, age_group, visit, ['Control', 'PD'], 'PD', ppmi_ad)
                common_genes_by_borda_ranks = extract_common_genes_by_borda_ranks(results)
                common_genes_by_borda_ranks.rename(columns={"Rank": visit}, inplace=True)
                if consolidated_results is None:
                    consolidated_results = common_genes_by_borda_ranks.set_index("Gene")[[visit]]
                else:
                    consolidated_results = consolidated_results.join(
                        common_genes_by_borda_ranks.set_index("Gene")[[visit]], how="outer"
                    )
            consolidated_results.to_csv(
                f"/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/feature_selection/borda_ranks_{gender}_{age_group}.csv",
                index=True)


if __name__ == '__main__':
    main()
