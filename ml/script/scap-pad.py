from typing import Final

import pandas as pd
import anndata as ad

PROJECT_ROOT :Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq"


def main():
    data_path = f"{PROJECT_ROOT}/GSE160299/GSE160299_Raw_gene_counts_matrix.txt"
    counts_df = pd.read_csv(data_path, sep='\t')

    common_genes_male_50_70_bl = pd.read_csv(
        "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification_500/common_genes_Male_50-70_BL.csv",
        index_col=0)
    classifier_features = [feature.rsplit('.', 1)[0] for feature in common_genes_male_50_70_bl.index]
    print(f"GSE160299 shape {counts_df.shape}")
    print(f"Number of classifier features: {len(classifier_features)}")
    matching_features_count = sum(counts_df['gene_id'].isin(classifier_features))
    print(f"Number of non-matching features: {matching_features_count}")

    print("========================================================")

    borda_ranks_male_50_70_bl = pd.read_csv(
        "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/feature_selection/borda_ranks_Male_50-70.csv",
        index_col=0)
    borda_ranks_male_50_70_bl.index = [feature.rsplit('.', 1)[0] for feature in borda_ranks_male_50_70_bl.index]
    ranked_features = borda_ranks_male_50_70_bl.index.tolist()
    print(f"Number of ranked features: {len(ranked_features)}")
    matching_ranked_features_counts = sum(counts_df['gene_id'].isin(ranked_features))
    print(f"Number of unmatched ranked features: {matching_ranked_features_counts}")

    print("========================================================")

    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    print(len(ppmi_ad.var_names))
    ppmi_ad_varnames = [var.rsplit('.', 1)[0] for var in ppmi_ad.var_names]
    non_matching_vars = counts_df[~counts_df['gene_id'].isin(ppmi_ad_varnames)]
    print(f"Number of non-matching vars: {len(non_matching_vars)}")
    print(f"Non-matching vars: {non_matching_vars['gene_id'].tolist()}")


if __name__ == '__main__':
    main()