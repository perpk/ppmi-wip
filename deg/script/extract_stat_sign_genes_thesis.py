import pandas as pd
import anndata as ad

def main():
    deg_data_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits"
    deg_results_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/results"

    genders = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']

    for gender in genders:
        for age_group in age_groups:
            deg_df = pd.read_csv(f"{deg_data_path}/DEGs_stratified_consoVisits_{gender}_{age_group}.csv", index_col=0)
            deg_sign = deg_df[(deg_df["log2FoldChange"].abs() > 0.5) & (deg_df["padj"] < 0.05)]
            deg_sign['abs_log2FoldChange'] = deg_sign['log2FoldChange'].abs()
            deg_sign = deg_sign.sort_values(by=["abs_log2FoldChange", "padj"], ascending=[False, True])
            deg_sign = deg_sign.merge(symbol_ensembl_mapping, left_index=True, right_index=True)

            deg_sign.to_csv(f"{deg_results_path}/deg_sign_genes_stratified_consoVisits_{gender}_{age_group}.csv")

if __name__ == '__main__':
    main()