from typing import Final
import pandas as pd
import anndata as ad

DEG_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits"
OUTPUT_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/cytoscape"

def main():
    genders = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']
    for gender in genders:
        for age_group in age_groups:
            deg_df = pd.read_csv(f"{DEG_PATH}/DEGs_stratified_consoVisits_{gender}_{age_group}.csv", index_col=0)
            deg_sign = deg_df[(deg_df["log2FoldChange"] > 0.5) & (deg_df["padj"] < 0.05)]
            deg_sign = deg_sign.merge(symbol_ensembl_mapping, left_index=True, right_index=True)[
                ['trunc_eid', 'log2FoldChange', 'padj', 'gene_symbol']]
            cyto_prep = deg_sign.set_index('gene_symbol')
            cyto_prep.to_csv(f"{OUTPUT_PATH}/cytoscape_prep_{gender}_{age_group}.csv")

if __name__ == '__main__':
    main()

