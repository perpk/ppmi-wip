from typing import Final

import anndata as ad
import pandas as pd
import numpy as np
import gseapy as gp

DEG_DATA_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/dge_stratified/"
GSEA_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea"


def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['>80']
    genders = ['Female']
    gene_sets = ['MSigDB_Hallmark_2020',
                 'KEGG_2021_Human',
                 'WikiPathways_2024_Human',
                 'Human_Phenotype_Ontology',
                 'GO_Biological_Process_2023',
                 'GO_Molecular_Function_2023',
                 'GO_Cellular_Component_2023',
                 'SynGO_2024',
                 'OMIM_Disease',
                 'ARCHS4_TFs_Coexp',
                 'ChEA_2013',
                 'ChEA_2015',
                 'ChEA_2016',
                 'ChEA_2022',
                 'ENCODE_TF_ChIP-seq_2014',
                 'ENCODE_TF_ChIP-seq_2015',
                 'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
                 'Enrichr_Submissions_TF-Gene_Coocurrence',
                 'TF-LOF_Expression_from_GEO',
                 'TF_Perturbations_Followed_by_Expression',
                 'TRRUST_Transcription_Factors_2019']

    for gender in genders:
        for age_group in age_groups:
            for visit in visits:
                print(f"Visit: {visit}, Age Group: {age_group}, Gender: {gender}")
                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])) &
                        (ppmi_ad.obs['Visit'] == visit))
                ppmi_ad_subset = ppmi_ad[mask]
                symbol_ensembl_mapping = ppmi_ad_subset.varm['symbol_ensembl_mapping']
                deg_data = pd.read_csv(f"{DEG_DATA_PATH}/DEGs_stratified_{gender}_{visit}_{age_group}.csv", index_col=0)
                deg_sign = deg_data[
                    (np.abs(deg_data['log2FoldChange']) >= 0.5) & (deg_data['padj'] <= 0.05)]
                deg_sign = deg_sign.merge(symbol_ensembl_mapping, left_index=True, right_index=True)
                ranked_genes = deg_sign.set_index('gene_symbol')['stat'].sort_values(ascending=False)
                ranked_genes = ranked_genes[~ranked_genes.isna()]
                ranked_genes = ranked_genes.sort_values(ascending=False, key=abs)
                if ranked_genes.empty:
                    continue

                enr = gp.enrichr(gene_list=ranked_genes.index.tolist(),
                                 gene_sets=gene_sets,
                                 organism='human')
                enr_results_sorted = enr.results.sort_values(by='Adjusted P-value', ascending=True)
                enr_results_sorted.to_csv(f"{GSEA_PATH}/enr_results_sorted_{gender}_{visit}_{age_group}.csv")


if __name__ == '__main__':
    main()
