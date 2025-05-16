from pathlib import Path
from typing import Final
import pandas as pd
import anndata as ad
import gseapy as gp

CLASSIFICATION_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification"
GSEA_ML_PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/ml"

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    classificator_dirs = ["LR2", "SVM2", "RF2", "XGBOOST2"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    genders = ["Male", "Female"]
    visits = ["BL", "V02", "V04", "V06", "V08"]
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
    n_top_features = 200
    debug = False

    for classificator in classificator_dirs:
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
                    feature_importances_file = Path(CLASSIFICATION_PATH) / f"{classificator}/feature_importance_data_{gender}_{age_group}_{visit}.csv"
                    if Path.exists(feature_importances_file) == False:
                        print(f"file {feature_importances_file} does not exist")
                        continue;
                    feature_importances = pd.read_csv(feature_importances_file, index_col=1)
                    feature_importances = feature_importances[feature_importances['abs_importance'] != 0]
                    feature_importances = feature_importances.sort_values(by='abs_importance', ascending=False).head(
                        n_top_features)

                    feature_importances = feature_importances.merge(symbol_ensembl_mapping, left_index=True, right_index=True)
                    ranked_genes = feature_importances.set_index('gene_symbol')['abs_importance'].sort_values(ascending=False)
                    ranked_genes = ranked_genes[~ranked_genes.isna()]
                    ranked_genes = ranked_genes.sort_values(ascending=False, key=abs)
                    if ranked_genes.empty:
                        continue

                    enr = gp.enrichr(gene_list=ranked_genes.index.tolist(),
                                     gene_sets=gene_sets,
                                     organism='human')
                    enr_results_sorted = enr.results.sort_values(by='Adjusted P-value', ascending=True)
                    enr_results_sorted.to_csv(f"{GSEA_ML_PATH}/enr_ml_results_sorted_{gender}_{visit}_{age_group}.csv")

                    if debug:
                        print(f"{classificator}_{age_group}_{gender}_{visit} = {feature_importances.shape}")
                        min_abs_importance = feature_importances['abs_importance'].min()
                        max_abs_importance = feature_importances['abs_importance'].max()
                        print(f"Min abs_importance: {min_abs_importance}, Max abs_importance: {max_abs_importance}")


if __name__ == '__main__':
    main()