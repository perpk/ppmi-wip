import pandas as pd
import mygene
import re
import anndata as ad
import numpy as np
import anndata as ad
import numpy as np

genetic_mapping = {
    'ENRLPINK1': 'PINK1',
    'ENRLPRKN': 'PRKN',
    'ENRLSRDC': 'SRDC',
    'ENRLHPSM': 'HPSM',
    'ENRLRBD': 'RBD',
    'ENRLLRRK2': 'LRRK2',
    'ENRLSNCA': 'SNCA',
    'ENRLGBA': 'GBA'
}

def get_genetic_group(row):
    for col, group in genetic_mapping.items():
        if row[col] == 1:
            return group
    return 'None'

def map_age_group(age):
    if 30 <= age <= 50:
        return "30-50"
    elif 50 < age <= 70:
        return "50-70"
    elif 70 < age <= 80:
        return "70-80"
    elif age > 80:
        return ">80"
    return "Unknown"

def main():
    bulk_feature_counts = pd.read_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_clean_counts_meta.csv", index_col=0)
    patient_status = pd.read_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/Participant_Status_19Mar2025.csv")
    genetic_groups = patient_status[['PATNO'] + list(genetic_mapping.keys())].copy()
    genetic_groups['Genetic_Group'] = genetic_groups.apply(get_genetic_group, axis=1)
    bulk_feature_counts = bulk_feature_counts.merge(
        genetic_groups[['PATNO', 'Genetic_Group']],
        left_on='Patient',
        right_on='PATNO',
        how='left'
    ).set_index(bulk_feature_counts.index)

    counts_raw = bulk_feature_counts.iloc[:,bulk_feature_counts.columns.str.startswith("ENSG")]
    counts_raw_t = counts_raw.T
    trunc_eid = [re.sub(r"\.\d+$", "", eid) for eid in counts_raw_t.index.values.tolist()]

    mg = mygene.MyGeneInfo()
    mappings = mg.querymany(
        trunc_eid,
        scopes="ensembl.gene",
        fields="symbol",
        species="human"
    )
    df = pd.DataFrame(mappings)
    df = df[['query', 'symbol']].rename(columns={'query': 'Ensembl_ID', 'symbol': 'Gene_Symbol'})
    df['Gene_Symbol'] = df['Gene_Symbol'].fillna(df['Ensembl_ID'])
    dup_counts = df['Ensembl_ID'].value_counts()
    print(f"Total duplicates: {len(dup_counts[dup_counts > 1])}")
    print(dup_counts[dup_counts > 1])


    df_deduped = df.drop_duplicates(subset=['Ensembl_ID'], keep='first')
    counts_raw_export = counts_raw_t.reset_index(drop=False)
    counts_raw_export['index'] = trunc_eid
    counts_raw_export = counts_raw_export.rename(columns={'index':'Ensembl_ID'})
    counts_raw_export = (
        counts_raw_export.merge(df_deduped, on='Ensembl_ID', how='left')
        .set_index('Gene_Symbol')
        .drop(columns=['Ensembl_ID'])
    )

    bulk_feature_counts['Genetic_Group'] = bulk_feature_counts['Genetic_Group'].fillna('Unknown')

    patient_status['Age_Group'] = patient_status['ENROLL_AGE'].apply(map_age_group)

    age_group_mapping = patient_status[['PATNO', 'Age_Group']]

    bulk_feature_counts = bulk_feature_counts.merge(
        age_group_mapping,
        left_on='Patient',
        right_on='PATNO',
        how='left'
    ).set_index(bulk_feature_counts.index)

    counts_raw_export.index.name = None

    ensembl_symbol_mapping = pd.DataFrame({
        "gene_symbol": counts_raw_export.index,
        "ensembl_id": counts_raw_t.index,
        "trunc_eid": trunc_eid
    }).set_index('ensembl_id')

    ppmi_adata = ad.AnnData(bulk_feature_counts.loc[:, bulk_feature_counts.columns.str.startswith("ENSG")])
    ppmi_adata.obs['Sample'] = bulk_feature_counts.index
    ppmi_adata.obs['Diagnosis'] = bulk_feature_counts['Diagnosis'].values
    ppmi_adata.obs['Visit'] = bulk_feature_counts['Visit'].values
    ppmi_adata.obs['Gender'] = bulk_feature_counts['Gender'].values
    ppmi_adata.obs['Patient'] = bulk_feature_counts['Patient'].values
    ppmi_adata.obs['Genetic_Group'] = bulk_feature_counts['Genetic_Group'].values
    ppmi_adata.obs['Age_Group'] = bulk_feature_counts['Age_Group'].values
    ppmi_adata.obs['Phase'] = bulk_feature_counts['Phase'].values
    ppmi_adata.obs['IR'] = bulk_feature_counts['IR'].values
    ppmi_adata.obs['HudAlphaID'] = bulk_feature_counts['HudAlphaID'].values
    ppmi_adata.layers['counts_log2'] = np.log2(ppmi_adata.X + 1)
    ppmi_adata.layers['gene_symbols'] = counts_raw_export.T
    ppmi_adata.varm['symbol_ensembl_mapping'] = ensembl_symbol_mapping

    counts = ppmi_adata.X.copy()
    cpm = (counts / counts.sum(axis=1)[:, np.newaxis]) * 1e6
    ppmi_adata.layers["log2_cpm"] = np.log2(cpm + 1)

    ppmi_adata.write_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

if __name__ == '__main__':
    main()