from typing import Final

import anndata as ad
import csv

from ml.script.feature_importance_calcs import calculate_stratified_importances

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification"

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Male', 'Female']


    for gender in genders:
        for age_group in age_groups:
            consensus_genes_per_visit = {}
            for visit in visits:
                print(f"Visit: {visit}, Age Group: {age_group}, Gender: {gender}")
                mask = ((ppmi_ad.obs['Age_Group'] == age_group) &
                        (ppmi_ad.obs['Gender'] == gender) &
                        (ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])) &
                        (ppmi_ad.obs['Visit'] == visit))
                ppmi_ad_subset = ppmi_ad[mask]
                common_genes = calculate_stratified_importances(ppmi_ad_subset, 'PD')
                consensus_genes_per_visit[visit] = common_genes
            union_genes = set.union(*consensus_genes_per_visit.values())
            csv_file_path = f"{PATH}/common_union_visit_genes_{gender}_{age_group}.csv"
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Gene"])
                writer.writerows([[gene] for gene in union_genes])

if __name__ == '__main__':
    main()