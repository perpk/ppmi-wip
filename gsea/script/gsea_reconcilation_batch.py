from pathlib import Path

import pandas as pd

def consolidate_and_write_to_csv(source_dir, filename_prefix, output_dir, age_groups, genders, visits):
    for gender in genders:
        for age_group in age_groups:
            print(f"{source_dir} : Currently working on => Visit: {age_group}, Gender: {gender}")
            gsea_dfs_raw = {}
            for visit in visits:
                gsea_file = Path(source_dir) / f"{filename_prefix}_{gender}_{visit}_{age_group}.csv"
                if Path.exists(gsea_file) == False:
                    print(f"{source_dir} : no GSEA results available for {gender}, {age_group}, {visit}")
                    continue;
                gse_df = pd.read_csv(gsea_file)
                gsea_dfs_raw[(gender, age_group, visit)] = gse_df
            if (gsea_dfs_raw == {}):
                continue;
            common_terms = set.intersection(*(set(df['Term']) for df in gsea_dfs_raw.values()))
            if not common_terms:
                print(f"no common terms found for {gender}, {age_group} among visits")
                continue;
            rows_with_common_terms = pd.concat([df[df['Term'].isin(common_terms)] for df in gsea_dfs_raw.values()], ignore_index=True)
            rows_with_common_terms = rows_with_common_terms.sort_values(by="Adjusted P-value", ascending=True)
            rows_with_common_terms.to_csv(f"{output_dir}/enr_results_sorted_common_terms_{gender}_{age_group}.csv")

GSEA_PATH_DEG = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea"
GSEA_RESULTS_PATH_DEG = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/dge_enr_results"
GSEA_PATH_ML = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/ml"
GSEA_RESULTS_PATH_ML = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/ml/enr_ml_cons"

def main():
    visits = ["BL", "V02", "V04", "V06", "V08"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    genders = ["Male", "Female"]
    consolidate_and_write_to_csv(GSEA_PATH_DEG, "enr_results_sorted", GSEA_RESULTS_PATH_DEG, age_groups, genders, visits)
    consolidate_and_write_to_csv(GSEA_PATH_ML, "enr_ml_results_sorted", GSEA_RESULTS_PATH_ML, age_groups, genders, visits)

if __name__ == '__main__':
    main()