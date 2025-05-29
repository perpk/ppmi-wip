from typing import Final

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

GSEA_PATH = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/gsea"
RESULTS_PATH = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/gsea/results"

GENE_SETS: Final = ['MSigDB_Hallmark_2020',
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_gene_set_heatmaps(df: pd.DataFrame, filename_prefix, pval_threshold: float = 0.05):
    for gene_set in GENE_SETS:
        # Filter by gene set and non-empty genes
        subset = df[
            df['Gene_set'].str.contains(gene_set, case=False, na=False) &
            df['Genes'].notna()
            ].copy()

        if subset.empty:
            print(f"Skipping {gene_set}: No genes match.")
            continue

        # Split genes into lists and explode to one gene per row
        subset['Genes'] = subset['Genes'].str.split(';')
        exploded = subset.explode('Genes')

        # Pivot table: Term x Genes, values = Adjusted P-value
        heatmap_data = exploded.pivot_table(
            index='Term',
            columns='Genes',
            values='Adjusted P-value',
            aggfunc='first'  # Keep the first p-value per term-gene pair
        )

        # Filter for significant terms/genes (optional)
        heatmap_data = heatmap_data.loc[
            heatmap_data.index[heatmap_data.min(axis=1) <= pval_threshold],
            heatmap_data.columns[heatmap_data.min(axis=0) <= pval_threshold]
        ]

        if heatmap_data.empty:
            print(f"Skipping {gene_set}: No significant terms/genes after filtering.")
            continue

        # Plot (no annotations)
        plt.figure(figsize=(12, 8))
        heatmap_data = -np.log10(heatmap_data)
        sns.heatmap(
            heatmap_data,
            cmap='viridis',  # Reversed colormap (darker = more significant)
            cbar_kws={'label': '-log10(Adjusted P-value)'},
            linewidths=0.5,
            mask=heatmap_data.isna(),  # Hide NaN (gene not in term)
            annot=False,  # Disable text in cells
        )
        plt.title(f"Gene-Term Significance: {gene_set}", fontsize=14)
        plt.xlabel('Gene')
        plt.ylabel('Term')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PATH}/{filename_prefix}_{gene_set}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()


def find_and_save_common_terms_across_groups(gsea_dfs: dict, output_file: str):
    # Create a dictionary to store common terms
    terms_data = defaultdict(lambda: defaultdict(dict))

    # Iterate through each gender and age group
    for (gender, age_group), df in gsea_dfs.items():
        for _, row in df.iterrows():
            term = row['Term']
            p_value = row['Adjusted P-value']
            terms_data[term][(gender, age_group)] = p_value

    # Filter common terms across all groups
    common_terms = [term for term, groups in terms_data.items() if len(groups) == len(gsea_dfs)]
    common_data = []

    for term in common_terms:
        row = {'Term': term}
        for (gender, age_group), p_value in terms_data[term].items():
            row[f"{gender}_{age_group}"] = p_value
        common_data.append(row)

    # Save results to CSV
    common_df = pd.DataFrame(common_data)
    common_df.to_csv(output_file, index=False)


def main():
    filename_pattern = "enr_results_sorted_consoVisits"
    genders = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]

    # for gender in genders:
    #     for age_group in age_groups:
    #         print(f"Processing {gender} {age_group}")
    #         filename = f"{GSEA_PATH}/{filename_pattern}_{gender}_{age_group}.csv"
    #         gsea_df = pd.read_csv(filename)
    #         gsea_df_filtered = gsea_df[gsea_df["Adjusted P-value"] < 0.05]
    #         generate_gene_set_heatmaps(gsea_df_filtered, f"gsea_results_{gender}_{age_group}")

    # Collect GSEA DataFrames for all groups
    gsea_dfs = {}
    for gender in genders:
        for age_group in age_groups:
            filename = f"{GSEA_PATH}/{filename_pattern}_{gender}_{age_group}.csv"
            gsea_dfs[(gender, age_group)] = pd.read_csv(filename)

    # Process each gene set and find common terms
    for gene_set in GENE_SETS:
        print(f"Finding common terms for {gene_set}")
        filtered_dfs = {
            (gender, age_group): df[df['Gene_set'].str.contains(gene_set, case=False, na=False)]
            for (gender, age_group), df in gsea_dfs.items()
        }
        output_file = f"{RESULTS_PATH}/common_terms_{gene_set}.csv"
        find_and_save_common_terms_across_groups(filtered_dfs, output_file)


if __name__ == "__main__":
    main()