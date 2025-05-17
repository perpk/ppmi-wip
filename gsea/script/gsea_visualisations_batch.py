from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network

def plot_and_save_heatmaps(age_groups, genders, source_filename_prefix, gsea_path, result_file_prefix):
    for gender in genders:
        for age_group in age_groups:
            print(f"Currently working on => Age-Group: {age_group}, Gender: {gender}")
            gsea_results_file = Path(gsea_path) / f"{source_filename_prefix}_{gender}_{age_group}.csv"
            if Path.exists(gsea_results_file) == False:
                print(f"{gsea_path} : no GSEA results available for {gender}, {age_group}")
                continue
            gsea_results = pd.read_csv(gsea_results_file)
            p_value = 'Adjusted P-value'
            filtered_results = gsea_results[gsea_results[p_value] <= 0.05]

            if filtered_results.empty:
                p_value = 'P-value'
                filtered_results = gsea_results[gsea_results[p_value] <= 0.05]
                if filtered_results.empty:
                    print(f"no statistically significant enrichment results found for {gender}, {age_group}")
                    continue
            filtered_results['Gene_list'] = filtered_results['Genes'].str.split(';')
            df_exploded = filtered_results.explode('Gene_list', ignore_index=True)

            heatmap_data = df_exploded.pivot_table(
                index=filtered_results['Gene_set'] + " | " + filtered_results['Term'],
                columns='Gene_list',
                aggfunc='size',
                fill_value=0
            )
            if heatmap_data.empty:
                print(f"heatmap data is empty for {gender}, {age_group}")
                continue

            plt.figure(figsize=(12, 8))
            heatmap = sns.heatmap(heatmap_data, cmap='viridis', cbar=True)
            heatmap.set_title('Heatmap of Genes by Gene Set and Term', fontsize=14)
            heatmap.set_xlabel('Genes', fontsize=12)
            heatmap.set_ylabel('Gene Set | Term', fontsize=12)

            # Rotate column labels (gene names) to be vertical
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"{gsea_path}/{result_file_prefix}_heatmap_{gender}_{age_group}.png")

            G = nx.Graph()

            for _, row in filtered_results.iterrows():
                G.add_node(
                    row["Term"],
                    p_value=row[p_value],
                    category=row["Gene_set"],
                    title=f"Term: {row['Term']}<br>P-value: {row[p_value]}",  # Hover tooltip
                    size=10 * (1 - row[p_value])  # Larger nodes for more significant terms
                )

            # Add edges based on shared genes
            for i, row1 in filtered_results.iterrows():
                genes1 = set(row1["Genes"].split(","))
                for j, row2 in filtered_results.iterrows():
                    if i < j:  # Avoid duplicate edges
                        genes2 = set(row2["Genes"].split(","))
                        shared_genes = genes1 & genes2
                        if shared_genes:
                            G.add_edge(
                                row1["Term"],
                                row2["Term"],
                                weight=len(shared_genes),
                                title=f"Shared genes: {', '.join(shared_genes)}"  # Edge tooltip
                            )

            # ===== 3. Visualize with PyVis =====
            net = Network(notebook=True, height="600px", width="100%", bgcolor="white", font_color="black")

            # Transfer nodes/edges from NetworkX to PyVis
            net.from_nx(G)

            # Customize node colors by category
            category_colors = {"Biological": "#ff7f0e", "Cellular": "#1f77b4"}
            for node in net.nodes:
                node["color"] = category_colors.get(node["category"], "gray")

            # Adjust physics for better layout
            net.toggle_physics(True)
            net.show_buttons(filter_=["physics"])  # Add configuration UI

            # Save and display
            net.save_graph(f"{gsea_path}/{result_file_prefix}_network_{gender}_{age_group}.html")

GSEA_DGE_RESULTS_PATH = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/dge_enr_results"
GSEA_ML_RESULTS_PATH = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/gsea/ml/enr_ml_cons"

def main():
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    genders = ["Male", "Female"]
    # plot_and_save_heatmaps(age_groups, genders, "enr_results_sorted_common_terms", GSEA_DGE_RESULTS_PATH, "dge")
    plot_and_save_heatmaps(age_groups, genders, "enr_results_sorted_common_terms", GSEA_ML_RESULTS_PATH, "ml")

if __name__ == '__main__':
    main()