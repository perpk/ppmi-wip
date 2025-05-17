import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np


def get_common_genes_across_visits(filtered_dfs: dict, min_visits: int = 2) -> pd.DataFrame:
    genes_per_visit = {}

    for visit, df in filtered_dfs.items():
        for _, row in df.iterrows():
            gene = row['Gene']
            if gene not in genes_per_visit:
                genes_per_visit[gene] = {'Gene': gene}
                for v in filtered_dfs.keys():
                    genes_per_visit[gene][v] = False
                    genes_per_visit[gene][f"{v}_log2FC"] = None
            genes_per_visit[gene][visit] = True
            genes_per_visit[gene][f"{visit}_log2FC"] = row['log2FoldChange']

    genes_df = pd.DataFrame.from_dict(genes_per_visit.values())

    genes_df['visit_count'] = genes_df[list(filtered_dfs.keys())].sum(axis=1)
    common_genes_df = genes_df[genes_df['visit_count'] >= min_visits].drop(columns=['visit_count'])

    log2fc_columns = [f"{v}_log2FC" for v in filtered_dfs.keys()]
    non_log2fc_columns = [col for col in common_genes_df.columns if col not in log2fc_columns]
    common_genes_df = common_genes_df[non_log2fc_columns + log2fc_columns]
    common_genes_df = common_genes_df.set_index('Gene')

    return common_genes_df

def visualize_amounts_of_up_and_down_regulated_genes(dfs):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        count_up = ((df['log2FoldChange'] >= 0.5) & (df['log2FoldChange'] < 10)).sum()
        count_down = ((df['log2FoldChange'] <= -0.5) & (df['log2FoldChange'] > -10)).sum()
        ax.bar(['Upregulated > 0.5', 'Downregulated < -0.5'], [count_up, count_down], color=['red', 'blue'])
        ax.set_title(title)
        ax.set_ylabel('Gene Count')
        ax.set_xlabel('Expression')

    plt.tight_layout()
    return plt

def visualize_amounts_of_extremely_up_and_down_regulated_genes(dfs, extreme_threshold=10):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        count_up = (df['log2FoldChange'] >= extreme_threshold).sum()
        count_down = (df['log2FoldChange'] <= -extreme_threshold).sum()
        ax.bar([f"Upregulated > {extreme_threshold}", f"Downregulated < -{extreme_threshold}"], [count_up, count_down], color=['red', 'blue'])
        ax.set_title(title)
        ax.set_ylabel('Gene Count')
        ax.set_xlabel('Expression')

    plt.tight_layout()
    return plt

def visualize_volcano_plots(dfs):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=False)
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        significant = (df['log2FoldChange'].abs() >= 0.5) & (df['padj'] <= 0.05)
        ax.scatter(df['log2FoldChange'], -np.log10(df['padj']), color='gray', s=10, alpha=0.6, label='Non-significant')
        ax.scatter(df.loc[significant, 'log2FoldChange'],
                   -np.log10(df.loc[significant, 'padj']),
                   color='red',
                   s=10,
                   label='Significant')
        ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=0.8)
        ax.axvline(x=-0.5, color='blue', linestyle='--', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('log$_2$ Fold Change')
        ax.set_ylabel('-log$_{10}$(p-value)')
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    return plt

def visualize_volcano_plots_exclude_log2fc(dfs, log2FC_threshold=10):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=False)
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        filtered_df = df[(df['log2FoldChange'].abs() <= log2FC_threshold)]
        significant = (filtered_df['log2FoldChange'].abs() >= 0.5) & (filtered_df['padj'] <= 0.05)
        ax.scatter(filtered_df['log2FoldChange'], -np.log10(filtered_df['padj']), color='gray', s=10, alpha=0.6,
                   label='Non-significant')
        ax.scatter(filtered_df.loc[significant, 'log2FoldChange'],
                   -np.log10(filtered_df.loc[significant, 'padj']),
                   color='red',
                   s=10,
                   label='Significant')
        ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=0.8)
        ax.axvline(x=-0.5, color='blue', linestyle='--', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('log$_2$ Fold Change')
        ax.set_ylabel('-log$_{10}$(p-value)')
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    return plt

def main():
    gender = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]
    visits = ["BL", "V02", "V04", "V06", "V08"]

    deg_data_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/dge_stratified/"
    deg_results_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/dge_stratified/results"

    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    symbol_ensembl_mapping = ppmi_ad.varm['symbol_ensembl_mapping']

    dfs = {};
    dfs_filtered = {};
    for gender in gender:
        for age_group in age_groups:
            for visit in visits:
                df = pd.read_csv(deg_data_path + f"DEGs_stratified_{gender}_{visit}_{age_group}.csv")
                df_filtered = df[(df['log2FoldChange'].abs() >= 0.5) & (df['padj'] <= 0.05)]
                dfs_filtered[visit] = df_filtered
                dfs[visit] = df

            up_down_barplot = visualize_amounts_of_up_and_down_regulated_genes(dfs_filtered)
            up_down_barplot.savefig(deg_results_path + f"/up_down_barplot_{gender}_{age_group}.png")

            extreme_up_down_barplot = visualize_amounts_of_extremely_up_and_down_regulated_genes(dfs_filtered, extreme_threshold=10)
            extreme_up_down_barplot.savefig(deg_results_path + f"/extreme_up_down_barplot_{gender}_{age_group}.png")

            common_genes_across_visits = get_common_genes_across_visits(dfs)
            common_genes_across_visits = common_genes_across_visits.merge(symbol_ensembl_mapping, left_index=True, right_index=True)
            common_genes_across_visits.to_csv(
                deg_results_path + f"/common_genes_across_visits_{gender}_{age_group}.csv")

            volcano_plot = visualize_volcano_plots(dfs)
            volcano_plot.savefig(deg_results_path + f"/volcano_plot_{gender}_{age_group}.png")

            volcano_plot_exclude_log2_fc = visualize_volcano_plots_exclude_log2fc(dfs, log2FC_threshold=10)
            volcano_plot_exclude_log2_fc.savefig(deg_results_path + f"/volcano_plot_exclude_Log2FC_{gender}_{age_group}.png")



if __name__ == '__main__':
    main()
