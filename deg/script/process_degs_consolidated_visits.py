import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_amounts_of_up_and_down_regulated_genes(dfs):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        count_up = ((df['log2FoldChange'] > 0.5) & (df['padj'] < 0.05)).sum()
        count_down = ((df['log2FoldChange'] < -0.5) & (df['padj'] < 0.05)).sum()
        ax.bar(['Upregulated > 0.5', 'Downregulated < -0.5'], [count_up, count_down], color=['red', 'blue'])
        ax.set_title(title)
        ax.set_ylabel('Gene Count')
        ax.set_xlabel('Expression')

    plt.tight_layout()
    return plt

def visualize_volcano_plots(dfs):
    fig, axes = plt.subplots(1, 4, figsize=(25, 5), sharey=False)
    titles = list(dfs.keys())

    for ax, df, title in zip(axes, list(dfs.values()), titles):
        significant = (df['log2FoldChange'].abs() > 0.5) & (df['padj'] < 0.05)
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

def main():
    gender = ["Male", "Female"]
    age_groups = ["30-50", "50-70", "70-80", ">80"]

    deg_data_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/"
    deg_results_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/results"

    dfs = {};
    dfs_filtered = {};
    for gender in gender:
        for age_group in age_groups:
            df = pd.read_csv(deg_data_path + f"DEGs_stratified_consoVisits_{gender}_{age_group}.csv")
            df_filtered = df[(df['log2FoldChange'].abs() > 0.5) & (df['padj'] < 0.05)]
            dfs_filtered[age_group] = df_filtered
            dfs[age_group] = df

        volcano_plot = visualize_volcano_plots(dfs)
        volcano_plot.savefig(deg_results_path + f"/volcano_plot_consoVisits_{gender}.png")
        bar_plot = visualize_amounts_of_up_and_down_regulated_genes(dfs)
        bar_plot.savefig(deg_results_path + f"/barplot_plot_consoVisits_{gender}.png")


if __name__ == '__main__':
    main()
