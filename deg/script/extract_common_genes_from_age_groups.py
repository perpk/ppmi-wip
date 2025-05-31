import pandas as pd
import matplotlib.pyplot as plt


def main():
    deg_data_path = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/data/deg_consolidated_visits/results"

    age_groups = ["30-50", "50-70", "70-80", ">80"]

    for age_group in age_groups:
        sign_deg_males = pd.read_csv(f"{deg_data_path}/deg_sign_genes_stratified_consoVisits_Male_{age_group}.csv",
                                     index_col=0)
        sign_deg_females = pd.read_csv(f"{deg_data_path}/deg_sign_genes_stratified_consoVisits_Female_{age_group}.csv",
                                       index_col=0)

        common_genes = sign_deg_males.index.intersection(sign_deg_females.index)
        common_genes_df = pd.DataFrame({
            'log2FoldChange_Male': sign_deg_males.loc[common_genes, 'log2FoldChange'],
            'log2FoldChange_Female': sign_deg_females.loc[common_genes, 'log2FoldChange'],
            'gene_symbol': sign_deg_males.loc[common_genes, 'gene_symbol']
        })

        common_genes_df.to_csv(f"{deg_data_path}/common_genes_{age_group}.csv", index=True)

        # Prepare data for combined bubble plot
        common_genes_male = common_genes_df[['gene_symbol', 'log2FoldChange_Male']].copy()
        common_genes_male['Gender'] = 'Male'
        common_genes_male['Abs_log2FoldChange'] = common_genes_male['log2FoldChange_Male'].abs()
        common_genes_male['Color'] = common_genes_male['log2FoldChange_Male'].apply(
            lambda x: 'red' if x > 0 else 'blue')

        common_genes_female = common_genes_df[['gene_symbol', 'log2FoldChange_Female']].copy()
        common_genes_female = common_genes_female.rename(columns={'log2FoldChange_Female': 'log2FoldChange'})
        common_genes_female['Gender'] = 'Female'
        common_genes_female['Abs_log2FoldChange'] = common_genes_female['log2FoldChange'].abs()
        common_genes_female['Color'] = common_genes_female['log2FoldChange'].apply(lambda x: 'red' if x > 0 else 'blue')

        # Combine Male and Female data
        combined_data = pd.concat([common_genes_male.rename(columns={'log2FoldChange_Male': 'log2FoldChange'}),
                                   common_genes_female])

        # Create bubble plot
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.tick_params(axis='y', rotation=45)

        scatter = ax.scatter(
            combined_data['Gender'],
            combined_data['gene_symbol'],
            s=combined_data['Abs_log2FoldChange'] * 200,  # Scale bubble size
            c=combined_data['Color'],
            alpha=0.8
        )
        for tick in ax.get_yticks():
            ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Common Male and \nFemale DEGs - Age Group {age_group}")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Gene Symbol")
        ax.set_xticks(["Male", "Female"], minor=False)

        # Save the figure
        plt.tight_layout()
        fig.savefig(f"{deg_data_path}/bubbleplot_combined_{age_group}.png")
        plt.close()


if __name__ == "__main__":
    main()