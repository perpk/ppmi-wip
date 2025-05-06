import pandas as pd


def main():
    borda_m_50_70 = pd.read_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/feature_selection/borda_ranks_Male_50-70.csv", index_col=0)
    intersection_m_50_70 = pd.read_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/feature_selection_consensus_males_50-70.csv", index_col=0)

    common_genes = intersection_m_50_70.index.intersection(borda_m_50_70.index)
    common_genes.to_series().to_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/borda_intersection_m_50_70.csv")

    exclusive_genes = intersection_m_50_70.index.difference(borda_m_50_70.index)
    exclusive_genes.to_series().to_csv(
        "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/exclusive_intersection_m_50_70.csv",
        index=False)

if __name__ == '__main__':
    main()