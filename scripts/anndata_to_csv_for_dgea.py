import anndata as ad
import pandas as pd

def main():
    ppmi_ad = ad.read_h5ad("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")

    counts = ppmi_ad.X
    pd.DataFrame(counts, index=ppmi_ad.obs_names, columns=ppmi_ad.var_names).to_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/counts_matrix.csv")

    # Export metadata
    ppmi_ad.obs.to_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/metadata.csv")

if __name__ == '__main__':
    main()
