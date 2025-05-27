import pandas as pd
import os
from mygene import MyGeneInfo  # For ID conversion

# Initialize MyGeneInfo for ID mapping
mg = MyGeneInfo()


def translate_ensembl_to_symbol(ensembl_ids):
    """Convert Ensembl IDs (with versions) to gene symbols."""
    # Remove version numbers (e.g., ENSG000001234.5 → ENSG000001234)
    ensembl_ids_no_ver = [x.split('.')[0] for x in ensembl_ids]

    # Query MyGeneInfo (batch mode for efficiency)
    results = mg.querymany(ensembl_ids_no_ver, scopes='ensembl.gene', fields='symbol', species='human')

    # Create mapping dictionary
    id_to_symbol = {}
    for res in results:
        if 'symbol' in res:
            original_id = ensembl_ids[ensembl_ids_no_ver.index(res['query'])]
            id_to_symbol[original_id] = res['symbol']

    return id_to_symbol


# Directory containing your .sf files
directory = "/Volumes/Elements/quant"
tpm_files = [f for f in os.listdir(directory) if f.endswith(".genes.sf")]

# Load LM22 genes (replace with actual path to LM22)
lm22 = pd.read_csv("/Volumes/Elements/quant/LM22.txt", sep='\t')
lm22_genes = set(lm22.iloc[:, 0])  # Assuming gene symbols are in first column

# Process files
tpm_data = {}
for file in tpm_files:
    file_path = os.path.join(directory, file)
    sample_id = file.split(".longRNA")[0]

    # Read Salmon output
    df = pd.read_csv(file_path, sep="\t", usecols=["Name", "TPM"], index_col="Name")

    # Store TPM values for this sample
    tpm_data[sample_id] = df["TPM"]

def main():
    # Combine all samples
    tpm_matrix = pd.concat(tpm_data, axis=1)

    # Translate Ensembl IDs to symbols (only do this once)
    print("Translating Ensembl IDs to gene symbols...")
    ensembl_ids = tpm_matrix.index.tolist()
    id_map = translate_ensembl_to_symbol(ensembl_ids)

    # Apply translation and filter to LM22 genes
    tpm_matrix.index = tpm_matrix.index.map(id_map)
    tpm_matrix = tpm_matrix[~tpm_matrix.index.isna()]  # Remove unmapped genes
    tpm_matrix = tpm_matrix[tpm_matrix.index.isin(lm22_genes)]  # Keep only LM22 genes

    # Save the filtered matrix
    output_path = "/Volumes/Elements/quant/PPMI_TPM_LM22_Compatible.txt"
    tpm_matrix.to_csv(output_path, sep="\t")

    print(f"Success! Saved filtered TPM matrix ({tpm_matrix.shape[0]} genes) to {output_path}")

if __name__ == "__main__":
    main()