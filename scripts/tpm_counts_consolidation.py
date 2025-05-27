import pandas as pd
import os

def main():
    # Set the directory path
    directory = "/Volumes/Elements/quant"

    # Collect all .sf files from the specified directory
    tpm_files = [f for f in os.listdir(directory) if f.endswith(".genes.sf")]
    tpm_data = {}

    for file in tpm_files:
        # Use full path to read the file
        file_path = os.path.join(directory, file)
        sample_id = file.split(".longRNA")[0]  # Extract sample ID
        df = pd.read_csv(file_path, sep="\t", usecols=["Name", "TPM"], index_col="Name")
        tpm_data[sample_id] = df["TPM"]

    # Merge into a matrix (genes x samples)
    tpm_matrix = pd.concat(tpm_data, axis=1)

    # Save the output to current working directory
    tpm_matrix.to_csv("/Volumes/Elements/quant/PPMI_TPM_for_CIBERSORTx.txt", sep="\t")

    print(f"Successfully processed {len(tpm_files)} files. Output saved to PPMI_TPM_for_CIBERSORTx.txt")

if __name__ == "__main__":
    main()