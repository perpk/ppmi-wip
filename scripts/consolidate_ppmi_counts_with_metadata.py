import pandas as pd

def main():
    metadata = pd.read_csv("/Volumes/Elements/metaDataIR3.csv", index_col=0)
    metadata = metadata[metadata["QCflagIR3"].str.lower() == "pass"]
    metadata = metadata[["PATNO", "GENDER", "CLINICAL_EVENT", "DIAGNOSIS"]]
    metadata.rename(
        columns={"PATNO": "Patient", "GENDER": "Gender", "CLINICAL_EVENT": "Visit", "DIAGNOSIS": "Diagnosis"},
        inplace=True)

    counts = pd.read_csv("/Volumes/Elements/PD-Pre-Proc/scripts/ppmi_counts_matrix.csv", index_col=0)
    counts_t = counts.T
    counts_t.index = counts_t.index.str.replace(r"\.featureCounts.*", "", regex=True)
    # counts_t = counts_t[counts_t.index.str.contains("IR3", na=False)]

    counts_t["Phase"] = counts_t.index.str.extract(r"(Phase\d?)", expand=False)

    counts_t["IR"] = counts_t.index.str.extract(r"(IR\d{1})", expand=False)

    pattern = r"(\d+\-SL-\d+)"
    counts_t["HudAlphaID"] = counts_t.index.str.extract(pattern, expand=False)

    merged = pd.merge(metadata, counts_t, left_index=True, right_on="HudAlphaID")
    merged.to_csv("/Volumes/Elements/counts/ppmi_clean_counts_meta.csv", index=True)

if __name__ == '__main__':
    main()
