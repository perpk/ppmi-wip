library(DESeq2)

setwd('/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq')

counts <- read.csv("counts.csv", row.names = 1)
metadata <- read.csv("metadata.csv", row.names = 1)

metadata_filtered <- subset(metadata, Visit == "BL" & Diagnosis %in% c("PD", "Control"))
counts_filtered <- counts[rownames(metadata_filtered),]

BiocManager::install("sva")
library(sva)

mod <- model.matrix(~ Diagnosis + Gender + Age_Group, metadata_filtered)
mod0 <- model.matrix(~ Gender + Age_Group, metadata)

svaseq <- svaseq(t(counts_filtered), mod, mod0)
metadata$SV1 <- svseq$sv[, 1]
metadata$SV2 <- svseq$sv[, 2]

dds <- DESeqDataSetFromMatrix(
  countData = t(counts_filtered),
  colData = metadata_filtered,
  design = ~ Gender + Age_Group + Diagnosis + Gender:Diagnosis + Age_Group:Diagnosis
)

dds <- DESeq(dds)
res_main <- results(dds, name = "Diagnosis_PD_vs_Control")
res_gender <- results(dds, contrast = list(c("Diagnosis_PD_vs_Control", "GenderMale.DiagnosisPD")))
res_age50_70 <- results(dds, contrast = list(c("Diagnosis_PD_vs_Control", "Age_Group50-70.DiagnosisPD")))