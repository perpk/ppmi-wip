library(DESeq2)
library(tibble)
library(tidyverse)

setwd('/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq')

metadata <- read.csv("metadata.csv", row.names=1)
counts <-  read.csv("counts_matrix.csv", row.names=1)

design <- "Visit + Diagnosis"

strata <- list(
  list(Gender="Male", Age_Group="30-50", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Male", Age_Group="50-70", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Male", Age_Group="70-80", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Male", Age_Group=">80", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Female", Age_Group="30-50", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Female", Age_Group="50-70", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Female", Age_Group="70-80", Diagnosis=c("PD", "Control"), Design=design),
  list(Gender="Female", Age_Group=">80", Diagnosis=c("PD", "Control"), Design=design)
)

results_list <- lapply(strata, function(stratum) {
  print(paste("design=",stratum$Design))
  meta_stratum <- metadata %>% filter(Gender == stratum$Gender,
                                      Age_Group == stratum$Age_Group,
                                      # Visit == stratum$Visit,
                                      Diagnosis %in% stratum$Diagnosis)
  counts_stratum <- counts[rownames(meta_stratum), ]
  dds <- DESeqDataSetFromMatrix(
    t(counts_stratum),
    meta_stratum,
    design = formula(paste("~ ", stratum$Design))
  )
  dds <- DESeq(dds)
  (dds)
})

names(results_list) <- sapply(strata, function(s) paste(s$Gender, s$Age_Group, sep="_"))
for (i in seq_along(results_list)) {
  stratum_name <- ifelse(is.null(names(results_list)),
                         paste0("Stratum_", i),
                         names(results_list)[i])
  res <- results(results_list[[i]], contrast=c("Diagnosis", "PD", "Control")) %>%
    as.data.frame() %>%
    rownames_to_column("Gene") %>%
    arrange(padj, desc(abs(log2FoldChange)))
  filename <- file.path("./data/deg_consolidated_visits", paste0("DEGs_stratified_consoVisits_", stratum_name, ".csv"))
  write.csv(res, file=filename, row.names=FALSE)
}