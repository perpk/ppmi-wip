library(DESeq2)

setwd('/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq')

# cell_props = pd.read_csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/cibersortx_metadata_df.csv", index_col=0)
# filtered_cell_meta = filtered_cell_meta.loc[:, filtered_cell_meta.columns.isin(["Diagnosis", "Neutrophils", "Monocytes", "T cells CD4 naive", "NK cells activated", "NK cells resting", "T cells CD4 memory resting"])]
# filtered_cell_meta = filtered_cell_meta.loc[:, filtered_cell_meta.sum(axis=0) != 0]
# design_factors = filtered_cell_meta.columns.values.tolist()
# 

counts <- read.csv("counts.csv", row.names = 1)
metadata <- read.csv("metadata.csv", row.names = 1)
cell_props <- read.csv("cibersortx_metadata_df.csv", row.names=1)


rownames(metadata)
library(tidyverse)

cell_props <- cell_props %>% select(where(~ is.numeric(.) && sum(., na.rm=TRUE) > 0))
cell_props <- cell_props[, !(names(cell_props) %in% c("RMSE", "Correlation"))]
metadata <- cbind(metadata, cell_props[row.names(metadata), ])

design_full <- "T.cells.CD4.memory.resting + T.cells.CD4.naive + NK.cells.resting + Monocytes + Neutrophils + Diagnosis"

design_reduced <- "Monocytes + Neutrophils + Diagnosis"

strata <- list(
  # list(Gender="Male", Age_Group="30-50", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="50-70", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="70-80", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group=">80", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_reduced),
  list(Gender="Female", Age_Group="30-50", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="50-70", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="70-80", Visit="BL", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group=">80", Visit="BL", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
  
  # list(Gender="Male", Age_Group="30-50", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="50-70", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="70-80", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group=">80", Visit="V02", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
  list(Gender="Female", Age_Group="30-50", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="50-70", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="70-80", Visit="V02", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group=">80", Visit="V02", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
  
  # list(Gender="Male", Age_Group="30-50", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="50-70", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="70-80", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group=">80", Visit="V04", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
  list(Gender="Female", Age_Group="30-50", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="50-70", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="70-80", Visit="V04", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group=">80", Visit="V04", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
   
  # list(Gender="Male", Age_Group="30-50", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="50-70", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="70-80", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group=">80", Visit="V06", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
  list(Gender="Female", Age_Group="30-50", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="50-70", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="70-80", Visit="V06", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group=">80", Visit="V06", Diagnosis=c("PD", "Control"), Design="Diagnosis"),
   
  # list(Gender="Male", Age_Group="30-50", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full),
  # list(Gender="Male", Age_Group="50-70", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full)
  # list(Gender="Male", Age_Group="70-80", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full)
  # list(Gender="Male", Age_Group=">80", Visit="V08", Diagnosis=c("PD", "Control"), Design="Diagnosis")
  list(Gender="Female", Age_Group="30-50", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="50-70", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group="70-80", Visit="V08", Diagnosis=c("PD", "Control"), Design=design_full),
  list(Gender="Female", Age_Group=">80", Visit="V08", Diagnosis=c("PD", "Control"), Design="Diagnosis")
)

for (stratum in strata) {
  meta_stratum <- metadata %>% filter(Gender == stratum$Gender,
                                      Age_Group == stratum$Age_Group,
                                      Visit == stratum$Visit,
                                      Diagnosis %in% stratum$Diagnosis)
  n_samples <- nrow(meta_stratum)
  n_coeffs <- 8
  if (n_samples <= n_coeffs) {
    print(paste("For stratum ", stratum$Age_Group,", ",  stratum$Gender," there are ", n_samples, " samples but ", n_coeffs, " coefficients."))
  }
}

results_list <- lapply(strata, function(stratum) {
  print(paste("design=",stratum$Design))
  meta_stratum <- metadata %>% filter(Gender == stratum$Gender,
                                      Age_Group == stratum$Age_Group,
                                      Visit == stratum$Visit,
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

# View(results_list)
names(results_list) <- sapply(strata, function(s) paste(s$Gender, s$Visit, s$Age_Group, sep="_"))
library(tibble)
for (i in seq_along(results_list)) {
  stratum_name <- ifelse(is.null(names(results_list)),
                         paste0("Stratum_", i),
                         names(results_list)[i])
  res <- results(results_list[[i]], contrast=c("Diagnosis", "PD", "Control")) %>%
    as.data.frame() %>%
    rownames_to_column("Gene") %>%
    arrange(padj, desc(abs(log2FoldChange)))
  filename <- file.path("./dge_stratified", paste0("DEGs_stratified_", stratum_name, ".csv"))
  write.csv(res, file=filename, row.names=FALSE)
}

# BiocManager::install("metaRNASeeq")
# library(metaRNASeq)
# pvals <- sapply(results_list, function(x) x$pvalue)
# log2FCs <- sapply(results_list, function(x) x$log2FoldChange)
# combined_pvals <- fishercomb(pvals)

#metadata_filtered <- subset(metadata, Visit == "BL" & Diagnosis %in% c("PD", "Control") & Age_Group == "50-70" & Gender == "Male")
#counts_filtered <- counts[rownames(metadata_filtered),]
#head(metadata_filtered)
#dim(metadata_filtered)
#head(cell_props)
#dim(cell_props)

# library(dplyr)
# cell_props <- cell_props %>% select(where(~ is.numeric(.) && sum(., na.rm=TRUE) > 0))
# dim(cell_props)
# 
# cell_props <- cell_props[, !(names(cell_props) %in% c("RMSE", "Correlation"))]
# colnames(cell_props)
# 
# metadata_with_cell_props <- cbind(metadata_filtered, cell_props[row.names(metadata_filtered), ])
# 
# dds <- DESeqDataSetFromMatrix(
#   countData = t(counts_filtered),
#   colData = metadata_with_cell_props,
#   design = ~ T.cells.CD4.memory.resting + 
#              T.cells.CD4.naive +
#              NK.cells.activated + 
#              NK.cells.resting +
#              Monocytes + 
#              Neutrophils +
#              Diagnosis  
#   )
# 
# dds <- DESeq(dds)
# res <- results(dds, contrast=c("Diagnosis", "PD", "Control"))