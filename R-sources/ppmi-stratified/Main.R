setwd('/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq')

counts <- read.csv("counts.csv", row.names=1)
metadata <- read.csv("metadata.csv", row.names=1)

library(DESeq2)

metadata_filtered <- subset(metadata, Gender=="Male" & 
                              Visit == "BL" &
                              Diagnosis %in% c("PD", "Control") & 
                              Age_Group == "50-70")

counts_filtered <- counts[rownames(metadata_filtered),]
counts_filtered_t <- t(counts_filtered)
dds <- DESeqDataSetFromMatrix(
  countData=counts_filtered_t,
  colData=metadata_filtered,
  design = ~ Diagnosis
)

dds <- DESeq(dds)
res <- results(dds, contrast=c("Diagnosis", "PD", "Control"))

res_sorted <- res[order(-abs(res$log2FoldChange)), ]
res_sorted <- res_sorted[order(res_sorted$padj, na.last=TRUE),]
head(res_sorted)

res_df <- as.data.frame(res_sorted)
write.csv(res_sorted, "dge_stratified/ppmi-deg_males_50-70_BL_PD-vs-Ctrl.csv")

res_df$gene <- rownames(res_df)
library(dplyr)
res_df <- res_df %>%
  mutate(
    significant = ifelse(
      abs(log2FoldChange) >= 0.5 & padj <= 0.05,
      "Significant",
      "Not significant"
    )
  )

library(ggplot2)
library(ggrepel)

top_genes <- res_df %>%
  filter(significant == "Significant") %>%
  arrange(padj) %>%
  head(10)  # Adjust number of genes to label

ggplot(res_df, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = significant), alpha = 0.6, size = 2) +
  scale_color_manual(values = c("Not significant" = "gray", "Significant" = "red")) +
  theme_minimal() +
  labs(
    x = "log2(Fold Change)",
    y = "-log10(Adjusted p-value)",
    title = "PD vs Control - Males, Baseline Visit, ages 50-70"
  ) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "blue") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +
  geom_text_repel(  # Replaces geom_text
    data = top_genes,
    aes(label = gene),
    size = 3,
    box.padding = 0.5,  # Adjust spacing around labels
    point.padding = 0.3, # Space between labels and points
    max.overlaps = 100,  # Increase to show more labels
    segment.color = "grey50"  # Color of label connector lines
  )

