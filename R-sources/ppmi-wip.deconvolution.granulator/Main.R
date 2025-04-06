setwd('/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq')

rna_expr <- read.csv("/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/ppmi_133_rna_counts_for_deconv.tsv", sep="\t", header=TRUE)
library(dplyr)
rna_expr_proc <- rna_expr %>% distinct(Gene_Symbol, .keep_all=TRUE)
rownames(rna_expr_proc) <- rna_expr_proc$Gene_Symbol
rna_expr_proc = rna_expr_proc[,-1]

install.packages('Seurat')
library(Seurat)

gse184916_scRNA <- Read10X(data.dir="/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/GSE184916/")

so <- CreateSeuratObject(counts = gse184916_scRNA, project = "GSE184916")
so <- NormalizeData(so)

BiocManager::install("granulator")
library(granulator)

# rna_matrix <- as.matrix(rna_expr)
rna_matrix <- as.matrix(rna_expr_proc[, sapply(rna_expr_proc, is.numeric)])
rownames(rna_matrix) <- rownames(rna_expr_proc)
##
library(Matrix)

pseudobulk_ref <- matrix(
  Matrix::rowSums(gse184916_scRNA),
  ncol = 1,
  dimnames = list(
    rownames(gse184916_scRNA),  # Keep gene names
    "Pseudobulk_Reference"       # Required column name
  )
)
##

common_genes <- intersect(rownames(rna_matrix), rownames(pseudobulk_ref))
rna_matrix <- rna_matrix[common_genes, ]
pseudobulk_ref <- pseudobulk_ref[common_genes, , drop = FALSE] 

decon <- deconvolute(rna_matrix, celltype_ref, method = "nnls")

head(rna_matrix)
any(rna_matrix < 0)
any(gse184916_pseudobulk < 0) 


seurat_obj <- CreateSeuratObject(gse184916_scRNA)
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)
seurat_obj <- FindNeighbors(seurat_obj)
seurat_obj <- FindClusters(seurat_obj)

# Get cluster markers and manually annotate
markers <- FindAllMarkers(seurat_obj)
seurat_obj$cell_type <- paste0("Cluster_", seurat_obj$seurat_clusters)

# Create cell-type reference
celltype_ref <- AverageExpression(seurat_obj, 
                                  group.by = "cell_type",
                                  assays = "RNA",
                                  slot = "counts")$RNA


