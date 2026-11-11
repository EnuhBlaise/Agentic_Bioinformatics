#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(DESeq2))
a <- commandArgs(trailingOnly = TRUE)
counts <- read.delim(a[1], row.names = 1, check.names = FALSE)
meta <- read.delim(a[2], row.names = 1, check.names = FALSE, stringsAsFactors = TRUE)
meta <- meta[colnames(counts), , drop = FALSE]
dds <- DESeqDataSetFromMatrix(round(as.matrix(counts)), meta, as.formula(a[3]))
dds <- DESeq(dds)
res <- results(dds, contrast = c(a[4], a[5], a[6]))
dir.create(a[7], showWarnings = FALSE, recursive = TRUE)
out <- as.data.frame(res[order(res$pvalue), ])
write.table(cbind(gene = rownames(out), out), file.path(a[7], "deseq2_results.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
