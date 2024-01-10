# Install DESeq2 package
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("DESeq2")

# Import library
library(DESeq2)

# Upload CountMatrix file
count_matrix <- read.csv("YourFile.csv")

# Upload MetaData
MetaData <- read.csv('metadata.csv')

# Create a DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = count_matrix,
                              colData = col_data, # replace with your sample metadata
                              design = ~ condition) # replace with your experimental design

# Perform DESeq2 normalization
dds <- DESeq(dds)
dds

# PCA on normalized counts
pcaData <- plotPCA(dds, intgroup = "condition", returnData = TRUE)

# Plot PCA
plotPCA(dds, intgroup = "condition", col = c("red", "blue"))

# Identify potential outliers
outliers <- pcaData$PC1 > threshold_value | pcaData$PC2 > threshold_value
outlier_samples <- rownames(pcaData[outliers, ])
