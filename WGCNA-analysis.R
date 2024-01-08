# Library
library(WGCNA)
library(DESeq2)
library(GEOquery)
library(tidyverse)
library(CorLevelPlot)
library(gridExtra)

allowWGCNAThreads() 

# Import normalised count matrix
# Read the CSV file
data <- read.csv("YourFile.csv")

# transposed the data
norm.count <- data %>%
  t()

# Network Construction  ---------------------------------------------------
# Choose a set of soft-thresholding powers
power <- c(c(1:10), seq(from = 11, to = 20, by = 1))

# Call the network topology analysis function
sft <- pickSoftThreshold(norm.count,
                         powerVector = power,
                         networkType = "signed",
                         verbose = 5)

sft.data <- sft$fitIndices

# visualization to pick power

a1 <- ggplot(sft.data, aes(Power, SFT.R.sq, label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  geom_hline(yintercept = 0.8, color = 'red') +
  labs(x = 'Power', y = 'Scale free topology model fit, signed R^2') +
  theme_classic()


a2 <- ggplot(sft.data, aes(Power, mean.k., label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  labs(x = 'Power', y = 'Mean Connectivity') +
  theme_classic()


grid.arrange(a1, a2, nrow = 2)

# convert matrix to numeric
norm.count[] <- sapply(norm.count, as.numeric)

soft_power <- 11
temp_cor <- cor
cor <- WGCNA::cor

# blockwise network construction
# memory estimate w.r.t blocksize
# choose maxBlockSize based on your computational power

bwnet <- blockwiseModules(norm.count,
                          maxBlockSize = 20000,
                          TOMType = "signed",
                          power = soft_power,
                          mergeCutHeight = 0.25,
                          numericLabels = FALSE,
                          randomSeed = 1234,
                          verbose = 0)


cor <- temp_cor

# Module Eigengenes ---------------------------------------------------------
module_eigengenes <- bwnet$MEs

# Print out a preview
head(module_eigengenes)

# get number of genes for each module
table(bwnet$colors)

# Subset unmergedColors to match the length of subset_colors

unique_order <- unique(bwnet$dendrograms[[1]]$order)
subset_colors <- bwnet$colors[unique_order]
subset_unmergedColors <- bwnet$unmergedColors[unique_order]

# Create a data frame with the required columns
plot_data <- data.frame(
  unmergedColors = subset_unmergedColors,
  mergedColors = subset_colors
)

# Use the data frame in the plotDendroAndColors function
plotDendroAndColors(
  bwnet$dendrograms[[1]],
  plot_data,
  c("Dyanamic Tree Cut", "Merged Dynamic"),
  dendroLabels = FALSE,
  addGuide = TRUE,
  hang = 0.03,
  guideHang = 0.05
)

# module trait associations
# create traits file - binarize categorical variables

traits <- colData %>% 
  mutate(disease_state = ifelse(grepl('Disease', diagnosis), 1, 0)) %>%
  select(5)

# Define numbers of genes and samples
nSamples <- nrow(norm.count)
nGenes <- ncol(norm.count)


module.trait.corr <- cor(module_eigengenes, traits, use = 'p')
module.trait.corr.pvals <- corPvalueStudent(module.trait.corr, nSamples)

# visualize module-trait association as a heatmap

heatmap.data <- merge(module_eigengenes, traits, by = 'row.names')
head(heatmap.data)

heatmap.data <- heatmap.data %>% 
  column_to_rownames(var = 'Row.names')

# Your existing code for CorLevelPlot
CorLevelPlot(heatmap.data,
             x = names(heatmap.data)[37],
             y = names(heatmap.data)[1:36],
             col = c("blue1", "skyblue", "white", "pink", "red"),
             main = "Module Trait Relationships")
