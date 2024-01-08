# upload file
data <- read.csv("YourFile.csv")

# transposed
data <- t(data)

# convert into dataframe
data <- as.data.frame(data)


# Assuming your data columns are named 'GeneExpression' and 'Class'
gene_expression <- data$GeneExpression
class <- factor(data$Class)

#import library
library(pROC)

# Create ROC curve
roc_curve <- roc(class, gene_expression)

# Print the AUC (Area Under the Curve)
print(auc(roc_curve))

# Compute confidence intervals
ci_values <- ci(roc_curve)

# Print confidence intervals
print(ci_values)

# Plot the ROC curve
plot(roc_curve, main = "gene name", col = "red", lwd = 2)

# Add AUC to the plot
text(0.8, 0.2, paste("AUC =", round(auc(roc_curve), 2)), col = "black", cex = 1.2)
