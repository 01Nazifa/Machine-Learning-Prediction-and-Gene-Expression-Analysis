import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your count matrix data
file_path = "/content/lncRNA_ML_transposed.csv"
data = pd.read_csv(file_path)
data = data.drop('Sample', axis=1)

# Assume 'X' contains your features, and 'y' contains your target variable 'Class'
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of StandardScaler and fit on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing and cross-validation data
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Define models
models = {
    'RF': RandomForestClassifier(n_estimators=100, max_features=None, random_state=42),
    'LR': LogisticRegression(max_iter=500, solver='lbfgs'),
    'SVM': SVC(probability=True, random_state=42, C=5, kernel='rbf', gamma=0.001, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=500),
    'KNN': KNeighborsClassifier(n_neighbors=3, p=1, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian Process': GaussianProcessClassifier(kernel=1.0 * RBF(), optimizer="fmin_l_bfgs_b", n_restarts_optimizer=9, max_iter_predict=500, random_state=42, multi_class="one_vs_rest"),
    'Naive Bayes': GaussianNB(priors=None)
}

# Number of bootstrap samples for CI calculation
n_bootstrap_samples = 500

# Initialize dictionaries to store AUC and CI values
auc_values = {}
ci_low_values = {}
ci_high_values = {}

# Initialize dictionaries to store additional metrics
accuracy_values = {}
f1_values = {}
precision_values = {}
recall_values = {}

# Initialize dictionaries to store cross-validation results
cv_auc_values = {}
cv_accuracy_values = {}
cv_f1_values = {}
cv_precision_values = {}
cv_recall_values = {}

# Function to calculate AUC and CI using bootstrap
def calculate_auc_ci(model, X_train, y_train, X_test, y_test, n_bootstrap_samples):
    auc_scores = []
    for _ in range(n_bootstrap_samples):
        # Bootstrap resample
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=np.random.randint(1, 1000))
        # Fit model on resampled data
        model.fit(X_boot, y_boot)
        # Predict probabilities on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        # Calculate AUC
        auc_scores.append(roc_auc_score(y_test, y_prob))

    # Calculate 95% CI
    ci_low, ci_high = np.percentile(auc_scores, [2.5, 97.5])
    return np.mean(auc_scores), ci_low, ci_high

# Function to calculate additional metrics using bootstrap
def calculate_additional_metrics(model, X_train, y_train, X_test, y_test, n_bootstrap_samples):
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for _ in range(n_bootstrap_samples):
        # Bootstrap resample
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=np.random.randint(1, 1000))
        # Fit model on resampled data
        model.fit(X_boot, y_boot)
        # Predict probabilities on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
        y_pred = (y_prob >= 0.5).astype(int)

        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=1))

    # Calculate 95% CI for each metric
    def calculate_ci(metric_scores):
        ci_low, ci_high = np.percentile(metric_scores, [2.5, 97.5])
        return ci_low, ci_high

    return (
        np.mean(accuracy_scores), *calculate_ci(accuracy_scores),
        np.mean(f1_scores), *calculate_ci(f1_scores),
        np.mean(precision_scores), *calculate_ci(precision_scores),
        np.mean(recall_scores), *calculate_ci(recall_scores)
    )

# Function to calculate cross-validation metrics
def calculate_cross_val_metrics(model, X, y, n_bootstrap_samples):
    cv_auc_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    cv_accuracy_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    cv_f1_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring='f1')
    cv_precision_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring='precision')
    cv_recall_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring='recall')

    def calculate_ci(metric_scores):
        ci_low, ci_high = np.percentile(metric_scores, [2.5, 97.5])
        return ci_low, ci_high

    return (
        np.mean(cv_auc_scores), *calculate_ci(cv_auc_scores),
        np.mean(cv_accuracy_scores), *calculate_ci(cv_accuracy_scores),
        np.mean(cv_f1_scores), *calculate_ci(cv_f1_scores),
        np.mean(cv_precision_scores), *calculate_ci(cv_precision_scores),
        np.mean(cv_recall_scores), *calculate_ci(cv_recall_scores)
    )

# Iterate through models for AUC
for model_name, model in models.items():
    auc_value, ci_low, ci_high = calculate_auc_ci(model, X_train_scaled, y_train, X_test_scaled, y_test, n_bootstrap_samples)
    auc_values[model_name] = auc_value
    ci_low_values[model_name] = ci_low
    ci_high_values[model_name] = ci_high

# Iterate through models for additional metrics
for model_name, model in models.items():
    acc_mean, acc_ci_low, acc_ci_high, \
    f1_mean, f1_ci_low, f1_ci_high, \
    precision_mean, precision_ci_low, precision_ci_high, \
    recall_mean, recall_ci_low, recall_ci_high = calculate_additional_metrics(model, X_train_scaled, y_train, X_test_scaled, y_test, n_bootstrap_samples)

    accuracy_values[model_name] = (acc_mean, acc_ci_low, acc_ci_high)
    f1_values[model_name] = (f1_mean, f1_ci_low, f1_ci_high)
    precision_values[model_name] = (precision_mean, precision_ci_low, precision_ci_high)
    recall_values[model_name] = (recall_mean, recall_ci_low, recall_ci_high)

# Iterate through models for cross-validation metrics
for model_name, model in models.items():
    cv_auc_mean, cv_auc_ci_low, cv_auc_ci_high, \
    cv_acc_mean, cv_acc_ci_low, cv_acc_ci_high, \
    cv_f1_mean, cv_f1_ci_low, cv_f1_ci_high, \
    cv_precision_mean, cv_precision_ci_low, cv_precision_ci_high, \
    cv_recall_mean, cv_recall_ci_low, cv_recall_ci_high = calculate_cross_val_metrics(model, X_scaled, y, n_bootstrap_samples)

    cv_auc_values[model_name] = (cv_auc_mean, cv_auc_ci_low, cv_auc_ci_high)
    cv_accuracy_values[model_name] = (cv_acc_mean, cv_acc_ci_low, cv_acc_ci_high)
    cv_f1_values[model_name] = (cv_f1_mean, cv_f1_ci_low, cv_f1_ci_high)
    cv_precision_values[model_name] = (cv_precision_mean, cv_precision_ci_low, cv_precision_ci_high)
    cv_recall_values[model_name] = (cv_recall_mean, cv_recall_ci_low, cv_recall_ci_high)

# Print AUC and CI values
for model_name in models.keys():
    print(f"{model_name}: AUC = {auc_values[model_name]:.3f}, 95% CI = [{ci_low_values[model_name]:.3f}, {ci_high_values[model_name]:.3f}]")
    print(f"{model_name}: "
          f"Accuracy = {accuracy_values[model_name][0]:.3f} [{accuracy_values[model_name][1]:.3f}, {accuracy_values[model_name][2]:.3f}], "
          f"F1 Score = {f1_values[model_name][0]:.3f} [{f1_values[model_name][1]:.3f}, {f1_values[model_name][2]:.3f}], "
          f"Precision = {precision_values[model_name][0]:.3f} [{precision_values[model_name][1]:.3f}, {precision_values[model_name][2]:.3f}], "
          f"Recall = {recall_values[model_name][0]:.3f} [{recall_values[model_name][1]:.3f}, {recall_values[model_name][2]:.3f}]")

# Print Cross-validation metrics
for model_name in models.keys():
    print(f"{model_name}: "
          f"CV AUC = {cv_auc_values[model_name][0]:.3f} [{cv_auc_values[model_name][1]:.3f}, {cv_auc_values[model_name][2]:.3f}], "
          f"CV Accuracy = {cv_accuracy_values[model_name][0]:.3f} [{cv_accuracy_values[model_name][1]:.3f}, {cv_accuracy_values[model_name][2]:.3f}], "
          f"CV F1 Score = {cv_f1_values[model_name][0]:.3f} [{cv_f1_values[model_name][1]:.3f}, {cv_f1_values[model_name][2]:.3f}], "
          f"CV Precision = {cv_precision_values[model_name][0]:.3f} [{cv_precision_values[model_name][1]:.3f}, {cv_precision_values[model_name][2]:.3f}], "
          f"CV Recall = {cv_recall_values[model_name][0]:.3f} [{cv_recall_values[model_name][1]:.3f}, {cv_recall_values[model_name][2]:.3f}]")
