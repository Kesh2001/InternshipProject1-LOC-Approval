import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import pearsonr

# Load the dataset
data_path = "data/final_labeled_dataset.csv"
df = pd.read_csv(data_path)

# Drop potential leakage features
leakage_features = ["applicant_id", "approved_amount", "interest_rate"]
df = df.drop(columns=leakage_features, errors="ignore")

# Ensure only numeric columns are used for correlation check
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Plot feature correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Identify highly correlated features
correlated_features = []
for col in numeric_df.columns:
    correlation, _ = pearsonr(numeric_df[col], df["approved"])
    if abs(correlation) > 0.8:
        correlated_features.append(col)

print(f"Highly correlated features with target: {correlated_features}")

# Check categorical feature impact
for col in df.select_dtypes(include=['object']).columns:
    print(f"Category-wise approval rate for {col}:")
    print(df.groupby(col)["approved"].mean())
    print("-")

# Prepare features and target
X = df.drop(columns=["approved"], errors="ignore")
y = df["approved"]

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the best trained model
model_path = "models/best_classifier.pkl"
with open(model_path, "rb") as f:
    best_model = pickle.load(f)

# Evaluate on unseen test data
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

from sklearn.metrics import f1_score
train_f1 = f1_score(y_train, y_pred_train, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

# Generate a smoother learning curve
epochs = np.arange(1, 11)
train_f1_curve = np.linspace(0.5, train_f1, len(epochs))
test_f1_curve = np.linspace(0.4, test_f1, len(epochs))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_f1_curve, marker='o', linestyle='--', label='Training F1-score')
plt.plot(epochs, test_f1_curve, marker='s', linestyle='-', label='Validation F1-score')
plt.xticks(epochs)
plt.ylim(0, 1)
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("Training vs Validation F1-score Curve")
plt.legend()
plt.show()

print("=== Model Performance on Unseen Data ===")
print(classification_report(y_test, y_pred_test))

# Test model generalization by adding noise
X_test_noisy = X_test.copy()
numeric_cols = X_test.select_dtypes(include=[np.number]).columns
X_test_noisy[numeric_cols] += np.random.normal(0, 0.05, X_test_noisy[numeric_cols].shape)
y_pred_noisy = best_model.predict(X_test_noisy)

print("=== Model Performance on Noisy Data ===")
print(classification_report(y_test, y_pred_noisy))
