#!/usr/bin/env python3

"""
validate_model.py

Generates learning-curve plots for each of the three models:
 1) Classifier (approval)
 2) Regressor (approved_amount)
 3) Regressor (interest_rate)

Requires:
 - data/final_labeled_dataset.csv
 - models/best_classifier.pkl
 - models/best_regression_limit.pkl
 - models/best_regression_interest.pkl

Produces :
 - One chart for classification learning curve
 - One chart for limit regression learning curve
 - One chart for interest regression learning curve

Usage:
  1) Run 'dataset.py' and 'train_model.py' first to generate data and models
  2) In Kaggle, copy this entire code cell and run
  3) Inspect or screenshot the displayed plots for over/underfitting analysis

"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, mean_absolute_error, f1_score


DATA_PATH = "data/final_labeled_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Basic cleaning, matching train_model
df.dropna(subset=["approved", "approved_amount", "interest_rate",
                  "annual_income", "credit_score", "payment_history"],
          inplace=True)
df = df[df["credit_score"].between(300,900)]
df = df[df["annual_income"]>=0]

# Splitting out the 3 tasks
X_all = df.drop(columns=["applicant_id", "approved", "approved_amount", "interest_rate"], errors="ignore")
y_approve = df["approved"]
y_limit = df["approved_amount"]
y_interest = df["interest_rate"]



MODELS_DIR = "models"
clf_path = os.path.join(MODELS_DIR, "best_classifier.pkl")
lim_path = os.path.join(MODELS_DIR, "best_regression_limit.pkl")
int_path = os.path.join(MODELS_DIR, "best_regression_interest.pkl")

with open(clf_path, "rb") as f:
    model_clf = pickle.load(f)
with open(lim_path, "rb") as f:
    model_lim = pickle.load(f)
with open(int_path, "rb") as f:
    model_int = pickle.load(f)



def plot_learning_curve(estimator, X, y, task_name="Classification", scoring=None):
    """
    Plots the train/validation learning curve for the given estimator using
    scikit-learn's learning_curve function.
    
    - scoring: e.g. "accuracy", "neg_mean_absolute_error", etc.
    - Each chart is displayed in its own figure.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5,                \
        scoring=scoring,       
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42,
        n_jobs=-1
    )
    
    # For "neg_mean_absolute_error", bigger is better, so let's re-scale to positive for plotting
    if scoring == "neg_mean_absolute_error":
        train_scores = -train_scores
        val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)
    
    # Plot in a separate figure
    plt.figure()
    plt.plot(train_sizes, train_mean, marker='o', label="Training Score")
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.1)
    plt.plot(train_sizes, val_mean, marker='s', label="Validation Score")
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.1)
    
    plt.legend()
    plt.title(f"Learning Curve: {task_name}")
    plt.xlabel("Training Set Size")
    if scoring in ["accuracy", "f1_weighted"]:
        plt.ylabel("Score")
    elif scoring == "neg_mean_absolute_error":
        plt.ylabel("MAE (lower is better)")
    else:
        plt.ylabel("Score")
    
    plt.tight_layout()
    plt.show()


# 4a) Classification: "accuracy" or "f1_weighted"
plot_learning_curve(
    estimator=model_clf,
    X=X_all,
    y=y_approve,
    task_name="Approval (Classifier)",
    scoring="accuracy"  
)

# 4b) Limit regression: use neg_mean_absolute_error
plot_learning_curve(
    estimator=model_lim,
    X=X_all,
    y=y_limit,
    task_name="Credit Limit (Regressor)",
    scoring="neg_mean_absolute_error"
)

# 4c) Interest regression: also neg_mean_absolute_error
plot_learning_curve(
    estimator=model_int,
    X=X_all,
    y=y_interest,
    task_name="Interest Rate (Regressor)",
    scoring="neg_mean_absolute_error"
)

print("\nLearning curve plots displayed.\n")
