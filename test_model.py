#!/usr/bin/env python3
"""
test_model.py

Loads the best_classifier.pkl, best_regression_limit.pkl, best_regression_interest.pkl
and evaluates them on a final hold-out set (20% of data).

Prints:
  - Classification metrics: accuracy, confusion matrix, classification report
  - Regression metrics (MAE, RMSE) for approved_amount and interest_rate

Usage:
    python test_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error
)

DATA_PATH = "data/final_labeled_dataset.csv"
CLASSIFIER_PATH = "models/best_classifier.pkl"
REG_LIMIT_PATH = "models/best_regression_limit.pkl"
REG_INT_PATH = "models/best_regression_interest.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["approved","approved_amount","interest_rate"], inplace=True)
    df = df[df["credit_score"].between(300,900)]
    df = df[df["annual_income"]>=0]
    return df

def main():
    for path in [CLASSIFIER_PATH, REG_LIMIT_PATH, REG_INT_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Please run train_model.py first.")
            return
    
    df = load_data()
    
    X = df.drop(columns=["applicant_id","approved","approved_amount","interest_rate"], errors="ignore")
    y_cls = df["approved"]
    y_amt = df["approved_amount"]
    y_int = df["interest_rate"]
    
    # 20% hold-out
    X_train, X_test, y_cls_train, y_cls_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=123
    )
    y_amt_train, y_amt_test = train_test_split(
        y_amt, test_size=0.2, random_state=123
    )
    y_int_train, y_int_test = train_test_split(
        y_int, test_size=0.2, random_state=123
    )
    
    # Load saved models
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(REG_LIMIT_PATH, "rb") as f:
        reg_amt = pickle.load(f)
    with open(REG_INT_PATH, "rb") as f:
        reg_int = pickle.load(f)
    
    # Evaluate classification
    y_pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_cls_test, y_pred_cls)
    cm = confusion_matrix(y_cls_test, y_pred_cls)
    report = classification_report(y_cls_test, y_pred_cls)
    
    print("=== Final Test Results: Classification (approved) ===")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    # Evaluate regression for approved_amount
    y_pred_amt = reg_amt.predict(X_test)
    mae_amt = mean_absolute_error(y_amt_test, y_pred_amt)
    rmse_amt = np.sqrt(np.mean((y_amt_test - y_pred_amt)**2))
    
    print("\n=== Final Test Results: 'approved_amount' ===")
    print(f"MAE:  {mae_amt:.2f}")
    print(f"RMSE: {rmse_amt:.2f}")
    
    # Evaluate regression for interest_rate
    y_pred_int = reg_int.predict(X_test)
    mae_int = mean_absolute_error(y_int_test, y_pred_int)
    rmse_int = np.sqrt(np.mean((y_int_test - y_pred_int)**2))
    
    print("\n=== Final Test Results: 'interest_rate' ===")
    print(f"MAE:  {mae_int:.2f}")
    print(f"RMSE: {rmse_int:.2f}")

if __name__ == "__main__":
    main()
