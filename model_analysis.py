#!/usr/bin/env python3
"""
model_analysis.py

Loads the saved classifier (best_classifier.pkl) and does:
  - 5-fold cross-validation
  - Plots train vs. validation F1
  - Prints average F1, precision, recall (weighted)

Usage:
    python model_analysis.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_PATH = "data/final_labeled_dataset.csv"
MODEL_PATH = "models/best_classifier.pkl"

def load_data(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["approved"], inplace=True)
    df = df[df["credit_score"].between(300,900)]
    df = df[df["annual_income"]>=0]
    return df

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run train_model.py first.")
        return
    
    df = load_data(DATA_PATH)
    
    X = df.drop(columns=["applicant_id","approved","approved_amount","interest_rate"], errors="ignore")
    y = df["approved"].values
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_f1_scores = []
    val_f1_scores = []
    val_prec_scores = []
    val_rec_scores = []
    
    fold_i = 1
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Re-fit on each fold
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val   = model.predict(X_val)
        
        f1_train = f1_score(y_train, y_pred_train, average="weighted")
        f1_val   = f1_score(y_val, y_pred_val, average="weighted")
        
        train_f1_scores.append(f1_train)
        val_f1_scores.append(f1_val)
        
        val_prec = precision_score(y_val, y_pred_val, average="weighted")
        val_rec  = recall_score(y_val, y_pred_val, average="weighted")
        val_prec_scores.append(val_prec)
        val_rec_scores.append(val_rec)
        
        print(f"Fold {fold_i}: Train F1={f1_train:.4f}, Val F1={f1_val:.4f}")
        fold_i += 1
    
    print("\n=== 5-Fold Cross-Validation (F1) ===")
    print(f"Mean Train F1: {np.mean(train_f1_scores):.4f} ± {np.std(train_f1_scores):.4f}")
    print(f"Mean Val F1:   {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}")
    
    print("\n=== Validation Precision & Recall (weighted) ===")
    print(f"Precision: {np.mean(val_prec_scores):.4f} ± {np.std(val_prec_scores):.4f}")
    print(f"Recall:    {np.mean(val_rec_scores):.4f} ± {np.std(val_rec_scores):.4f}")
    
    # Plot
    folds = np.arange(1, kf.n_splits+1)
    plt.figure(figsize=(8,5))
    plt.plot(folds, train_f1_scores, marker='o', label='Train F1')
    plt.plot(folds, val_f1_scores, marker='o', label='Validation F1')
    plt.title("Training vs. Validation F1 for 'approved' (Classification)")
    plt.xlabel("Fold")
    plt.ylabel("F1 Score")
    plt.ylim([0.95, 1.0])  # Adjust if your scores are super high
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
