#!/usr/bin/env python3
"""
train_model.py

Integrates:
 - Removing 'approved_amount' & 'interest_rate' to avoid label leakage
 - Shuffling cross-validation using KFold(shuffle=True)
 - Adding small noise to numeric features
 - Training multiple classifiers & picking the best by Weighted F1
 - Saving best classifier

Usage:
    python train_model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, f1_score

DATA_PATH = "data/final_labeled_dataset.csv"
MODELS_DIR = "models"

def load_and_clean_data(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    # Remove rows missing these crucial columns
    df.dropna(subset=["approved", "annual_income", "credit_score", "payment_history"], inplace=True)
    
    # Filter out weird credit scores or negative income
    df = df[df["credit_score"].between(300, 900)]
    df = df[df["annual_income"] >= 0]
    
    return df

def build_preprocessor():
    """
    Create a ColumnTransformer that:
     - Standardizes numeric columns
     - OneHotEncodes categorical columns
    """
    numeric_features = [
        "self_reported_expenses",
        "annual_income",
        "self_reported_debt",
        "requested_amount",
        "age",
        "months_employed",
        "credit_score",
        "credit_utilization",
        "num_open_accounts",
        "num_credit_inquiries",
        "estimated_debt",
        "DTI"
    ]
    categorical_features = ["province", "employment_status"]
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor

def train_classification_model(X, y):
    """
    Trains multiple classification models on X,y.
    Picks best by Weighted F1 in GridSearchCV with KFold(shuffle=True).
    Returns the best pipeline (estimator).
    """
    # 1) Build pipelines
    preprocessor = build_preprocessor()
    
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    pipe_xgb = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])
    
    # 2) Define parameter grids
    param_lr = {
        "clf__C": [0.1, 1, 5],
        "clf__solver": ["lbfgs", "liblinear"]
    }
    param_rf = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [5, 10, 20]
    }
    param_xgb = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, 6, 10]
    }
    
    # Weighted F1
    f1_scorer = make_scorer(f1_score, average="weighted")
    
    # 3) Create KFold with shuffle=True
    cv_split = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4) GridSearchCV for each pipeline, using the same cv=KFold(...)
    gs_lr = GridSearchCV(
        pipe_lr, param_lr, scoring=f1_scorer,
        cv=cv_split, n_jobs=-1, return_train_score=True
    )
    gs_rf = GridSearchCV(
        pipe_rf, param_rf, scoring=f1_scorer,
        cv=cv_split, n_jobs=-1, return_train_score=True
    )
    gs_xgb = GridSearchCV(
        pipe_xgb, param_xgb, scoring=f1_scorer,
        cv=cv_split, n_jobs=-1, return_train_score=True
    )
    
    # Fit
    gs_lr.fit(X, y)
    gs_rf.fit(X, y)
    gs_xgb.fit(X, y)
    
    # Compare best cross-val Weighted F1
    results = [
        ("LogisticRegression", gs_lr.best_score_, gs_lr.best_estimator_),
        ("RandomForest",       gs_rf.best_score_, gs_rf.best_estimator_),
        ("XGBoost",            gs_xgb.best_score_, gs_xgb.best_estimator_),
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n=== Classification Model Results (Weighted F1) ===")
    for name, score, _ in results:
        print(f"{name}: {score:.4f}")
    
    best_name, best_score, best_estimator = results[0]
    print(f"\nBest classifier: {best_name} with Weighted F1={best_score:.4f}\n")
    
    return best_estimator

def main():
    # Ensure we have a models directory
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # 1) Load data
    df = load_and_clean_data(DATA_PATH)
    
    # 2) Remove label columns to avoid leakage (approved_amount, interest_rate)
    #    We keep just the features for X, and y=approved
    X = df.drop(columns=["applicant_id", "approved", "approved_amount", "interest_rate"], errors="ignore")
    y = df["approved"]
    
    # 3) Add small noise to numeric features
    #    so the model doesn't memorize exact feature values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] += np.random.normal(0, 0.01, X[numeric_cols].shape)
    
    # 4) Train classification
    print("=== Training Classification (Approved) ===")
    best_classifier = train_classification_model(X, y)
    
    # 5) Save the best classifier
    with open(os.path.join(MODELS_DIR, "best_classifier.pkl"), "wb") as f:
        pickle.dump(best_classifier, f)
    print("Saved best classifier to 'models/best_classifier.pkl'")

if __name__ == "__main__":
    main()
