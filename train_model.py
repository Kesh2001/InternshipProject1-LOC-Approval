#!/usr/bin/env python3
"""
train_model.py

An updated training script for a Line of Credit Auto-Approval application.
This version improves the data preprocessing by:
 - Dropping rows only with missing target values (approved, approved_amount, interest_rate)
 - Retaining noisy feature data and handling missing values via imputation
 - Using a preprocessing pipeline that imputes numeric (median) and categorical (constant) values before scaling/encoding
 - Reducing overfitting by constraining hyperparameters for RandomForest and XGBoost models
 - Using 5-fold cross-validation to select the best model among LogisticRegression, RandomForest, and XGBoost for classification,
   and among LinearRegression, RandomForestRegressor, and XGBRegressor for regression tasks.

Outputs three models:
 - models/best_classifier.pkl
 - models/best_regression_limit.pkl
 - models/best_regression_interest.pkl

Usage:
   1) Generate data (using your synthetic data generation process)
   2) python train_model.py
   3) Check the holdout performance metrics and models saved in the 'models/' directory.
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, mean_absolute_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Additional import for imputation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DATA_PATH = "data/final_labeled_dataset.csv"
MODELS_DIR = "models"

def load_and_clean_data(path=DATA_PATH):
    """
    Loads data from CSV and cleans it.
    
    Instead of removing rows with missing features, we drop rows only if target columns
    (approved, approved_amount, interest_rate) are missing. This preserves realistic noise.
    
    Additionally, it filters out rows with invalid credit scores or negative incomes.
    """
    df = pd.read_csv(path)
    
    # Drop rows only if target values are missing
    df.dropna(subset=["approved", "approved_amount", "interest_rate"], inplace=True)
    
    # Filter out rows with invalid credit scores (should be between 300 and 900)
    df = df[df["credit_score"].between(300, 900)]
    
    # Filter out rows with negative annual_income values
    df = df[df["annual_income"] >= 0]
    return df

# Define the columns for preprocessing
NUMERIC_COLS = [
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
CAT_COLS = [
    "province",
    "employment_status",
    "payment_history"
]

def build_preprocessor():
    """
    Builds a preprocessing pipeline that:
     - For numeric features: imputes missing values using the median, then standardizes.
     - For categorical features: imputes missing values with a constant ('missing'), then applies one-hot encoding.
    """
    numeric_transformer = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
         ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ])

def grid_search_classification(X, y):
    """
    Performs grid search for classification models using a pipeline that includes preprocessing.
    Three classifiers are evaluated: LogisticRegression, RandomForestClassifier, and XGBClassifier.
    """
    pre = build_preprocessor()
    
    # Constrained RandomForestClassifier to reduce overfitting
    rf_clf = RandomForestClassifier(
        random_state=42,
        n_estimators=50,
        max_depth=8,
        min_samples_split=20
    )
    # Constrained XGBClassifier
    xgb_clf = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
    
    # Pipelines for each classifier
    pipe_lr = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe_rf = Pipeline([
        ("pre", pre),
        ("clf", rf_clf)
    ])
    pipe_xgb = Pipeline([
        ("pre", pre),
        ("clf", xgb_clf)
    ])
    
    # Parameter grids for grid search
    param_lr = {
        "clf__C": [0.1, 1, 5],
        "clf__solver": ["lbfgs", "liblinear"]
    }
    param_rf = {
        "clf__max_depth": [5, 10],
        "clf__min_samples_split": [10, 20]
    }
    param_xgb = {
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.05, 0.1]
    }
    
    from sklearn.metrics import f1_score
    f1_scorer = make_scorer(f1_score, average="weighted")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    gs_lr = GridSearchCV(pipe_lr, param_lr, scoring=f1_scorer, cv=cv, n_jobs=-1)
    gs_rf = GridSearchCV(pipe_rf, param_rf, scoring=f1_scorer, cv=cv, n_jobs=-1)
    gs_xgb = GridSearchCV(pipe_xgb, param_xgb, scoring=f1_scorer, cv=cv, n_jobs=-1)
    
    gs_lr.fit(X, y)
    gs_rf.fit(X, y)
    gs_xgb.fit(X, y)
    
    results = [
        ("LogisticRegression", gs_lr.best_estimator_, gs_lr.best_score_),
        ("RandomForest",       gs_rf.best_estimator_, gs_rf.best_score_),
        ("XGBoost",            gs_xgb.best_estimator_, gs_xgb.best_score_)
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    
    best_name, best_model, best_score = results[0]
    print(f"Best Classifier => {best_name}, Weighted F1={best_score:.4f}")
    return best_model

def grid_search_regression(X, y, label="limit"):
    """
    Performs grid search for regression models using a pipeline that includes preprocessing.
    Three regressors are evaluated: LinearRegression, RandomForestRegressor, and XGBRegressor.
    """
    pre = build_preprocessor()
    
    # Constrained RandomForestRegressor to reduce overfitting
    rf_reg = RandomForestRegressor(
        random_state=42,
        n_estimators=50,
        max_depth=7,
        min_samples_split=30
    )
    # Constrained XGBRegressor
    xgb_reg = XGBRegressor(
        random_state=42,
        n_estimators=50,
        max_depth=2,
        learning_rate=0.08
    )
    
    # Pipelines for each regressor
    pipe_lin = Pipeline([
        ("pre", pre),
        ("reg", LinearRegression())
    ])
    pipe_rf = Pipeline([
        ("pre", pre),
        ("reg", rf_reg)
    ])
    pipe_xgb = Pipeline([
        ("pre", pre),
        ("reg", xgb_reg)
    ])
    
    # Parameter grids for grid search (empty grid for LinearRegression)
    param_lin = {}
    param_rf = {
        "reg__max_depth": [5, 10],
        "reg__min_samples_split": [10, 20]
    }
    param_xgb = {
        "reg__max_depth": [2, 3],
        "reg__learning_rate": [0.05, 0.1]
    }
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    gs_lin = GridSearchCV(pipe_lin, param_lin, scoring=mae_scorer, cv=cv, n_jobs=-1)
    gs_rf = GridSearchCV(pipe_rf, param_rf, scoring=mae_scorer, cv=cv, n_jobs=-1)
    gs_xgb = GridSearchCV(pipe_xgb, param_xgb, scoring=mae_scorer, cv=cv, n_jobs=-1)
    
    gs_lin.fit(X, y)
    gs_rf.fit(X, y)
    gs_xgb.fit(X, y)
    
    results = [
        ("LinearRegression", gs_lin.best_estimator_, gs_lin.best_score_),
        ("RandomForestReg",  gs_rf.best_estimator_, gs_rf.best_score_),
        ("XGBRegressor",     gs_xgb.best_estimator_, gs_xgb.best_score_)
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    
    best_name, best_model, best_score = results[0]
    print(f"Best Regressor for {label} => {best_name}, CV (Neg MAE)={best_score:.4f}")
    return best_model

def main():
    # Ensure that the models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    df = load_and_clean_data(DATA_PATH)
    if df.empty:
        print("No valid data after cleaning.")
        return
    
    # Feature set: drop columns not used as features
    drop_cols = ["applicant_id", "approved", "approved_amount", "interest_rate"]
    X_all = df.drop(columns=drop_cols, errors="ignore")
    
    # Targets
    y_approve = df["approved"]
    y_limit = df["approved_amount"]
    y_interest = df["interest_rate"]
    
    # NOTE: We remove the extra noise addition step because the synthetic data generation
    # already includes realistic noise and missing values.
    
    # 80-20 train-test split with fixed random state for reproducibility
    X_train, X_test, yA_train, yA_test, yL_train, yL_test, yI_train, yI_test = train_test_split(
        X_all, y_approve, y_limit, y_interest,
        test_size=0.2, random_state=42
    )
    
    # 1) Classification
    print("=== Classification (approved) ===")
    best_clf = grid_search_classification(X_train, yA_train)
    yA_pred = best_clf.predict(X_test)
    acc = accuracy_score(yA_test, yA_pred)
    rec_approve = recall_score(yA_test, yA_pred, pos_label=1)
    print(f"Holdout Accuracy: {acc:.3f}, Recall(1)={rec_approve:.3f}")
    
    with open(os.path.join(MODELS_DIR, "best_classifier.pkl"), "wb") as f:
        pickle.dump(best_clf, f)
    
    # 2) Regression: Credit Limit Prediction
    print("\n=== Regression (approved_amount) ===")
    best_limit = grid_search_regression(X_train, yL_train, label="Limit")
    yL_pred = best_limit.predict(X_test)
    mae_limit = mean_absolute_error(yL_test, yL_pred)
    print(f"Holdout MAE (Limit): ${mae_limit:,.2f}")
    with open(os.path.join(MODELS_DIR, "best_regression_limit.pkl"), "wb") as f:
        pickle.dump(best_limit, f)
    
    # 3) Regression: Interest Rate Prediction
    print("\n=== Regression (interest_rate) ===")
    best_interest = grid_search_regression(X_train, yI_train, label="Interest")
    yI_pred = best_interest.predict(X_test)
    mae_interest = mean_absolute_error(yI_test, yI_pred)
    print(f"Holdout MAE (Interest): {mae_interest:.3f}%")
    with open(os.path.join(MODELS_DIR, "best_regression_interest.pkl"), "wb") as f:
        pickle.dump(best_interest, f)
    
    print("\n=== Final Results on 20% Holdout ===")
    print(f"Approval Accuracy: {acc:.2%}, Recall(1): {rec_approve:.2%}")
    print(f"Credit Limit MAE: ${mae_limit:,.2f}")
    print(f"Interest Rate MAE: {mae_interest:.3f}%")
    print("Models saved in 'models/'.")

if __name__ == "__main__":
    main()
