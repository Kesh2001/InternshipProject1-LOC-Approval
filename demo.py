#!/usr/bin/env python3
"""
demo.py

A simple CLI demo:
 - Loads your trained models (classifier, limit regressor, interest regressor).
 - Uses predefined test cases instead of user input.
 - Predicts approval status, approved amount, and interest rate.
 - Prints the final decision.

Usage:
    python demo.py
"""

import pickle
import pandas as pd
import os
from train_model import build_preprocessor  # Import preprocessor for consistency

# Load trained models
model_files = {
    "approval": "models/best_classifier.pkl",
    "limit": "models/best_regression_limit.pkl",
    "interest": "models/best_regression_interest.pkl"
}

# Ensure models exist before loading
for key, path in model_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping {key} prediction.")

# Load models if available
with open(model_files["approval"], "rb") as f:
    model_approval = pickle.load(f)

model_limit = None
model_interest = None
if os.path.exists(model_files["limit"]):
    with open(model_files["limit"], "rb") as f:
        model_limit = pickle.load(f)
if os.path.exists(model_files["interest"]):
    with open(model_files["interest"], "rb") as f:
        model_interest = pickle.load(f)

# Load preprocessor for consistent transformation
preprocessor = build_preprocessor()


# Predefined test cases
test_cases = [
    {
        "self_reported_expenses": 2000,
        "annual_income": 80000,
        "self_reported_debt": 2100,
        "requested_amount": 15000,
        "age": 30,
        "months_employed": 24,
        "credit_score": 700,
        "credit_utilization": 0.40,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "estimated_debt": 600, 
        "DTI": 0.35,  
        "province": "ON",
        "employment_status": "Full-time",
        "payment_history": "On Time"
    },
    {
        "self_reported_expenses": 3500,
        "annual_income": 40000,
        "self_reported_debt": 5000,
        "requested_amount": 10000,
        "age": 45,
        "months_employed": 6,
        "credit_score": 650,
        "credit_utilization": 0.60,
        "num_open_accounts": 5,
        "num_credit_inquiries": 3,
        "estimated_debt": 900,
        "DTI": 0.50,
        "province": "BC",
        "employment_status": "Part-time",
        "payment_history": "Late <30"
    },
    {
        "self_reported_expenses": 1800,
        "annual_income": 120000,
        "self_reported_debt": 1500,
        "requested_amount": 25000,
        "age": 35,
        "months_employed": 36,
        "credit_score": 750,
        "credit_utilization": 0.30,
        "num_open_accounts": 2,
        "num_credit_inquiries": 0,
        "estimated_debt": 450,
        "DTI": 0.28,
        "province": "ON",
        "employment_status": "Full-time",
        "payment_history": "On Time"
    }
]

# Loop through test cases and make predictions
for case in test_cases:
    X_user = pd.DataFrame([case])

    # Apply same preprocessing as training
    X_user_transformed = preprocessor.transform(X_user)

    # Predict approval
    approval_pred = model_approval.predict(X_user_transformed)[0]
    
    print(f"\n=== Decision for Test Case (Credit Score: {case['credit_score']}, Income: ${case['annual_income']:,}) ===")
    
    if approval_pred == 0:
        print("**Denied**. Sorry, you do not meet our criteria.")
    else:
        print("**Approved**", end="")
        
        if model_limit and model_interest:
            limit_pred = model_limit.predict(X_user_transformed)[0]
            interest_pred = model_interest.predict(X_user_transformed)[0]
            print(f" for ${limit_pred:,.2f} at {interest_pred:.2f}% interest.")
        else:
            print(" (Credit limit & interest prediction unavailable).")

print("\n=== Demo Completed ===")
