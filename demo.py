#!/usr/bin/env python3
"""
demo.py

A simple CLI demo:
 - Loads your trained models (classifier, limit regressor, interest regressor).
 - Asks the user for inputs (credit_score, income, etc.) in the console.
 - Predicts approval status, approved amount, and interest rate.
 - Prints the final decision.

Usage:
    python demo.py
"""

import pickle
import pandas as pd
import numpy as np

def main():
    # 1) Load the trained models (adjust filenames if needed)
    #    e.g. best_classifier.pkl, best_regression_limit.pkl, best_regression_interest.pkl
    with open("models/best_classifier.pkl", "rb") as f:
        model_approval = pickle.load(f)
    with open("models/best_regression_limit.pkl", "rb") as f:
        model_limit = pickle.load(f)
    with open("models/best_regression_interest.pkl", "rb") as f:
        model_interest = pickle.load(f)
    
    print("=== LOC Auto-Approval Demo ===")
    print("Enter the following details to see if you're approved.\n")
    
    # 2) Prompt user for inputs
    #    For simplicity, we do minimal input validation here
    #    Make sure to match the same columns used in training!
    
    # Numeric inputs
    annual_income = float(input("Annual Income (e.g. 65000): "))
    self_reported_debt = float(input("Self-reported Monthly Debt (e.g. 1500): "))
    self_reported_expenses = float(input("Self-reported Monthly Expenses (e.g. 2000): "))
    requested_amount = float(input("Requested Amount (1k - 50k, e.g. 12000): "))
    age = int(input("Age (19-100): "))
    months_employed = int(input("Months Employed (0-600): "))
    credit_score = float(input("Credit Score (300-900): "))
    credit_utilization = float(input("Credit Utilization (%) (0-100): "))
    num_open_accounts = int(input("Number of Open Accounts (0-20): "))
    num_credit_inquiries = int(input("Number of Credit Inquiries (past 12 mo) (0-10): "))
    
    # Categorical inputs
    province = input("Province (ON or BC): ")
    employment_status = input("Employment Status (Full-time, Part-time, Unemployed): ")
    payment_history = input("Payment History (On Time, Late <30, Last 30-60, Late>60): ")
    
    # 3) We also need 'estimated_debt' and 'DTI' if your model expects them
    #    If your final model pipeline calculates them, you might skip
    #    But let's assume we need them as direct inputs for now:
    #    Usually we do not prompt for these because they're derived fields,
    #    but I'll show an example anyway:
    try:
        estimated_debt = float(input("Estimated Debt (0-10000, if known, else 0): "))
    except:
        estimated_debt = 0.0
    try:
        DTI = float(input("DTI ratio (0-1 range, e.g. 0.35) or 0 if unknown: "))
    except:
        DTI = 0.0
    
    # 4) Build a single-row DataFrame in the exact format your model expects
    #    Make sure the column names match your training pipeline
    data_dict = {
        "self_reported_expenses": [self_reported_expenses],
        "annual_income": [annual_income],
        "self_reported_debt": [self_reported_debt],
        "requested_amount": [requested_amount],
        "age": [age],
        "months_employed": [months_employed],
        "credit_score": [credit_score],
        "credit_utilization": [credit_utilization],
        "num_open_accounts": [num_open_accounts],
        "num_credit_inquiries": [num_credit_inquiries],
        "estimated_debt": [estimated_debt],
        "DTI": [DTI],
        "province": [province],
        "employment_status": [employment_status],
        "payment_history": [payment_history]
    }
    X_user = pd.DataFrame(data_dict)
    
    # 5) Predict approval
    approval_pred = model_approval.predict(X_user)[0]  # 0 or 1
    
    if approval_pred == 0:
        print("\n=== Decision ===")
        print("**Denied**. Sorry, you do not meet our criteria.")
    else:
        # If approved, predict limit + interest
        limit_pred = model_limit.predict(X_user)[0]
        interest_pred = model_interest.predict(X_user)[0]
        
        # Round or format as you wish
        print("\n=== Decision ===")
        print(f"**Approved** for ${limit_pred:,.2f} at {interest_pred:.2f}% interest.")
    
    print("\nThank you for trying the LOC Auto-Approval Demo!")

if __name__ == "__main__":
    main()
