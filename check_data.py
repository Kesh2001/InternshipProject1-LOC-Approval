#!/usr/bin/env python3
"""
check_data.py

Verifies that data/final_labeled_dataset.csv was generated correctly:
  - Checks shape
  - Ensures no large chunk of missing data in critical columns
  - Prints basic stats (approval rate, etc.)

Run:
    python check_data.py
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = "data/final_labeled_dataset.csv"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Please run dataset.py first or verify the path.")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}")

    # Check for NaNs in critical columns
    critical_cols = [
        "annual_income", "credit_score", "approved",
        "approved_amount", "interest_rate"
    ]
    missing = df[critical_cols].isnull().sum()
    print("\nMissing values in critical columns:")
    print(missing)
    
    # If you want a quick threshold check, e.g. no more than 2% missing:
    total_rows = len(df)
    for c in critical_cols:
        pct_missing = (missing[c] / total_rows) * 100
        if pct_missing > 2.0:
            print(f"WARNING: Column '{c}' has >2% missing data ({pct_missing:.2f}%).")
    
    # Check approval rate
    if "approved" in df.columns:
        approval_rate = df["approved"].mean() * 100
        print(f"\nApproval Rate: {approval_rate:.1f}%")
        if approval_rate < 60 or approval_rate > 70:
            print("WARNING: Approval rate is outside the ~60–70% range.")
    else:
        print("WARNING: 'approved' column not found.")
    
    # Basic stats
    print("\nBasic summary of numeric columns:")
    print(df.describe())

    print("\nData check complete.")

if __name__ == "__main__":
    main()
