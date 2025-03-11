#!/usr/bin/env python3
"""
dataset.py

Generates the LOC synthetic dataset with ~60–70% approval rate using the bank's rules.
Outputs:
  - data/applicant_input.csv
  - data/third_party_credit.csv
  - data/final_labeled_dataset.csv

Includes ~5% outliers and ~2% missing data for realism, per the project guidelines.
"""

import os
import numpy as np
import pandas as pd

# Create data/ folder if not exists
os.makedirs("data", exist_ok=True)

NUM_ROWS = 5000
np.random.seed(42)

def generate_applicant_input(num_rows):
    applicant_ids = [f"APP{i:05d}" for i in range(1, num_rows + 1)]
    
    # annual_income: lognormal with μ=log(100000)-0.5, σ=0.3, clipped to [20k,200k]
    mu, sigma = np.log(100000) - 0.5, 0.3
    annual_income = np.random.lognormal(mean=mu, sigma=sigma, size=num_rows)
    annual_income = np.clip(annual_income, 20000, 200000)
    
    # self_reported_expenses: uniform(0,10000)
    self_reported_expenses = np.random.uniform(0, 10000, num_rows)
    
    # monthly_income
    monthly_income = annual_income / 12
    
    # self_reported_debt: 5–15% of monthly_income
    self_reported_debt = monthly_income * np.random.uniform(0.05, 0.15, num_rows)
    self_reported_debt = np.clip(self_reported_debt, 0, 10000)
    
    # requested_amount: uniform(1000,12000)
    requested_amount = np.random.uniform(1000, 12000, num_rows)
    
    # other fields
    age = np.random.randint(19, 101, num_rows)
    province = np.random.choice(["ON", "BC"], size=num_rows, p=[0.65, 0.35])
    employment_status = np.random.choice(
        ["Full-time", "Part-time", "Unemployed"],
        size=num_rows, p=[0.6, 0.3, 0.1]
    )
    months_employed = [
        0 if status == "Unemployed" else np.random.randint(1, 601)
        for status in employment_status
    ]
    
    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "self_reported_expenses": self_reported_expenses,
        "annual_income": annual_income,
        "self_reported_debt": self_reported_debt,
        "requested_amount": requested_amount,
        "age": age,
        "province": province,
        "employment_status": employment_status,
        "months_employed": months_employed
    })
    return df

def generate_credit_dataset(applicant_df):
    num_rows = len(applicant_df)
    applicant_ids = applicant_df["applicant_id"].tolist()
    
    # credit_score: Normal(mean=720, std=80), clipped [300,900]
    credit_score = np.random.normal(720, 80, num_rows)
    credit_score = np.clip(credit_score, 300, 900).astype(int)
    
    # total_credit_limit: 50–200% of annual_income, clipped to max 50,000
    annual_income = applicant_df["annual_income"].values
    factors = np.random.uniform(0.5, 2.0, num_rows)
    total_credit_limit = annual_income * factors
    total_credit_limit = np.clip(total_credit_limit, 0, 50000)
    
    # credit_utilization: Beta(3,7)*100 => ~30% average
    credit_utilization = np.random.beta(3, 7, num_rows) * 100
    
    num_open_accounts = np.random.randint(0, 21, num_rows)
    num_credit_inquiries = np.random.randint(0, 11, num_rows)
    
    # payment_history
    payment_history = np.random.choice(
        ["On Time", "Late <30", "Last 30-60", "Late>60"],
        size=num_rows, p=[0.85, 0.10, 0.03, 0.02]
    )
    
    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "credit_score": credit_score,
        "total_credit_limit": total_credit_limit,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries": num_credit_inquiries,
        "payment_history": payment_history
    })
    return df

def merge_and_label(app_df, credit_df):
    # Merge on applicant_id
    df = pd.merge(app_df, credit_df, on="applicant_id")
    
    # estimated_debt
    df["estimated_debt"] = df["total_credit_limit"] * (df["credit_utilization"]/100) * 0.03
    
    # DTI
    df["DTI"] = (
        df["self_reported_debt"] + df["estimated_debt"] + (df["requested_amount"]*0.03)
    ) / (df["annual_income"] / 12)
    
    import numpy as np
    # --- Compute Approved Amount ---
    base_limit = np.where(
        df["credit_score"] >= 660, 0.50 * df["annual_income"],
        np.where(df["credit_score"] >= 500, 0.25 * df["annual_income"], 0.10 * df["annual_income"])
    )
    dti_adj = np.where(
        df["DTI"] <= 0.30, 1.0,
        np.where(df["DTI"] <= 0.40, 0.75, 0.50)
    )
    limit_adj = base_limit * dti_adj
    
    # +10% if Full-time & months_employed>=12
    emp_bonus = np.where(
        (df["employment_status"]=="Full-time") & (df["months_employed"]>=12), 1.10, 1.0
    )
    limit_adj *= emp_bonus
    
    # -50% if "Late>60"
    pay_penalty = np.where(df["payment_history"]=="Late>60", 0.50, 1.0)
    limit_adj *= pay_penalty
    
    # -20% if credit_utilization>50%
    util_penalty = np.where(df["credit_utilization"]>50, 0.80, 1.0)
    limit_adj *= util_penalty
    
    # Score-based cap
    cap = np.where(
        df["credit_score"]>=750, 25000,
        np.where(df["credit_score"]>=660, 15000,
                 np.where(df["credit_score"]>=500, 10000, 5000))
    )
    df["approved_amount"] = np.minimum(limit_adj, cap)
    
    # Deny conditions
    df.loc[df["DTI"]>0.40, "approved_amount"] = 0
    df.loc[df["credit_score"]<500, "approved_amount"] = 0
    df.loc[df["credit_utilization"]>80, "approved_amount"] = 0
    
    # --- Interest Rate ---
    base_rate = 6.75
    rate = np.full(len(df), base_rate)
    rate = np.where(
        df["credit_score"]>=750, rate-1.0,
        np.where(
            df["credit_score"]<500, rate+4.0,
            np.where((df["credit_score"]>=500)&(df["credit_score"]<=659), rate+2.0, rate)
        )
    )
    # +1% if DTI>30%
    rate += np.where(df["DTI"]>0.30, 1.0, 0.0)
    # +2% if "Late>60"
    rate += np.where(df["payment_history"]=="Late>60", 2.0, 0.0)
    # +1% if num_open_accounts>5
    rate += np.where(df["num_open_accounts"]>5, 1.0, 0.0)
    
    # clip to [3,15]
    rate = np.clip(rate, 3.0, 15.0)
    
    # if approved_amount=0 => interest_rate=0
    df["interest_rate"] = np.where(df["approved_amount"]>0, rate, 0.0)
    
    # --- Determine Approval (0/1) ---
    cond_approve = (
        (df["credit_score"]>=660) &
        (df["DTI"]<=0.40) &
        (df["payment_history"]=="On Time")
    )
    df["approved"] = np.where(cond_approve, 1, 0)
    
    return df

def introduce_noise_for_realism(df: pd.DataFrame):
    """
    Adds ~5% outliers and ~2% missing data, following the doc:
      - 5% "irregular" values in annual_income
      - 2% missing credit_score
    """
    n = len(df)
    
    # 5% outliers in annual_income => multiply by ~[1.5..3.0]
    outlier_count = int(0.05 * n)
    outlier_rows = np.random.choice(df.index, size=outlier_count, replace=False)
    df.loc[outlier_rows, "annual_income"] *= np.random.uniform(1.5, 3.0, size=outlier_count)
    
    # 2% missing credit_score
    missing_count = int(0.02 * n)
    missing_rows = np.random.choice(df.index, size=missing_count, replace=False)
    df.loc[missing_rows, "credit_score"] = np.nan
    
    return df

def main():
    print("Generating Applicant Input dataset...")
    app_df = generate_applicant_input(NUM_ROWS)
    
    print("Generating Third-Party Credit dataset...")
    cred_df = generate_credit_dataset(app_df)
    
    # Save them
    app_df.to_csv("data/applicant_input.csv", index=False)
    cred_df.to_csv("data/third_party_credit.csv", index=False)
    print("Saved 'data/applicant_input.csv' and 'data/third_party_credit.csv'.")
    
    print("Merging datasets and applying rules...")
    final_df = merge_and_label(app_df, cred_df)
    
    # 5–10% outliers, 1–2% missing data => let's do 5% & 2%
    print("Adding ~5% outliers and ~2% missing credit_score for realism...")
    final_df = introduce_noise_for_realism(final_df)
    
    # Save final
    final_df.to_csv("data/final_labeled_dataset.csv", index=False)
    
    # Print actual approval rate
    approval_rate = final_df["approved"].mean() * 100
    print(f"Final labeled dataset saved as 'data/final_labeled_dataset.csv'.")
    print(f"Natural Approval Rate: {approval_rate:.2f}%")

if __name__ == "__main__":
    main()
