import os
import numpy as np
import pandas as pd

# Ensure the output folder exists
os.makedirs("data", exist_ok=True)

NUM_ROWS = 5000
np.random.seed(42)


def generate_applicant_input(num_rows):
    """
    Generate synthetic applicant input data.
    """
    applicant_ids = [f"APP{i:05d}" for i in range(1, num_rows + 1)]
    
    # Annual income: Lognormal distribution with μ=log(100000)-0.5, σ=0.3, clipped to [20k, 200k]
    mu, sigma = np.log(100000) - 0.5, 0.3
    annual_income = np.random.lognormal(mean=mu, sigma=sigma, size=num_rows)
    annual_income = np.clip(annual_income, 20000, 200000)
    
    # Self reported expenses: Uniform distribution between 0 and 10,000
    self_reported_expenses = np.random.uniform(0, 10000, num_rows)
    
    # Calculate monthly income for internal debt calculations
    monthly_income = annual_income / 12
    
    # Self reported debt: 5-15% of monthly income, clipped to maximum 10,000
    self_reported_debt = monthly_income * np.random.uniform(0.05, 0.15, num_rows)
    self_reported_debt = np.clip(self_reported_debt, 0, 10000)
    
    # Requested credit amount: Uniform distribution between 1,000 and 12,000
    requested_amount = np.random.uniform(1000, 12000, num_rows)
    
    # Other fields
    age = np.random.randint(19, 101, num_rows)
    province = np.random.choice(["ON", "BC"], size=num_rows, p=[0.65, 0.35])
    employment_status = np.random.choice(["Full-time", "Part-time", "Unemployed"],
                                         size=num_rows, p=[0.6, 0.3, 0.1])
    months_employed = [0 if status == "Unemployed" else np.random.randint(1, 601)
                       for status in employment_status]
    
    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "annual_income": annual_income,
        "self_reported_expenses": self_reported_expenses,
        "self_reported_debt": self_reported_debt,
        "requested_amount": requested_amount,
        "age": age,
        "province": province,
        "employment_status": employment_status,
        "months_employed": months_employed
    })
    return df

def generate_credit_dataset(applicant_df):
    """
    Generate a third-party credit dataset based on applicant data.
    """
    num_rows = len(applicant_df)
    applicant_ids = applicant_df["applicant_id"].tolist()
    
    # Credit score: Normal distribution (mean=720, std=80), clipped to [300, 900]
    credit_score = np.random.normal(720, 80, num_rows)
    credit_score = np.clip(credit_score, 300, 900).astype(int)
    
    # Total credit limit: 50–200% of annual income, clipped to a maximum of 50,000
    annual_income = applicant_df["annual_income"].values
    factor = np.random.uniform(0.5, 2.0, num_rows)
    total_credit_limit = annual_income * factor
    total_credit_limit = np.clip(total_credit_limit, 0, 50000)
    
    # Credit utilization: Beta distribution (3,7) scaled to percentage
    credit_utilization = np.random.beta(3, 7, num_rows) * 100
    
    # Other credit features
    num_open_accounts = np.random.randint(0, 21, num_rows)
    num_credit_inquiries = np.random.randint(0, 11, num_rows)
    payment_history = np.random.choice(
        ["On Time", "Late <30", "Last 30-60", "Late>60"],
        size=num_rows,
        p=[0.85, 0.10, 0.03, 0.02]
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


def compute_estimated_debt(total_credit_limit, credit_utilization):
    """
    Compute estimated (unreported) debt.
    """
    return total_credit_limit * (credit_utilization / 100) * 0.03

def compute_dti(self_reported_debt, estimated_debt, requested_amount, annual_income):
    """
    Compute Debt-to-Income (DTI) ratio.
    """
    monthly_income = annual_income / 12
    return (self_reported_debt + estimated_debt + (requested_amount * 0.03)) / monthly_income

# ---- Approved Amount Calculation Components ----
def base_limit(credit_score, annual_income):
    """
    Calculate base limit depending on credit score.
    """
    return np.where(
        credit_score >= 660, 0.50 * annual_income,
        np.where(credit_score >= 500, 0.25 * annual_income, 0.10 * annual_income)
    )

def dti_adjustment(DTI):
    """
    Adjustment factor based on DTI.
    """
    return np.where(
        DTI <= 0.30, 1.0,
        np.where(DTI <= 0.40, 0.75, 0.50)
    )

def employment_bonus(employment_status, months_employed):
    """
    Return bonus factor: +10% if Full-time and months_employed >= 12.
    """
    return np.where((employment_status == "Full-time") & (months_employed >= 12), 1.10, 1.0)

def payment_penalty(payment_history):
    """
    Return penalty factor: -50% if payment history is "Late>60".
    """
    return np.where(payment_history == "Late>60", 0.50, 1.0)

def utilization_penalty(credit_utilization):
    """
    Return penalty factor: -20% if credit utilization is above 50%.
    """
    return np.where(credit_utilization > 50, 0.80, 1.0)

def score_cap(credit_score):
    """
    Determine the cap on approved credit based on credit score.
    """
    return np.where(
        credit_score >= 750, 25000,
        np.where(credit_score >= 660, 15000,
                 np.where(credit_score >= 500, 10000, 5000))
    )

def compute_approved_amount(credit_score, annual_income, DTI, employment_status, months_employed,
                            payment_history, credit_utilization):
    """
    Compute approved amount by combining each separate adjustment.
    """
    base = base_limit(credit_score, annual_income)
    dti_adj = dti_adjustment(DTI)
    bonus = employment_bonus(employment_status, months_employed)
    penalty_pay = payment_penalty(payment_history)
    penalty_util = utilization_penalty(credit_utilization)
    cap = score_cap(credit_score)
    
    limit_adj = base * dti_adj * bonus * penalty_pay * penalty_util
    approved_amount = np.minimum(limit_adj, cap)
    return approved_amount

def apply_denial_conditions(approved_amount, DTI, credit_score, credit_utilization):
    """
    Set approved amount to zero if any denial conditions are met.
    """
    approved_amount = np.where(DTI > 0.40, 0, approved_amount)
    approved_amount = np.where(credit_score < 500, 0, approved_amount)
    approved_amount = np.where(credit_utilization > 80, 0, approved_amount)
    return approved_amount

# ---- Interest Rate Calculation Components ----
def base_interest_rate(n):
    """
    Return an array of base interest rate (6.75%).
    """
    return np.full(n, 6.75)

def adjust_for_credit_score(rate, credit_score):
    """
    Adjust interest rate based on credit score.
    """
    return np.where(
        credit_score >= 750, rate - 1.0,
        np.where(credit_score < 500, rate + 4.0,
                 np.where((credit_score >= 500) & (credit_score <= 659), rate + 2.0, rate))
    )

def adjust_for_dti(rate, DTI):
    """
    Add 1% if DTI is greater than 30%.
    """
    return rate + np.where(DTI > 0.30, 1.0, 0.0)

def adjust_for_payment_history(rate, payment_history):
    """
    Add 2% if payment history is "Late>60".
    """
    return rate + np.where(payment_history == "Late>60", 2.0, 0.0)

def adjust_for_open_accounts(rate, num_open_accounts):
    """
    Add 1% if the number of open accounts is greater than 5.
    """
    return rate + np.where(num_open_accounts > 5, 1.0, 0.0)

def finalize_interest_rate(rate, approved_amount):
    """
    Finalize interest rate:
      - Clip the rate between 3.0 and 15.0.
      - If no approved amount, set interest to 0.
    """
    rate = np.clip(rate, 3.0, 15.0)
    return np.where(approved_amount > 0, rate, 0.0)

def compute_interest_rate(n, credit_score, DTI, payment_history, num_open_accounts, approved_amount):
    """
    Combine the interest rate components into a single interest rate calculation.
    """
    rate = base_interest_rate(n)
    rate = adjust_for_credit_score(rate, credit_score)
    rate = adjust_for_dti(rate, DTI)
    rate = adjust_for_payment_history(rate, payment_history)
    rate = adjust_for_open_accounts(rate, num_open_accounts)
    return finalize_interest_rate(rate, approved_amount)

def compute_basic_approval(credit_score, DTI, payment_history):
    """
    Basic rule-based approval:
      Approve if credit_score >= 660, DTI <= 0.40, and payment history is "On Time".
    """
    cond = (credit_score >= 660) & (DTI <= 0.40) & (payment_history == "On Time")
    return np.where(cond, 1, 0)

def apply_random_overrides(df):
    """
    Apply small random adjustments for borderline cases:
      - For borderline credit score (650-669) or near-boundary DTI (39%-42%), flip approval with 30% chance.
      - For "Late <30" or "Last 30-60", override to approve with a 10% chance.
    Also, if approval is flipped to approved, ensure a minimal approved amount and interest rate.
    """
    rng = np.random.default_rng(42)
    
    # Borderline conditions
    borderline_mask = (df["credit_score"].between(650, 669)) | (df["DTI"].between(0.39, 0.42))
    borderline_indices = df[borderline_mask].index
    if len(borderline_indices) > 0:
        flip_indices = rng.choice(borderline_indices, size=int(len(borderline_indices) * 0.3), replace=False)
        df.loc[flip_indices, "approved"] = 1 - df.loc[flip_indices, "approved"]
        # Ensure minimal values for newly approved applications
        just_approved = flip_indices[df.loc[flip_indices, "approved"] == 1]
        df.loc[just_approved, "approved_amount"] = np.where(
            df.loc[just_approved, "approved_amount"] == 0, 5000, df.loc[just_approved, "approved_amount"]
        )
        df.loc[just_approved, "interest_rate"] = np.where(
            df.loc[just_approved, "interest_rate"] == 0, 12, df.loc[just_approved, "interest_rate"]
        )
    
    # Overrides for specific payment history conditions
    late_mask = df["payment_history"].isin(["Late <30", "Last 30-60"])
    late_indices = df[late_mask].index
    if len(late_indices) > 0:
        flip_late = rng.choice(late_indices, size=int(len(late_indices) * 0.1), replace=False)
        df.loc[flip_late, "approved"] = 1
        df.loc[flip_late, "approved_amount"] = np.where(
            df.loc[flip_late, "approved_amount"] == 0, 3000, df.loc[flip_late, "approved_amount"]
        )
        df.loc[flip_late, "interest_rate"] = np.where(
            df.loc[flip_late, "interest_rate"] == 0, 14, df.loc[flip_late, "interest_rate"]
        )
    return df


def merge_and_label(app_df, credit_df):
    """
    Merge the applicant and credit datasets and compute all target features.
    """
    df = pd.merge(app_df, credit_df, on="applicant_id")
    
    # Compute estimated debt using the separate function
    df["estimated_debt"] = compute_estimated_debt(df["total_credit_limit"], df["credit_utilization"])
    
    # Compute Debt-to-Income ratio
    df["DTI"] = compute_dti(df["self_reported_debt"], df["estimated_debt"], df["requested_amount"], df["annual_income"])
    
    # Compute approved amount in two steps:
    raw_approved = compute_approved_amount(
        df["credit_score"],
        df["annual_income"],
        df["DTI"],
        df["employment_status"],
        df["months_employed"],
        df["payment_history"],
        df["credit_utilization"]
    )
    df["approved_amount"] = apply_denial_conditions(raw_approved, df["DTI"], df["credit_score"], df["credit_utilization"])
    
    # Compute interest rate
    n = len(df)
    df["interest_rate"] = compute_interest_rate(
        n,
        df["credit_score"],
        df["DTI"],
        df["payment_history"],
        df["num_open_accounts"],
        df["approved_amount"]
    )
    
    # Basic approval decision based on rules
    df["approved"] = compute_basic_approval(df["credit_score"], df["DTI"], df["payment_history"])
    
    # Apply random overrides to simulate real-world noise
    df = apply_random_overrides(df)
    
    return df


def introduce_noise_for_realism(df):
    """
    Add noise to mimic real-world data:
      - 5% outliers in annual_income.
      - 2% missing values in credit_score.
    """
    n = len(df)
    # Outliers in annual_income
    outlier_count = int(0.05 * n)
    outlier_rows = np.random.choice(df.index, size=outlier_count, replace=False)
    df.loc[outlier_rows, "annual_income"] *= np.random.uniform(1.5, 3.0, size=outlier_count)
    
    # Missing values in credit_score
    missing_count = int(0.02 * n)
    missing_rows = np.random.choice(df.index, size=missing_count, replace=False)
    df.loc[missing_rows, "credit_score"] = np.nan
    return df


def main():
    print("Generating applicant input dataset...")
    app_df = generate_applicant_input(NUM_ROWS)
    
    print("Generating third-party credit dataset...")
    cred_df = generate_credit_dataset(app_df)
    
    # Save individual datasets
    app_df.to_csv("data/applicant_input.csv", index=False)
    cred_df.to_csv("data/third_party_credit.csv", index=False)
    print("Saved: applicant_input.csv and third_party_credit.csv.")
    
    print("Merging datasets and computing features...")
    final_df = merge_and_label(app_df, cred_df)
    
    print("Introducing noise for realism...")
    final_df = introduce_noise_for_realism(final_df)
    
    final_df.to_csv("data/final_labeled_dataset.csv", index=False)
    approval_rate = final_df["approved"].mean() * 100
    print("Final dataset saved => data/final_labeled_dataset.csv")
    print(f"Natural Approval Rate: {approval_rate:.2f}%")

if __name__ == "__main__":
    main()
