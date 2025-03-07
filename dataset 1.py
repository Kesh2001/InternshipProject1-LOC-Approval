import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
num_records = 5000

# Function to generate Applicant Input Dataset
def generate_applicant_data(num_records):
    data = {
        "applicant_id": [f"A{str(i).zfill(5)}" for i in range(1, num_records + 1)],
        "annual_income": np.random.lognormal(mean=np.log(60000), sigma=0.5, size=num_records).astype(int),
        "self_reported_debt": np.random.uniform(0, 10000, num_records),
        "self_reported_expenses": np.random.uniform(0, 10000, num_records),
        "requested_amount": np.random.uniform(1000, 50000, num_records),
        "age": np.random.randint(19, 101, num_records),
        "province": np.random.choice(["ON", "BC", "QC", "AB", "MB"], num_records),
        "employment_status": np.random.choice(["Full-time", "Part-time", "Unemployed"], num_records),
        "months_employed": np.random.randint(0, 600, num_records)
    }
    return pd.DataFrame(data)

# Function to generate Third-Party Credit Dataset
def generate_credit_data(num_records):
    data = {
        "applicant_id": [f"A{str(i).zfill(5)}" for i in range(1, num_records + 1)],
        "credit_score": np.random.normal(loc=680, scale=100, size=num_records).astype(int),
        "total_credit_limit": np.random.uniform(20000, 100000, num_records),
        "credit_utilization": np.random.uniform(0, 1, num_records),
        "num_open_accounts": np.random.randint(0, 21, num_records),
        "num_credit_inquiries": np.random.randint(0, 11, num_records),
        "payment_history": np.random.choice(["On Time", "Late <30", "Late 30-60", "Late >60"], num_records)
    }
    return pd.DataFrame(data)

# Generate datasets
applicant_df = generate_applicant_data(num_records)
credit_df = generate_credit_data(num_records)

# Merge datasets
final_df = pd.merge(applicant_df, credit_df, on="applicant_id")

# Calculate estimated_debt
final_df["estimated_debt"] = final_df["total_credit_limit"] * final_df["credit_utilization"] * 0.03

# Calculate total debt
final_df["total_monthly_debt"] = final_df["self_reported_debt"] + final_df["estimated_debt"]

# Calculate DTI (Debt-to-Income Ratio)
final_df["DTI"] = (final_df["total_monthly_debt"] + (final_df["requested_amount"] * 0.03)) / (final_df["annual_income"] / 12)

# Determine approval status
final_df["approved"] = np.where(
    (final_df["credit_score"] >= 660) & (final_df["DTI"] <= 0.40) & (final_df["payment_history"] == "On Time"),
    1,
    np.where(
        (final_df["credit_score"] < 500) | (final_df["DTI"] > 0.50) | (final_df["credit_utilization"] > 0.80),
        0,
        np.random.choice([0, 1], p=[0.3, 0.7], size=num_records)  # Handle uncertain cases randomly
    )
)

# Calculate Credit Limit
def calculate_credit_limit(row):
    base_limit = 0.5 * row["annual_income"] if row["credit_score"] >= 660 else (
        0.25 * row["annual_income"] if row["credit_score"] >= 500 else 0.1 * row["annual_income"]
    )
    
    # Adjust for DTI
    if row["DTI"] > 0.40:
        base_limit *= 0.5
    elif row["DTI"] > 0.30:
        base_limit *= 0.75

    # Apply credit score caps
    if row["credit_score"] >= 750:
        base_limit = min(base_limit, 25000)
    elif row["credit_score"] >= 660:
        base_limit = min(base_limit, 15000)
    elif row["credit_score"] >= 500:
        base_limit = min(base_limit, 10000)
    else:
        base_limit = min(base_limit, 5000)

    # Employment bonus
    if row["employment_status"] == "Full-time" and row["months_employed"] >= 12:
        base_limit *= 1.1

    # Payment penalty
    if row["payment_history"] == "Late >60":
        base_limit *= 0.5

    return min(max(base_limit, 1000), row["requested_amount"])

final_df["approved_amount"] = final_df.apply(calculate_credit_limit, axis=1)

# Calculate Interest Rate
def calculate_interest_rate(row):
    rate = 6.75  # Base rate

    if row["credit_score"] >= 750:
        rate -= 1.0
    elif row["credit_score"] >= 660:
        rate += 0.0
    elif row["credit_score"] >= 500:
        rate += 2.0
    else:
        rate += 4.0

    if row["DTI"] > 0.30:
        rate += 1.0

    if row["payment_history"] == "Late >60":
        rate += 2.0

    return min(max(rate, 3.0), 15.0)

final_df["interest_rate"] = final_df.apply(calculate_interest_rate, axis=1)

# Save datasets
applicant_df.to_csv("applicant_input.csv", index=False)
credit_df.to_csv("third_party_credit.csv", index=False)
final_df.to_csv("final_dataset.csv", index=False)

print("Datasets generated successfully!")
