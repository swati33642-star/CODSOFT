"""
generate_sample_data.py

Creates a SYNTHETIC dataset that mimics the structure of the popular
Kaggle "Credit Card Fraud Detection" dataset (Time, V1-V28, Amount, Class),
so you can run and test the full pipeline immediately.

This is NOT real data. For your actual internship submission, replace
data/creditcard.csv with the real dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_NORMAL = 20000
N_FRAUD = 100  # keeps a realistic ~0.5% fraud rate

n_features = 28  # V1 ... V28

def make_class(n, fraud=False):
    if fraud:
        # Shift mean/variance so fraud rows are statistically distinguishable
        base = np.random.normal(loc=1.5, scale=2.5, size=(n, n_features))
        amount = np.random.exponential(scale=300, size=n) + 50
    else:
        base = np.random.normal(loc=0.0, scale=1.0, size=(n, n_features))
        amount = np.random.exponential(scale=60, size=n)

    time = np.sort(np.random.uniform(0, 172792, size=n))
    df = pd.DataFrame(base, columns=[f"V{i}" for i in range(1, n_features + 1)])
    df.insert(0, "Time", time)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = 1 if fraud else 0
    return df

normal_df = make_class(N_NORMAL, fraud=False)
fraud_df = make_class(N_FRAUD, fraud=True)

full_df = pd.concat([normal_df, fraud_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

full_df.to_csv("data/creditcard.csv", index=False)

print(f"Synthetic dataset created at data/creditcard.csv")
print(f"Shape: {full_df.shape}")
print(f"Fraud cases: {full_df['Class'].sum()} ({full_df['Class'].mean()*100:.3f}% of data)")
