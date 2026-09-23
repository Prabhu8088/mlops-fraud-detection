import numpy as np
import pandas as pd

np.random.seed(42)

NUM_RECORDS = 10000

amount = np.random.exponential(scale=200, size=NUM_RECORDS)
transaction_hour = np.random.randint(0, 24, NUM_RECORDS)
distance_from_home = np.random.exponential(scale=20, size=NUM_RECORDS)
previous_transactions = np.random.poisson(lam=5, size=NUM_RECORDS)

merchant_categories = [
    "grocery",
    "electronics",
    "travel",
    "restaurant",
    "clothing"
]

merchant_category = np.random.choice(
    merchant_categories,
    size=NUM_RECORDS
)

fraud_score = (
    (amount > 500) * 2
    + (transaction_hour < 5) * 2
    + (distance_from_home > 100) * 2
    + (previous_transactions < 2) * 1
    + (merchant_category == "electronics") * 1
)

fraud_probability = 1 / (1 + np.exp(-(fraud_score - 3)))

fraud = (
    np.random.random(NUM_RECORDS) < fraud_probability
).astype(int)

df = pd.DataFrame({
    "amount": amount,
    "transaction_hour": transaction_hour,
    "merchant_category": merchant_category,
    "distance_from_home": distance_from_home,
    "previous_transactions": previous_transactions,
    "fraud": fraud
})

df.to_csv("fraud_data.csv", index=False)

print(f"Generated {len(df)} records")
print(df.head())
print("\nFraud distribution:")
print(df["fraud"].value_counts())
