
import pandas as pd
from pathlib import Path

# Locate the original customer reviews dataset
files = list(Path(".").rglob("customer_reviews.jsonl"))

if not files:
    print("ERROR: customer_reviews.jsonl not found.")
    raise SystemExit(1)

file_path = files[0]
print("Dataset location:", file_path)

# Load JSON Lines dataset
df = pd.read_json(file_path, lines=True)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 RECORDS:")
print(df.head().to_string(index=False))

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)

# Check possible sentiment-related columns
for column in df.columns:
    if any(
        keyword in column.lower()
        for keyword in ["sentiment", "rating", "label"]
    ):
        print(f"\nVALUE DISTRIBUTION: {column}")
        print(df[column].value_counts(dropna=False).head(20))
