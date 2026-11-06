import pandas as pd
from pathlib import Path

DATA_DIR = Path("Cleaned datasets")

files = [
    "smartlogix_maintenance_ml_dataset.csv",
    "maintenance_model_comparison.csv"
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("DATASET:", filename)
    print("=" * 60)

    if not path.exists():
        print("File not found:", path)
        continue

    df = pd.read_csv(path)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())