
import pandas as pd
from pathlib import Path

DATA_DIR = Path("Cleaned datasets")

files = [
    "gps_routes_structured.csv",
    "traffic_clean.csv",
    "traffic_features.csv",
    "route_performance_summary.csv"
]

for filename in files:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("DATASET:", filename)
    print("=" * 60)

    if not file_path.exists():
        print("File not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())
