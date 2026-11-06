
import pandas as pd
from pathlib import Path

folder = Path("Cleaned datasets")

original = pd.read_csv(
    folder / "customer_reviews_cleaned.csv"
)

additional = pd.read_csv(
    folder / "sentiment_additional_examples.csv"
)

df = pd.concat(
    [original, additional],
    ignore_index=True
)

df = df[["review_text", "sentiment_label"]].dropna().copy()

df["normalized_review"] = (
    df["review_text"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df["word_count"] = (
    df["normalized_review"].str.split().str.len()
)

print("\nSENTIMENT DATASET AUDIT")
print("=" * 50)

print("Total records:", len(df))

print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())

print("\nDuplicate normalized reviews:")
print(df["normalized_review"].duplicated().sum())

conflicts = (
    df.groupby("normalized_review")["sentiment_label"]
    .nunique()
)

print("\nReviews with conflicting labels:")
print((conflicts > 1).sum())

print("\nReviews containing fewer than 4 words:")
print((df["word_count"] < 4).sum())

print("\nAverage word count by sentiment:")
print(
    df.groupby("sentiment_label")["word_count"].mean().round(2)
)

print("\nDataset audit completed.")
