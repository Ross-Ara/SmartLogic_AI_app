
import pandas as pd

df = pd.read_csv(
    "Cleaned datasets/customer_reviews_cleaned.csv"
)

df["normalized_text"] = (
    df["review_text"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("Total records:", len(df))
print("Exact duplicate reviews:",
      df["normalized_text"].duplicated().sum())

print("Unique reviews:",
      df["normalized_text"].nunique())

print("\nReviews with conflicting sentiment labels:")
conflicts = (
    df.groupby("normalized_text")["sentiment_label"]
    .nunique()
)
print((conflicts > 1).sum())

print("\nMost repeated reviews:")
print(df["normalized_text"].value_counts().head(10))
