
import pandas as pd
from pathlib import Path

file_path = Path("Cleaned datasets/customer_reviews_cleaned.csv")

df = pd.read_csv(file_path)

print("\nTOTAL RECORDS:", len(df))

print("\nSENTIMENT DISTRIBUTION:")
print(df["sentiment_label"].value_counts())

print("\nAVERAGE REVIEW LENGTH BY SENTIMENT:")
df["word_count"] = (
    df["review_text"].fillna("").astype(str).str.split().str.len()
)
print(df.groupby("sentiment_label")["word_count"].mean())

print("\nSAMPLE REVIEWS BY SENTIMENT:")

for sentiment in ["Positive", "Negative", "Neutral"]:
    print(f"\n===== {sentiment.upper()} =====")

    examples = df[
        df["sentiment_label"] == sentiment
    ]["review_text"].head(10)

    for review in examples:
        print("-", review)
