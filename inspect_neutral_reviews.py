
import pandas as pd

df = pd.read_csv(
    "Cleaned datasets/customer_reviews_cleaned.csv"
)

neutral_df = df[
    df["sentiment_label"] == "Neutral"
]

print("Total Neutral Reviews:", len(neutral_df))

print("\nALL NEUTRAL REVIEWS:")
for i, review in enumerate(neutral_df["review_text"], 1):
    print(f"{i}. {review}")

print("\nAverage Review Length:")
print(
    neutral_df["review_text"]
    .astype(str)
    .str.split()
    .str.len()
    .mean()
)
