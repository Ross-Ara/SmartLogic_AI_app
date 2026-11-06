
import pandas as pd
from pathlib import Path

# Locate original dataset
files = list(Path(".").rglob("customer_reviews.jsonl"))

if not files:
    raise FileNotFoundError("customer_reviews.jsonl not found")

df = pd.read_json(files[0], lines=True)

# Standardize sentiment labels
df["sentiment_label"] = (
    df["sentiment_label"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "pos": "Positive",
        "positive": "Positive",
        "neg": "Negative",
        "negative": "Negative",
        "neu": "Neutral",
        "neutral": "Neutral"
    })
)

valid_labels = ["Positive", "Negative", "Neutral"]

# Keep only records with known sentiment labels
clean_df = df[
    df["sentiment_label"].isin(valid_labels)
].copy()

# Identify the review text column
possible_columns = [
    "review_text",
    "review",
    "review_body",
    "comment",
    "feedback",
    "text"
]

text_column = next(
    (col for col in possible_columns if col in clean_df.columns),
    None
)

if text_column is None:
    raise ValueError(
        f"Review text column not found. Columns: {df.columns.tolist()}"
    )

# Clean review text
clean_df[text_column] = (
    clean_df[text_column]
    .astype("string")
    .str.strip()
)

clean_df = clean_df.dropna(subset=[text_column])
clean_df = clean_df[
    clean_df[text_column].str.len() > 0
].copy()

# Remove exact duplicate review-label pairs
clean_df = clean_df.drop_duplicates(
    subset=[text_column, "sentiment_label"]
)

# Save cleaned dataset
output_dir = Path("Cleaned datasets")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "customer_reviews_cleaned.csv"
clean_df.to_csv(output_file, index=False)

print("Review text column:", text_column)
print("Original records:", len(df))
print("Cleaned records:", len(clean_df))

print("\nCleaned sentiment distribution:")
print(clean_df["sentiment_label"].value_counts())

print("\nFirst 5 cleaned records:")
print(
    clean_df[[text_column, "sentiment_label"]]
    .head()
    .to_string(index=False)
)

print("\nSaved to:", output_file)
