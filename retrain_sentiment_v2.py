
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

folder = Path("Cleaned datasets")

# Load existing cleaned reviews
original_df = pd.read_csv(
    folder / "customer_reviews_cleaned.csv"
)

# Load additional training examples
additional_df = pd.read_csv(
    folder / "sentiment_additional_examples.csv"
)

# Combine datasets
combined_df = pd.concat(
    [original_df, additional_df],
    ignore_index=True
)

# Keep required columns
combined_df = combined_df[
    ["review_text", "sentiment_label"]
].dropna()

combined_df["review_text"] = (
    combined_df["review_text"].astype(str).str.strip()
)

combined_df = combined_df[
    combined_df["review_text"] != ""
].drop_duplicates(subset=["review_text"])

print("Combined training records:", len(combined_df))

print("\nSentiment distribution:")
print(combined_df["sentiment_label"].value_counts())

# Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    lowercase=True
)

X = vectorizer.fit_transform(
    combined_df["review_text"]
)

y = combined_df["sentiment_label"]

# Train Linear SVM
model = LinearSVC(
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

# Save updated model and vectorizer
joblib.dump(
    model,
    folder / "sentiment_model_v2.pkl"
)

joblib.dump(
    vectorizer,
    folder / "sentiment_tfidf_v2.pkl"
)

print("\nSTEP 5.9.12 COMPLETED")
print("TF-IDF features:", len(vectorizer.vocabulary_))
print("Updated model and vectorizer saved successfully!")
