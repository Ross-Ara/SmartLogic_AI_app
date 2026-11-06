
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

folder = Path("Cleaned datasets")

# Load original and additional reviews
original = pd.read_csv(folder / "customer_reviews_cleaned.csv")
additional = pd.read_csv(folder / "sentiment_diverse_examples.csv")

columns = ["review_text", "sentiment_label"]

original = original[columns].dropna().copy()
additional = additional[columns].dropna().copy()

# Preserve the original evaluation split
train_df, test_df = train_test_split(
    original,
    test_size=0.2,
    random_state=42,
    stratify=original["sentiment_label"]
)

# Add new examples only to training data
train_df = pd.concat(
    [train_df, additional],
    ignore_index=True
)

# Remove duplicate training reviews
train_df = train_df.drop_duplicates(
    subset=["review_text"]
)

# Ensure training and testing texts do not overlap
train_df["_key"] = (
    train_df["review_text"].str.strip().str.lower()
)
test_df["_key"] = (
    test_df["review_text"].str.strip().str.lower()
)

train_df = train_df[
    ~train_df["_key"].isin(test_df["_key"])
].drop(columns="_key")

test_df = test_df.drop(columns="_key")

# TF-IDF + Linear SVM in one pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        lowercase=True
    )),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(
    train_df["review_text"],
    train_df["sentiment_label"]
)

predictions = model.predict(test_df["review_text"])

print("\nSTEP 5.9.18 - MODEL V3 EVALUATION")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Accuracy:", round(
    accuracy_score(test_df["sentiment_label"], predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    test_df["sentiment_label"],
    predictions,
    zero_division=0
))

# Save without overwriting previous models
joblib.dump(model, folder / "sentiment_model_v3.pkl")

print("\nSTEP 5.9.18 COMPLETED")
print("Saved: Cleaned datasets/sentiment_model_v3.pkl")
