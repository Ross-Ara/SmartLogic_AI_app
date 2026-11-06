
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

folder = Path("Cleaned datasets")

# Load datasets
original = pd.read_csv(
    folder / "customer_reviews_cleaned.csv"
)

additional = pd.read_csv(
    folder / "neutral_examples_additional.csv"
)

# Combine original and additional reviews
df = pd.concat(
    [original, additional],
    ignore_index=True
)

df = df[["review_text", "sentiment_label"]].dropna()

# Remove exact duplicates
df = df.drop_duplicates(subset=["review_text"])

print("Total training candidates:", len(df))
print("\nSentiment Distribution:")
print(df["sentiment_label"].value_counts())

# Split original reviews first, keeping new examples
# exclusively for training.
original = original[
    ["review_text", "sentiment_label"]
].dropna().drop_duplicates(subset=["review_text"])

train_original, test_df = train_test_split(
    original,
    test_size=0.2,
    random_state=42,
    stratify=original["sentiment_label"]
)

train_df = pd.concat(
    [train_original, additional],
    ignore_index=True
)

# Prevent exact overlap between train and test
test_texts = set(
    test_df["review_text"].str.lower().str.strip()
)

train_df = train_df[
    ~train_df["review_text"].str.lower().str.strip()
    .isin(test_texts)
].copy()

# Fit TF-IDF only on training reviews
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train = vectorizer.fit_transform(
    train_df["review_text"]
)

X_test = vectorizer.transform(
    test_df["review_text"]
)

y_train = train_df["sentiment_label"]
y_test = test_df["sentiment_label"]

# Train improved Linear SVM
model = LinearSVC(
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nIMPROVED MODEL EVALUATION")
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save separately to preserve existing model
joblib.dump(
    model,
    folder / "sentiment_model_improved.pkl"
)

joblib.dump(
    vectorizer,
    folder / "sentiment_tfidf_improved.pkl"
)

print("\nImproved model files saved successfully!")
