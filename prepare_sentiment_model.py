
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load cleaned reviews
data_dir = Path("Cleaned datasets")
df = pd.read_csv(data_dir / "customer_reviews_cleaned.csv")

# Step 2: Select features and target
X = df["review_text"].fillna("").astype(str)
y = df["sentiment_label"]

# Step 3: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Apply TF-IDF
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Save files
joblib.dump(vectorizer, data_dir / "sentiment_tfidf.pkl")

joblib.dump(
    (X_train_tfidf, X_test_tfidf, y_train, y_test),
    data_dir / "sentiment_train_test.pkl"
)

print("STEP 5.3 COMPLETED")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nFiles saved successfully!")
