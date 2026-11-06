
import joblib
from pathlib import Path

# Load the TF-IDF vectorizer
vectorizer_path = Path("Cleaned datasets/sentiment_tfidf_improved.pkl")
vectorizer = joblib.load(vectorizer_path)

print("Vectorizer file loaded:", vectorizer_path.resolve())
print("N-gram range:", vectorizer.ngram_range)
print("Maximum features:", vectorizer.max_features)


# Words we want to investigate
test_words = [
    "excellent",
    "terrible",
    "disappointing",
    "damaged",
    "late",
    "quick",
    "polite",
    "slow",
    "yesterday",
    "delivered",
    "packaging"
]

vocabulary = vectorizer.vocabulary_

print("\nSENTIMENT VOCABULARY CHECK")
print("=" * 45)

for word in test_words:
    if word in vocabulary:
        print(f"{word:20} FOUND")
    else:
        print(f"{word:20} NOT FOUND")

print("\nTotal TF-IDF features:", len(vocabulary))
