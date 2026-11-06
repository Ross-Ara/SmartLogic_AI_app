
import joblib
import re
from pathlib import Path

folder = Path("Cleaned datasets")

vectorizer = joblib.load(
    folder / "sentiment_tfidf_v2.pkl"
)

reviews = [
    "Everything arrived in perfect condition.",
    "I loved the delivery experience.",
    "My order arrived broken and unusable.",
    "The package was received on Monday.",
    "The driver left the parcel at the entrance.",
    "I collected the shipment from the office."
]

vocabulary = vectorizer.vocabulary_

print("V2 VOCABULARY ANALYSIS")
print("=" * 50)

for review in reviews:
    words = re.findall(r"\b\w+\b", review.lower())

    missing_words = [
        word for word in words
        if word not in vocabulary
    ]

    print("\nReview:", review)
    print("Missing words:", missing_words)

print("\nTotal TF-IDF features:", len(vocabulary))
