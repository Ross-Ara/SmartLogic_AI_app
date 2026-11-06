
import joblib
import pandas as pd
from pathlib import Path

model_path = Path("sentiment_model.pkl")

if not model_path.exists():
    model_path = Path("Cleaned datasets/sentiment_model.pkl")

model = joblib.load(model_path)

test_reviews = [
    "The delivery was excellent and arrived on time.",
    "My package arrived damaged and very late.",
    "The parcel was delivered yesterday afternoon.",
    "The delivery was okay, nothing special.",
    "The package arrived on time but the packaging was damaged.",
    "The driver was polite and the delivery was quick.",
    "The product was fine but delivery was slow.",
    "I received my order today."
]

from pathlib import Path
import joblib

vectorizer_path = Path("sentiment_vectorizer.pkl")

if not vectorizer_path.exists():
    vectorizer_path = Path(
        "Cleaned datasets/sentiment_tfidf.pkl"
    )

vectorizer = joblib.load(vectorizer_path)

# Convert review text into numerical TF-IDF features
X_test_reviews = vectorizer.transform(test_reviews)

# Predict using the trained model
predictions = model.predict(X_test_reviews)
results = pd.DataFrame({
    "Review": test_reviews,
    "Predicted Sentiment": predictions
})

print("\nSENTIMENT MODEL TEST RESULTS")
print(results.to_string(index=False))

