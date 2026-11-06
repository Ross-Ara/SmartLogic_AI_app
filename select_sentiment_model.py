
import joblib
import pandas as pd
from pathlib import Path

data_dir = Path("Cleaned datasets")

# Load model comparison
comparison = pd.read_csv(
    data_dir / "sentiment_model_comparison.csv"
)

print("Sentiment Model Comparison:")
print(comparison.to_string(index=False))

# Select Linear SVM (models tied on test metrics)
best_model = joblib.load(
    data_dir / "sentiment_linear_svm.pkl"
)

# Save final model
joblib.dump(
    best_model,
    data_dir / "sentiment_model.pkl"
)

# Verify TF-IDF vectorizer
vectorizer = joblib.load(
    data_dir / "sentiment_tfidf.pkl"
)

# Test with new example reviews
examples = [
    "The package arrived on time and the service was excellent.",
    "My order was damaged and the delivery was very late.",
    "The package was delivered yesterday."
]

features = vectorizer.transform(examples)
predictions = best_model.predict(features)

print("\nSample Predictions:")
for review, prediction in zip(examples, predictions):
    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {prediction}")

print("\nSTEP 5.5 COMPLETED")
print("Final model saved: sentiment_model.pkl")
