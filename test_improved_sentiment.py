
import joblib
import pandas as pd
from pathlib import Path

folder = Path("Cleaned datasets")

model = joblib.load(
    folder / "sentiment_model_improved.pkl"
)

vectorizer = joblib.load(
    folder / "sentiment_tfidf_improved.pkl"
)

test_reviews = [
    "The delivery was excellent and arrived on time.",
    "My package arrived damaged and very late.",
    "The parcel was delivered yesterday afternoon.",
    "The delivery was okay, nothing special.",
    "The package arrived on time but the packaging was damaged.",
    "The driver was polite and the delivery was quick.",
    "The product was fine but delivery was slow.",
    "I received my order today.",
    "The courier delivered the parcel today.",
    "The service was terrible and disappointing."
]

expected = [
    "Positive",
    "Negative",
    "Neutral",
    "Neutral",
    "Negative",
    "Positive",
    "Neutral",
    "Neutral",
    "Neutral",
    "Negative"
]

X_test = vectorizer.transform(test_reviews)
predictions = model.predict(X_test)



# Check the model's prediction confidence
if hasattr(model, "decision_function"):
    scores = model.decision_function(X_test)

    print("\nMODEL DECISION SCORES")
    print("Class order:", model.classes_)

    score_df = pd.DataFrame(
        scores,
        columns=model.classes_
    )

    score_df.insert(0, "Review", test_reviews)

    print(score_df.round(3).to_string(index=False))

results = pd.DataFrame({
    "Review": test_reviews,
    "Expected": expected,
    "Predicted": predictions
})

results["Correct"] = (
    results["Expected"] == results["Predicted"]
)

print("\nIMPROVED SENTIMENT MODEL TEST")
print(results.to_string(index=False))

print("\nCorrect Predictions:",
      results["Correct"].sum(),
      "/", len(results))

print("Accuracy on these examples:",
      round(results["Correct"].mean() * 100, 2), "%")
