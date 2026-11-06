
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

model_path = Path("Cleaned datasets/sentiment_model_v3.pkl")
model = joblib.load(model_path)

test_data = [
    ("The delivery was excellent and arrived on time.", "Positive"),
    ("My package arrived damaged and very late.", "Negative"),
    ("The parcel was delivered yesterday afternoon.", "Neutral"),
    ("The delivery was okay, nothing special.", "Neutral"),
    ("The package arrived on time but the packaging was damaged.", "Negative"),
    ("The driver was polite and the delivery was quick.", "Positive"),
    ("The product was fine but delivery was slow.", "Neutral"),
    ("I received my order today.", "Neutral"),
    ("The courier delivered the parcel today.", "Neutral"),
    ("The service was terrible and disappointing.", "Negative"),
    ("The driver was helpful and my parcel arrived safely.", "Positive"),
    ("My shipment was delayed and the box was broken.", "Negative")
]

reviews = [item[0] for item in test_data]
expected = [item[1] for item in test_data]

predicted = model.predict(reviews)

results = pd.DataFrame({
    "Review": reviews,
    "Expected": expected,
    "Predicted": predicted
})

results["Correct"] = results["Expected"] == results["Predicted"]

print("\nSENTIMENT MODEL V3 TEST RESULTS")
print(results.to_string(index=False))

print("\nCorrect Predictions:", results["Correct"].sum(), "/", len(results))
print("Accuracy:", round(accuracy_score(expected, predicted) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(expected, predicted, zero_division=0))

print("\nIncorrect Predictions:")
incorrect = results[~results["Correct"]]

if incorrect.empty:
    print("No incorrect predictions.")
else:
    print(incorrect.to_string(index=False))
