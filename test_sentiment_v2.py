
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

folder = Path("Cleaned datasets")

# Load V2 model and its matching vectorizer
model = joblib.load(folder / "sentiment_model_v2.pkl")
vectorizer = joblib.load(folder / "sentiment_tfidf_v2.pkl")

# New review examples for testing
test_data = [
    ("The courier was friendly and the package came early.", "Positive"),
    ("Everything arrived in perfect condition.", "Positive"),
    ("I loved the delivery experience.", "Positive"),
    ("The delivery service was outstanding.", "Positive"),

    ("My order arrived broken and unusable.", "Negative"),
    ("The shipment was delayed for several days.", "Negative"),
    ("I am frustrated with the poor service.", "Negative"),
    ("The delivery experience was awful.", "Negative"),

    ("The package was received on Monday.", "Neutral"),
    ("The driver left the parcel at the entrance.", "Neutral"),
    ("I collected the shipment from the office.", "Neutral"),
    ("The order was handed over this afternoon.", "Neutral")
]

test_df = pd.DataFrame(
    test_data,
    columns=["Review", "Expected"]
)

# Transform reviews into TF-IDF features
X_test = vectorizer.transform(test_df["Review"])

# Predict sentiment
test_df["Predicted"] = model.predict(X_test)

# Compare predictions
test_df["Correct"] = (
    test_df["Expected"] == test_df["Predicted"]
)

print("\nSENTIMENT MODEL V2 TEST RESULTS")
print(test_df.to_string(index=False))

accuracy = accuracy_score(
    test_df["Expected"],
    test_df["Predicted"]
)


print("\nINCORRECT PREDICTIONS")
print("=" * 60)

incorrect_df = test_df[
    test_df["Correct"] == False
]

print(
    incorrect_df[
        ["Review", "Expected", "Predicted"]
    ].to_string(index=False)
)

print("\nTotal incorrect:", len(incorrect_df))

print("\nPREDICTION DISTRIBUTION")
print(test_df["Predicted"].value_counts())
