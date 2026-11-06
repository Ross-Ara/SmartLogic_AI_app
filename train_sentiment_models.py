
import pandas as pd
import joblib
from pathlib import Path

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

data_dir = Path("Cleaned datasets")

# Load prepared training and testing data
X_train, X_test, y_train, y_test = joblib.load(
    data_dir / "sentiment_train_test.pkl"
)

# Define classification models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n========== {name} ==========")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    filename = {
        "Naive Bayes": "sentiment_naive_bayes.pkl",
        "Logistic Regression": "sentiment_logistic_regression.pkl",
        "Linear SVM": "sentiment_linear_svm.pkl"
    }[name]

    joblib.dump(model, data_dir / filename)

# Compare model performance
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    "F1 Score", ascending=False
)

comparison.to_csv(
    data_dir / "sentiment_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nSTEP 5.4 COMPLETED")
print("All sentiment models and comparison results saved!")
