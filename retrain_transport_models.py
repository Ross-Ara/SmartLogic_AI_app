
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = Path("Cleaned datasets")
df = pd.read_csv(DATA_DIR / "smartlogix_transport_cleaned.csv")

features = [
    "quantity",
    "distance_km",
    "origin_hub",
    "origin_city",
    "destination_city",
    "destination_state"
]

target = "transport_mode"

# Avoid repeated order IDs appearing in both training and test sets.
if "order_id" in df.columns:
    df = df.drop_duplicates(subset=["order_id"]).copy()

# Clean numeric input features before training
df["distance_km"] = (
    df["distance_km"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"([-+]?\d*\.?\d+)")[0]
)

df["distance_km"] = pd.to_numeric(
    df["distance_km"], errors="coerce"
)

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
)

X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = ["quantity", "distance_km"]
categorical_features = [
    "origin_hub",
    "origin_city",
    "destination_city",
    "destination_state"
]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_test_processed)

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    })

results_df = pd.DataFrame(results).sort_values(
    "F1 Score", ascending=False
)

print("\nRETRAINED MODEL COMPARISON")
print(results_df.to_string(index=False))

# Majority-class baseline on the same test set
majority_class = y_train.value_counts().idxmax()
baseline_accuracy = (y_test == majority_class).mean()
print(f"\nMajority-class baseline accuracy: {baseline_accuracy:.4f}")

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print(f"\nSelected model: {best_name}")

# Save all files needed by the existing Streamlit dashboard.
joblib.dump(
    best_model,
    DATA_DIR / "transport_best_model.pkl"
)

joblib.dump(
    preprocessor,
    DATA_DIR / "transport_preprocessor.pkl"
)

joblib.dump(
    features,
    DATA_DIR / "transport_feature_columns.pkl"
)

joblib.dump(
    {
        "model_name": best_name,
        "feature_columns": features,
        "target_column": target,
        "preprocessor_file": "transport_preprocessor.pkl"
    },
    DATA_DIR / "transport_model_info.pkl"
)

results_df.to_csv(
    DATA_DIR / "transport_model_comparison.csv",
    index=False
)

print("\nModel files and comparison CSV saved successfully!")
