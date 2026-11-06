
import pandas as pd
from pathlib import Path

folder = Path("Cleaned datasets")

# Additional examples for future model training
examples = [
    ("The delivery was excellent and arrived on time.", "Positive"),
    ("The courier was polite and very helpful.", "Positive"),
    ("My package arrived quickly and safely.", "Positive"),
    ("The service was fantastic and reliable.", "Positive"),
    ("I am very happy with the delivery service.", "Positive"),

    ("The service was terrible and disappointing.", "Negative"),
    ("The package arrived damaged.", "Negative"),
    ("My delivery was extremely late.", "Negative"),
    ("The courier was rude and unprofessional.", "Negative"),
    ("I am unhappy with the delivery experience.", "Negative"),

    ("The parcel was delivered yesterday afternoon.", "Neutral"),
    ("I received my order today.", "Neutral"),
    ("The courier delivered the parcel today.", "Neutral"),
    ("My package was delivered this morning.", "Neutral"),
    ("The order was delivered at 3 PM.", "Neutral"),
]

df = pd.DataFrame(
    examples,
    columns=["review_text", "sentiment_label"]
)

output_path = folder / "sentiment_additional_examples.csv"

df.to_csv(output_path, index=False)

print("STEP 5.9.11 COMPLETED")
print("Additional examples:", len(df))
print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())
print("\nSaved to:", output_path)
