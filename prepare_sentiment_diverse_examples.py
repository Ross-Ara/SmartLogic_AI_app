
import pandas as pd
from pathlib import Path

folder = Path("Cleaned datasets")
folder.mkdir(exist_ok=True)

examples = {
    "Positive": [
        "I loved the delivery experience.",
        "Everything arrived in perfect condition.",
        "The driver was friendly and professional.",
        "The parcel arrived earlier than expected.",
        "Excellent service and very quick delivery.",
        "My order arrived safely without damage.",
        "The courier handled my package carefully.",
        "I am extremely happy with the delivery.",
        "The shipment arrived on time and intact.",
        "Outstanding delivery service from start to finish."
    ],
    "Negative": [
        "My order arrived broken and unusable.",
        "The service was terrible and disappointing.",
        "The parcel was damaged during transportation.",
        "I hated the delivery experience.",
        "The courier was rude and unhelpful.",
        "The shipment arrived several days late.",
        "The package was missing important items.",
        "The driver left my order at the wrong address.",
        "Very poor service and unacceptable delays.",
        "The delivery was slow and the box was damaged."
    ],
    "Neutral": [
        "The package was received on Monday.",
        "The parcel was delivered yesterday afternoon.",
        "I received my order today.",
        "The courier delivered the parcel today.",
        "The driver left the parcel at the entrance.",
        "I collected the shipment from the office.",
        "The order was delivered at three PM.",
        "The package arrived at the delivery address.",
        "The delivery agent handed me the parcel.",
        "The shipment was received this morning."
    ]
}

rows = [
    {"review_text": review, "sentiment_label": label}
    for label, reviews in examples.items()
    for review in reviews
]

df = pd.DataFrame(rows)

output_path = folder / "sentiment_diverse_examples.csv"
df.to_csv(output_path, index=False)

print("STEP 5.9.17 COMPLETED")
print("Additional diverse examples:", len(df))
print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())
print("\nSaved to:", output_path)
