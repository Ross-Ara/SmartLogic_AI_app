
import pandas as pd
from pathlib import Path

neutral_examples = [
    "The parcel was delivered yesterday afternoon.",
    "I received my order today.",
    "The package arrived this morning.",
    "The delivery was completed yesterday.",
    "The parcel was handed over to me.",
    "I collected the package from the delivery agent.",
    "The order was delivered on Monday.",
    "My shipment arrived at the delivery address.",
    "The courier delivered the parcel today.",
    "I received a notification about the delivery.",
    "The package was delivered to my home.",
    "The delivery agent handed me the order.",
    "The shipment was received yesterday.",
    "My parcel arrived in the afternoon.",
    "The order was delivered at 3 PM."
]

new_df = pd.DataFrame({
    "review_text": neutral_examples,
    "sentiment_label": "Neutral"
})

output_path = Path(
    "Cleaned datasets/neutral_examples_additional.csv"
)

new_df.to_csv(output_path, index=False)

print("Additional Neutral Examples:", len(new_df))
print("Saved to:", output_path)
print(new_df.to_string(index=False))
