import pandas as pd
import random

random.seed(42)

# --- Positive reviews ---
positive = [
    "This product is absolutely amazing, I love it!",
    "Exceeded my expectations, highly recommend to everyone.",
    "Best purchase I have ever made, works perfectly.",
    "Fantastic quality, arrived on time, very happy.",
    "Outstanding service and the product is top notch.",
    "Really impressed with the build quality and design.",
    "Works exactly as described, very satisfied customer.",
    "Great value for money, will definitely buy again.",
    "Superb product, my whole family loves it.",
    "Five stars, no complaints at all, brilliant!",
    "Delivery was fast and product quality is excellent.",
    "So happy with this purchase, highly recommended.",
    "Perfect in every way, exactly what I needed.",
    "Amazing product at a very affordable price.",
    "Loved it from the moment I opened the box.",
]

# --- Negative reviews ---
negative = [
    "Terrible product, broke after just two days.",
    "Complete waste of money, very disappointed.",
    "Worst purchase ever, does not work at all.",
    "Very poor quality, nothing like the description.",
    "Stopped working after one week, avoid this.",
    "Arrived damaged and customer service was useless.",
    "Absolute rubbish, do not buy this product.",
    "Horrible experience, will never shop here again.",
    "Product looks nothing like the pictures shown.",
    "Cheap materials and fell apart immediately.",
    "Very slow delivery and product was broken.",
    "Do not waste your money on this garbage.",
    "Extremely disappointed, worst quality ever seen.",
    "Returned it immediately, total disappointment.",
    "Faulty product and no response from support.",
]

# --- Neutral reviews ---
neutral = [
    "Product is okay, nothing special about it.",
    "It works fine but nothing to rave about.",
    "Average quality, does what it is supposed to.",
    "Delivery was on time, product is acceptable.",
    "Not bad but not great either, just okay.",
    "Meets basic expectations but could be better.",
    "It is fine for the price I paid.",
    "Does the job but I expected a bit more.",
    "Reasonable product, no major issues so far.",
    "Fairly standard item, nothing stood out to me.",
    "It is decent enough for everyday use.",
    "Works as expected, neither good nor bad.",
    "Acceptable product for the price point.",
    "Neutral experience overall, just an average item.",
    "Product arrived fine and works as described.",
]

# --- Build dataset ---
reviews = (
    [(r, 'positive') for r in positive] +
    [(r, 'negative') for r in negative] +
    [(r, 'neutral')  for r in neutral]
)

# Expand to 300 rows by slightly varying reviews
expanded = []
for review, label in reviews:
    expanded.append((review, label))
    expanded.append((review + " Would recommend.", label))
    expanded.append(("Overall: " + review, label))
    expanded.append((review + " Very honest opinion.", label))
    expanded.append((review + " Sharing my experience.", label))
    expanded.append((review + " Hope this helps others.", label))

random.shuffle(expanded)

df = pd.DataFrame(expanded[:300], columns=['review', 'sentiment'])
df.to_csv('data/reviews.csv', index=False)
print(f"Dataset created! {len(df)} reviews")
print(df['sentiment'].value_counts())
print(df.head())