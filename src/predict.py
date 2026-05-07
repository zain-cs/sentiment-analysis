import pickle

# Load model and vectorizer
model      = pickle.load(open('models/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# --- Test reviews ---
reviews = [
    "This product is absolutely amazing, best purchase ever!",
    "Terrible quality, broke after one day. Total waste of money.",
    "It is okay, nothing special, does the job fine.",
    "Exceeded all my expectations, highly recommend to everyone!",
    "Very disappointed, looks nothing like the pictures.",
    "Average product, not bad but not great either.",
]

# --- Predict ---
vectors = vectorizer.transform(reviews)
preds   = model.predict(vectors)
probs   = model.predict_proba(vectors)

emoji = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}

print("\n📋 Sentiment Prediction Results")
print("─" * 60)
for review, pred, prob in zip(reviews, preds, probs):
    confidence = max(prob)
    icon = emoji[pred]
    print(f"{icon}  {pred.upper():<10} ({confidence:.0%} confident)")
    print(f"    → \"{review[:55]}...\"" if len(review) > 55 else f"    → \"{review}\"")
    print()