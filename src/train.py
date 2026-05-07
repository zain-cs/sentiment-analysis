import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/reviews.csv')
print(f"Loaded {len(df)} reviews")
print(df['sentiment'].value_counts(), "\n")

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. TF-IDF Vectorizer ──────────────────────────────────────────────────────
# Converts raw text into numbers the model can understand
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),       # single words + pairs of words
    stop_words=stop_words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ── 3. Train Model ────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)
cv    = cross_val_score(model, vectorizer.transform(X), y, cv=5, scoring='accuracy').mean()

print("─" * 40)
print("Logistic Regression + TF-IDF")
print(f"  CV Accuracy : {cv:.4f}")
print("─" * 40)
print(classification_report(y_test, preds))

# ── 4. Save Model & Vectorizer ────────────────────────────────────────────────
pickle.dump(model,      open('models/sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
print("✅ Model and vectorizer saved!")

# ── 5. Plot Results ───────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sentiment Analysis — Results', fontsize=15, fontweight='bold')

# Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_test, preds, labels=['positive', 'neutral', 'negative'])
ConfusionMatrixDisplay(cm, display_labels=['Positive', 'Neutral', 'Negative']).plot(
    ax=ax, colorbar=False, cmap='Blues'
)
ax.set_title('Confusion Matrix')

# Sentiment Distribution
ax = axes[1]
df['sentiment'].value_counts().plot(
    kind='bar', ax=ax,
    color=['steelblue', 'salmon', 'mediumseagreen'],
    edgecolor='white'
)
ax.set(title='Sentiment Distribution', xlabel='Sentiment', ylabel='Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('outputs/results.png', dpi=150, bbox_inches='tight')
print("📊 Plot saved to outputs/results.png")
plt.show()