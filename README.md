<div align="center">

# 💬 Sentiment Analysis on Customer Reviews

<p>Instantly classify customer reviews as <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong> — with confidence scores.</p>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat-square&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

</div>

---

## 📌 Overview

Businesses receive thousands of customer reviews every day — reading them manually is impossible. This project builds an end-to-end NLP pipeline that automatically detects sentiment from raw review text, giving businesses instant insight into how their customers feel.

> **Core question: "What are customers actually feeling — and how strongly?"**

---

## ✨ Live Example

```
😊  POSITIVE   (74% confident)
    → "This product is absolutely amazing, best purchase ever!"

😠  NEGATIVE   (78% confident)
    → "Terrible quality, broke after one day. Total waste of money."

😐  NEUTRAL    (83% confident)
    → "It is okay, nothing special, does the job fine."
```

---

## 📊 Results

| Model | CV Accuracy | Sentiment Classes |
|---|---|---|
| ✅ Logistic Regression + TF-IDF | **100%** | Positive · Neutral · Negative |

![Results](outputs/results.png)

---

## 🔍 How It Works

```
  Raw Review Text
        │
        ▼
  Stopword Removal (NLTK)
        │
        ▼
  TF-IDF Vectorization
  (text → numerical features, bigrams)
        │
        ▼
  Logistic Regression Classifier
        │
        ▼
  Sentiment Label + Confidence Score
```

---

## 🗂️ Project Structure

```
📦 sentiment-analysis
 ┣ 📂 data
 ┃ ┗ 📄 reviews.csv              ← 270 labeled customer reviews
 ┣ 📂 src
 ┃ ┣ 🐍 generate_data.py         ← Generates balanced review dataset
 ┃ ┣ 🐍 train.py                 ← TF-IDF + model training pipeline
 ┃ ┗ 🐍 predict.py               ← Run predictions on new reviews
 ┣ 📂 models
 ┃ ┣ 📄 sentiment_model.pkl      ← Saved classifier
 ┃ ┗ 📄 vectorizer.pkl           ← Saved TF-IDF vectorizer
 ┣ 📂 outputs
 ┃ ┗ 🖼️ results.png              ← Confusion matrix + class distribution
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## 🚀 Quickstart

**1. Clone the repository**
```bash
git clone https://github.com/zain-cs/sentiment-analysis.git
cd sentiment-analysis
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the full pipeline**
```bash
python src/generate_data.py   # Generate dataset
python src/train.py           # Train and evaluate
python src/predict.py         # Predict on new reviews
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| NLTK | Stopword removal and text preprocessing |
| scikit-learn | TF-IDF vectorization and Logistic Regression |
| matplotlib & seaborn | Confusion matrix and distribution plots |
| pickle | Model and vectorizer persistence |

---

## 🗺️ Roadmap

- [x] Balanced dataset with 270 labeled reviews
- [x] TF-IDF vectorization with unigrams and bigrams
- [x] Logistic Regression with confidence scores
- [x] Confusion matrix and sentiment distribution plots
- [ ] Upgrade to HuggingFace transformer (BERT)
- [ ] Streamlit web app for live predictions
- [ ] Train on real Amazon or Yelp review dataset
- [ ] REST API deployment with FastAPI

---

## 👤 Author

**Zain** — [@zain-cs](https://github.com/zain-cs)

> Open to freelance ML and NLP projects.

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and build on it.
