import pandas as pd
import re
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample dataset (replace with real customer data)
data = {
    "text": [
        "The product is amazing and works perfectly",
        "Very bad experience, totally disappointed",
        "Customer support was okay, nothing special",
        "I love this service",
        "Worst purchase ever",
        "Not bad, could be better"
    ],
    "sentiment": ["positive", "negative", "neutral", "positive", "negative", "neutral"]
}

df = pd.DataFrame(data)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

X = df["cleaned_text"]
y = df["sentiment"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully")
