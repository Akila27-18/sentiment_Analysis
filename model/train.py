import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "aclImdb", "train")

def load_data(path):
    texts, labels = [], []
    for label in ["pos", "neg"]:
        folder = os.path.join(path, label)
        for file in os.listdir(folder)[:5000]:
            with open(os.path.join(folder, file), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

X, y = load_data(DATASET_PATH)
X = [clean_text(t) for t in X]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

joblib.dump(model, os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

print("✅ Model trained successfully with real IMDb data")
