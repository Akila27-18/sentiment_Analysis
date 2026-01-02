from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import os

app = FastAPI(title="Sentiment Analysis API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

class TextInput(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.post("/predict")
def predict(data: TextInput):
    cleaned = clean_text(data.text)
    vector = vectorizer.transform([cleaned])
    sentiment = model.predict(vector)[0]
    return {"sentiment": sentiment}
uvicorn app.main:app --reload
