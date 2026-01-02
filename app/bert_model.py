from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_reviews(reviews):
    results = {"pos": 0, "neg": 0}

    for review in reviews:
        output = sentiment_pipeline(review[:512])[0]
        if output["label"] == "POSITIVE":
            results["pos"] += 1
        else:
            results["neg"] += 1

    return results
