from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analyze_reviews(reviews):
    results = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "labeled_reviews": []
    }

    for review in reviews:
        if not review.strip():
            continue

        score = sia.polarity_scores(review)["compound"]

        if score >= 0.05:
            sentiment = "Positive"
            results["positive"] += 1
        elif score <= -0.05:
            sentiment = "Negative"
            results["negative"] += 1
        else:
            sentiment = "Neutral"
            results["neutral"] += 1

        results["labeled_reviews"].append({
            "review": review,
            "sentiment": sentiment
        })

    return results
