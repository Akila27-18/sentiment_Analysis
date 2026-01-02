from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

from app.sentiment_model import analyze_reviews

app = FastAPI(title="Customer Sentiment Analytics")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

analysis_result = {}
download_data = []


from wordcloud import WordCloud
import os

def generate_wordclouds(labeled_reviews):
    wordcloud_paths = {}

    for sentiment in ["Positive", "Negative"]:
        texts = [
            r["review"]
            for r in labeled_reviews
            if r["sentiment"] == sentiment and len(r["review"].split()) > 2
        ]

        if not texts:
            continue  # ← CRITICAL FIX

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(" ".join(texts))

        path = f"app/static/wordcloud_{sentiment.lower()}.png"
        wc.to_file(path)

        wordcloud_paths[sentiment.lower()] = f"/static/wordcloud_{sentiment.lower()}.png"

    return wordcloud_paths



@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    global analysis_result

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "Unsupported format"}

    reviews = df.iloc[:, 0].dropna().astype(str).tolist()

    result = analyze_reviews(reviews)

    total = result["positive"] + result["negative"] + result["neutral"]

    top_words = extract_top_keywords(reviews)
    wordclouds = generate_wordclouds(result["labeled_reviews"])

    analysis_result = {
        **result,
        "total": total,
        "positive_percent": round((result["positive"] / total) * 100, 2),
        "negative_percent": round((result["negative"] / total) * 100, 2),
        "neutral_percent": 0,
        "top_words": top_words,
        "wordclouds": wordclouds
    }

    return RedirectResponse("/dashboard", status_code=303)



@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if not analysis_result:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "data": analysis_result}
    )


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

@app.get("/download-pdf")
def download_pdf():
    file_path = "Sentiment_Analysis_Report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Customer Sentiment Analysis Report", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"<b>Total Reviews:</b> {analysis_result['total']}", styles["Normal"]))
    content.append(Paragraph(
        f"<b>Positive:</b> {analysis_result['positive']} ({analysis_result['positive_percent']}%)",
        styles["Normal"]
    ))
    content.append(Paragraph(
        f"<b>Negative:</b> {analysis_result['negative']} ({analysis_result['negative_percent']}%)",
        styles["Normal"]
    ))
    content.append(Paragraph(
        f"<b>Neutral:</b> {analysis_result['neutral']} ({analysis_result['neutral_percent']}%)",
        styles["Normal"]
    ))

    content.append(Spacer(1, 20))
    content.append(Paragraph("<b>Business Insights</b>", styles["Heading2"]))

    if analysis_result["positive_percent"] > 60:
        content.append(Paragraph("✔ Customers are highly satisfied.", styles["Normal"]))
    if analysis_result["negative_percent"] > 30:
        content.append(Paragraph("⚠ High negative feedback detected.", styles["Normal"]))
    if analysis_result["neutral_percent"] > 40:
        content.append(Paragraph("ℹ Many neutral reviews – engage customers.", styles["Normal"]))

    doc.build(content)

    return FileResponse(file_path, filename=file_path)

from collections import Counter
import re

def extract_top_keywords(reviews, top_n=10):
    text = " ".join(reviews).lower()
    words = re.findall(r'\b[a-z]{4,}\b', text)
    return Counter(words).most_common(top_n)

