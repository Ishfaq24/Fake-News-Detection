from pathlib import Path

from flask import Flask, render_template, request
import joblib
import re
import string

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"


app = Flask(__name__)


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r"https?://\S+|www\.\S+", '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text


def predict_news(title, news):

    combined_news = f"{title} {news}".strip()
    cleaned_news = clean_text(combined_news)

    vectorized_input = vectorizer.transform([cleaned_news])
    result = model.predict(vectorized_input)

    if result[0] == 1:
        return "Fake News"
    return "Real News"


model = joblib.load(MODEL_DIR / "fake_news_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        title = request.form.get("title", "")
        news = request.form.get("news", "")

        prediction = predict_news(title, news)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)