from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


model_path = os.path.join(BASE_DIR, "../models/fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "../models/tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        cleaned_news = clean_text(news)

        input_data = pd.DataFrame(
            {"text": [cleaned_news]}
        )

        vectorized_input = vectorizer.transform(
            input_data["text"]
        )

        result = model.predict(vectorized_input)

        if result[0] == 0:
            prediction = "Fake News"
        else:
            prediction = "Real News"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)