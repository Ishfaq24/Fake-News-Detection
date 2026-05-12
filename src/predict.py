from pathlib import Path

import joblib
import re
import string

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r"https?://\S+|www\.\S+", '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text


model = joblib.load(MODEL_DIR / "fake_news_model.pkl")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


def predict_news(news, title=""):

    combined_news = f"{title} {news}".strip()
    cleaned_news = clean_text(combined_news)

    vectorized_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vectorized_input)

    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"

if __name__ == "__main__":

    print("\nFake News Detection System\n")

    user_title = input("Enter News Headline (optional):\n\n")
    user_input = input("Enter News Text:\n\n")

    result = predict_news(user_input, user_title)

    print(f"\nPrediction: {result}")