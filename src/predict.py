import joblib
import re
import string
import pandas as pd


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r"https?://\S+|www\.\S+", '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text

model = joblib.load("../models/fake_news_model.pkl")

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")


def predict_news(news):

    cleaned_news = clean_text(news)

    input_data = pd.DataFrame(
        {"text": [cleaned_news]}
    )

    vectorized_input = vectorizer.transform(
        input_data["text"]
    )

    prediction = model.predict(vectorized_input)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"


if __name__ == "__main__":

    print("\nFake News Detection System\n")

    user_input = input("Enter News Text:\n\n")

    result = predict_news(user_input)

    print(f"\nPrediction: {result}")