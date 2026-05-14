from config import MODEL_PATH, VECTORIZER_PATH
from data_preprocessing import clean_text
from utils import combine_text_fields, load_model_bundle, predict_from_bundle


LABELS = {
    0: "Real News",
    1: "Fake News",
}

INFERENCE_THRESHOLD = 0.65


bundle = load_model_bundle(MODEL_PATH, VECTORIZER_PATH)


def predict_news(news, title=""):
    combined_news = combine_text_fields({"title": title, "text": news})
    cleaned_news = clean_text(combined_news)
    _, score = predict_from_bundle(bundle, cleaned_text=cleaned_news)
    prediction = int(float(score) >= INFERENCE_THRESHOLD)
    return LABELS[prediction]


def predict_news_with_details(news, title=""):
    combined_news = combine_text_fields({"title": title, "text": news})
    cleaned_news = clean_text(combined_news)
    _, score = predict_from_bundle(bundle, cleaned_news)

    threshold = INFERENCE_THRESHOLD
    score_value = float(score)
    prediction = int(score_value >= threshold)
    label = LABELS[prediction]

    return {
        "label": label,
        "prediction": prediction,
        "score": score_value,
        "threshold": threshold,
        "uncertain": False,
    }


if __name__ == "__main__":
    print("\nFake News Detection System")

    while True:
        print("\nEnter News Text:")

        news = input()

        if news.lower() == "exit":
            break

        details = predict_news_with_details(news)
        prediction = details["label"]
        score = details["score"]
        threshold = details["threshold"]

        print(f"\n{prediction.upper()}")
        print(f"Fake Probability: {score:.4f}")
        print(f"Threshold: {threshold:.4f}")