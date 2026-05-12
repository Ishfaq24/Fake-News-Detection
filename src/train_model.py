from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_preprocessing import load_and_prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

data = load_and_prepare_data()

x = data["text"]
y = data["label"]

print("Splitting dataset...")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_df=0.95,
    min_df=2
)

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(xv_train, y_train)

print("Predicting...")

predictions = model.predict(xv_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nSaving model...")

joblib.dump(model, MODEL_DIR / "fake_news_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")