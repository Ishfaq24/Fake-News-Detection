import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_preprocessing import load_and_prepare_data


print("Loading dataset...")

data = load_and_prepare_data()

x = data["text"]
y = data["label"]

print("Splitting dataset...")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)

print("Vectorizing text...")

vectorizer = TfidfVectorizer()

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

print("Training model...")

model = LogisticRegression()

model.fit(xv_train, y_train)

print("Predicting...")

predictions = model.predict(xv_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nSaving model...")

joblib.dump(model, "../models/fake_news_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")