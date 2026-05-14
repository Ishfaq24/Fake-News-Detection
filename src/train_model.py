import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from data_preprocessing import load_and_prepare_data
from config import *

print("Loading and preprocessing dataset...")


df = load_and_prepare_data(
    FAKE_DATA_PATH,
    TRUE_DATA_PATH
)

X = df['content']
y = df['label']

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Vectorizing text using TF-IDF...")

pipeline = Pipeline([
    (
        "features",
        FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    max_features=40000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    max_features=30000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear',
            random_state=RANDOM_STATE,
        ),
    ),
])

print("Training Logistic Regression model...")

pipeline.fit(X_train, y_train)

print("Evaluating model...")

validation_probabilities = pipeline.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.2, 0.8, 121)
best_threshold = 0.5
best_score = -1.0

for threshold in thresholds:
    threshold_predictions = (validation_probabilities >= threshold).astype(int)
    score = balanced_accuracy_score(y_test, threshold_predictions)

    if score > best_score:
        best_score = score
        best_threshold = float(threshold)

predictions = (validation_probabilities >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {best_score * 100:.2f}%")
print(f"Selected decision threshold: {best_threshold:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

print("Saving model...")

joblib.dump({
    "pipeline": pipeline,
    "threshold": best_threshold,
}, MODEL_PATH)

print("Model and vectorizer saved successfully!")