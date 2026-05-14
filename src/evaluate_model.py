import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_prepare_data
from config import *
from utils import load_model_bundle


df = load_and_prepare_data(
    FAKE_DATA_PATH,
    TRUE_DATA_PATH
)

X = df['content']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

bundle = load_model_bundle(MODEL_PATH, VECTORIZER_PATH)

if bundle.get("legacy", False):
    vectorizer = bundle.get("vectorizer")
    model = bundle["model"]
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
else:
    pipeline = bundle["pipeline"]
    threshold = bundle.get("threshold", 0.5)
    validation_probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (validation_probabilities >= threshold).astype(int)

print("Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Threshold:")
print(bundle.get("threshold", 0.5))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()