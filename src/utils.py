from pathlib import Path

import joblib
import pandas as pd

import re


TEXT_COLUMNS = ("title", "subject", "text")


def normalize_text(text):
	text = "" if text is None else str(text)
	text = text.lower()
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"<.*?>", " ", text)
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def combine_text_fields(row, columns=TEXT_COLUMNS):
	parts = []

	for column in columns:
		if column in row and pd.notna(row[column]):
			value = str(row[column]).strip()
			if value:
				parts.append(value)

	return " ".join(parts)


def load_model_bundle(model_path, vectorizer_path=None):
	artifact = joblib.load(model_path)

	if isinstance(artifact, dict) and "pipeline" in artifact:
		return {
			"pipeline": artifact["pipeline"],
			"threshold": artifact.get("threshold", 0.5),
			"legacy": False,
		}

	bundle = {
		"model": artifact,
		"threshold": 0.5,
		"legacy": True,
	}

	if vectorizer_path is not None and Path(vectorizer_path).exists():
		bundle["vectorizer"] = joblib.load(vectorizer_path)

	return bundle


def predict_from_bundle(bundle, cleaned_text):
	if not bundle.get("legacy", False):
		pipeline = bundle["pipeline"]
		threshold = bundle.get("threshold", 0.5)
		fake_probability = pipeline.predict_proba([cleaned_text])[0][1]
		prediction = int(fake_probability >= threshold)
		return prediction, fake_probability

	model = bundle["model"]
	vectorizer = bundle.get("vectorizer")

	if vectorizer is not None:
		features = vectorizer.transform([cleaned_text])
	else:
		features = [cleaned_text]

	prediction = int(model.predict(features)[0])

	if hasattr(model, "decision_function"):
		score = float(model.decision_function(features)[0])
	else:
		score = float(prediction)

	return prediction, score
