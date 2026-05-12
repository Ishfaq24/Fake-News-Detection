import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from predict import predict_news

test_file = Path(__file__).resolve().parent / "test_data.csv"
test_data = pd.read_csv(test_file)

print("\n" + "="*100)
print("FAKE NEWS DETECTION MODEL - TEST RESULTS")
print("="*100 + "\n")

for idx, row in test_data.iterrows():
    headline = row["headline"]
    text = row["text"]
    prediction = predict_news(text, headline)
    
    print(f"Test {idx + 1}")
    print(f"Headline: {headline}")
    print(f"Text: {text[:80]}...")
    print(f"Prediction: {prediction}")
    print("-" * 100)

print("\n")
