# Fake News Detection (Production-Ready Guide)

Production-focused binary text classification system for news credibility screening.

Classification policy in production:

- Fake News when fake probability is greater than or equal to 0.65
- Real News when fake probability is less than 0.65

This repository contains:

- Model training and evaluation pipeline
- Deterministic inference logic with fixed thresholding
- CLI and Flask web interface
- Basic integration-style test workflow

## 1. System Overview

The pipeline is organized into four stages:

1. Data ingestion and normalization
2. Feature extraction with TF-IDF (word and character signals)
3. Logistic Regression scoring
4. Threshold-based decisioning

Core behavior:

- Model outputs fake probability for each input
- Decision threshold is enforced at inference time as 0.65
- Input text can be headline plus body, or body only

## 2. Repository Layout

```text
fakeNewsPred/
	app/                     # Flask app
	data/                    # Training datasets
	models/                  # Serialized model artifacts
	notebooks/               # Experiments
	src/                     # Training, evaluation, prediction code
		config.py
		data_preprocessing.py
		evaluate_model.py
		predict.py
		train_model.py
		utils.py
	tests/
		run_tests.py
		test_data.csv
	requirements.txt
	README.md
```

## 3. Environment and Dependencies

Prerequisites:

- Windows, macOS, or Linux
- Python 3.10+
- Pip

Install dependencies:

```powershell
pip install -r requirements.txt
```

Recommended:

- Use a virtual environment per project
- Pin dependency versions before production releases

## 4. Data Contract

Expected CSV inputs:

- data/Fake.csv
- data/True.csv

Expected columns:

- title
- text
- subject
- date

Minimum required content for learning:

- Textual fields that produce normalized content longer than 20 characters

Preprocessing behavior:

- Lowercasing
- URL and markup cleanup
- Alphanumeric normalization
- Whitespace normalization
- Duplicate content removal

## 5. Training

Run from repository root:

```powershell
python src/train_model.py
```

What training does:

- Loads and prepares data
- Splits train and validation sets
- Fits feature + classifier pipeline
- Reports metrics
- Saves model artifact to models/fake_news_model.pkl

Operational note:

- Retrain after any data update, threshold policy update, or preprocessing change

## 6. Evaluation

```powershell
python src/evaluate_model.py
```

Outputs:

- Accuracy
- Classification report
- Effective threshold context in report output

Production recommendation:

- Track precision, recall, F1 for both classes
- Monitor false positive rate on recent real-world traffic

## 7. Inference (CLI)

```powershell
python src/predict.py
```

Runtime output includes:

- Label
- Fake probability
- Threshold used

Decision policy:

- fake probability greater than or equal to 0.65 => Fake News
- fake probability less than 0.65 => Real News

## 8. Web Serving

Start Flask app:

```powershell
python app/app.py
```

Default local URL:

- http://127.0.0.1:5000/

Production hardening checklist:

- Disable debug mode
- Run behind a production WSGI server
- Add request size limits
- Add input sanitation and rate limiting
- Add structured logging and error tracing

## 9. Production Test Suite

### Test
## Real News 
```
Several European countries announced new renewable energy projects aimed at reducing greenhouse gas emissions and improving long-term environmental sustainability across industrial sectors.
```

```Google introduced several artificial intelligence features during its annual developer conference, including improvements to Android security, battery optimization, and personalized search experiences powered by machine learning systems.
```


### 9.1 Batch Regression Test

```powershell
python tests/run_tests.py
```

Purpose:

- Validates end-to-end prediction behavior on the packaged dataset

### 9.2 Threshold Rule Validation

Use strict assertions to ensure production decision policy is never violated:

```powershell
@'
import sys
sys.path.insert(0, 'src')
from predict import predict_news_with_details

samples = [
		"The Indian Space Research Organisation successfully completed testing for its reusable launch vehicle technology during a recent experimental mission.",
		"Scientists discovered a hidden portal beneath the Pacific Ocean that allows instant travel between planets.",
		"The government approved a comprehensive infrastructure bill after months of negotiations in parliament.",
]

for idx, text in enumerate(samples, 1):
		result = predict_news_with_details(text)
		expected = "Fake News" if result["score"] >= 0.65 else "Real News"
		assert result["label"] == expected, f"Mismatch on sample {idx}: {result}"
		print(f"Sample {idx} OK | label={result['label']} | fake_prob={result['score']:.4f} | threshold={result['threshold']:.2f}")

print("Threshold policy verified.")
'@ | Set-Content _temp_threshold_policy_test.py

python _temp_threshold_policy_test.py

Remove-Item _temp_threshold_policy_test.py
```

### 9.3 Smoke Test for Service Startup

```powershell
python app/app.py
```

Validate manually:

- Server starts without exceptions
- Root page loads
- Single prediction request returns label

## 10. Release and Deployment Checklist

Before release:

1. Run training and evaluation
2. Run batch test and threshold assertions
3. Confirm artifact exists in models directory
4. Record model version, data snapshot date, and metrics
5. Tag release in source control

Deployment minimums:

1. Immutable artifact deployment
2. Environment variable based configuration
3. Health checks and startup probes
4. Access logs and error logs
5. Rollback plan to last known-good artifact

## 11. Monitoring and Drift Management

Track in production:

- Volume by predicted class
- Probability distribution drift
- Human-reviewed precision on sampled predictions
- Topic-wise error concentrations

Retraining triggers:

- Sustained precision drop
- Significant topic/domain shift
- New source language or writing-style changes

## 12. Security and Compliance Notes

- Do not treat model output as legal fact
- Keep raw user input logging minimal and privacy-safe
- Sanitize and bound all externally submitted text
- Apply standard dependency vulnerability scanning

## 13. Known Limits

- High offline accuracy does not guarantee universal real-world performance
- Domain shift can degrade quality quickly
- Model is assistive, not authoritative

## 14. Quick Command Reference

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train:

```powershell
python src/train_model.py
```

Evaluate:

```powershell
python src/evaluate_model.py
```

CLI inference:

```powershell
python src/predict.py
```

Batch tests:

```powershell
python tests/run_tests.py
```

Run web app:

```powershell
python app/app.py
```
