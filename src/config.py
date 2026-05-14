from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FAKE_DATA_PATH = str(BASE_DIR / "data" / "Fake.csv")
TRUE_DATA_PATH = str(BASE_DIR / "data" / "True.csv")

MODEL_PATH = str(BASE_DIR / "models" / "fake_news_model.pkl")
VECTORIZER_PATH = str(BASE_DIR / "models" / "vectorizer.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000