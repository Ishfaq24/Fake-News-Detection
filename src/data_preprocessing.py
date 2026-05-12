from pathlib import Path

import pandas as pd
import re
import string

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r"https?://\S+|www\.\S+", '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text


def load_and_prepare_data():

    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")

    fake["label"] = 1
    true["label"] = 0

    data = pd.concat([fake, true], ignore_index=True)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    data["text"] = (data["title"].fillna("") + " " + data["text"].fillna("")).apply(clean_text)

    return data