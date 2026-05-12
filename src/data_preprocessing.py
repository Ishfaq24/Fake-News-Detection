import pandas as pd
import re
import string


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

    fake = pd.read_csv("../data/raw/Fake.csv")
    true = pd.read_csv("../data/raw/True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])

    data = data.sample(frac=1)

    data.reset_index(inplace=True)

    data.drop(["index"], axis=1, inplace=True)

    data["text"] = data["text"].apply(clean_text)

    return data