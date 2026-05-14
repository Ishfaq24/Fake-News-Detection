import pandas as pd
import nltk
from sklearn.utils import shuffle

nltk.download('stopwords')
from utils import combine_text_fields, normalize_text


def clean_text(text):
    return normalize_text(text)

def load_and_prepare_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 1
    true_df['label'] = 0

    df = pd.concat([fake_df, true_df], axis=0)

    df = shuffle(df, random_state=42)

    df.reset_index(drop=True, inplace=True)

    df['content'] = df.apply(combine_text_fields, axis=1)
    df['content'] = df['content'].astype(str).map(normalize_text)
    df = df[df['content'].str.len() > 20]
    df = df.drop_duplicates(subset=['content'])

    return df[['content', 'label']]