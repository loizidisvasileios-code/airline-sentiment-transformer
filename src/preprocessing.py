"""Dataset loading and the minimal preparation a pretrained transformer needs.

The cleaning rules come from Exercise 1
"""

import html
import re

import pandas as pd

from src.config import RAW_DATA_PATH, TEXT_COL, LABEL_COL

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
CHAR_PATTERN = re.compile(r"[^a-z\s!?]")
WHITESPACE_PATTERN = re.compile(r"\s+")

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not", "wasn't": "was not",
    "aren't": "are not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "ain't": "is not", "i'm": "i am", "it's": "it is",
    "that's": "that is", "what's": "what is", "he's": "he is", "she's": "she is",
    "there's": "there is", "here's": "here is", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "i'd": "i would", "let's": "let us",
}

CONTRACTION_PATTERN = re.compile(r"\b(" + "|".join(CONTRACTIONS) + r")\b")


def load_raw_data():
    """Load the dataset"""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {RAW_DATA_PATH.name}")
    return df


def clean_text(text):
    """Clean a single tweet."""
    text = html.unescape(text)
    text = text.lower()
    text = text.replace("’", "'")
    text = CONTRACTION_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", " ")
    text = CHAR_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_dataset(df):
    """Keep the rows from exercise 1"""
    rows_in = len(df)

    df = df[[TEXT_COL, LABEL_COL]].copy()
    
    df[TEXT_COL] = df[TEXT_COL].apply(html.unescape)

    
    df["dedup_key"] = df[TEXT_COL].apply(clean_text)

    empty = (df["dedup_key"] == "").sum()
    df = df[df["dedup_key"] != ""]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["dedup_key"]).reset_index(drop=True)
    duplicates = before_dedup - len(df)

    df = df.drop(columns=["dedup_key"])

    print(f"Rows in             : {rows_in}")
    print(f"Empty after cleaning: {empty}")
    print(f"Duplicates removed  : {duplicates}")
    print(f"Rows out            : {len(df)}")
    return df
