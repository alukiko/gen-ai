from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from transformers import BertModel, BertTokenizer

from schema import Prediction, Weekday


PROJECT_ROOT = Path(__file__).resolve().parent
TWEETS_DIR = PROJECT_ROOT / "Tweets"


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s#]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


@lru_cache(maxsize=1)
def _load_components():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    weekday_ohe = joblib.load(TWEETS_DIR / "weekday_ohe.pkl")
    rf_model = joblib.load(TWEETS_DIR / "random_forest_model.pkl")
    return tokenizer, bert_model, weekday_ohe, rf_model


def _bert_vector(text: str) -> np.ndarray:
    tokenizer, bert_model, _, _ = _load_components()
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)


def predict_engagement(text: str, weekday: Weekday, hour: int) -> Prediction:
    _, _, weekday_ohe, rf_model = _load_components()
    bert_vector = _bert_vector(text)
    weekday_vector = weekday_ohe.transform(pd.DataFrame({"Weekday": [weekday]}))
    hour_vector = np.array(
        [[np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]]
    )
    features = np.hstack([bert_vector, weekday_vector, hour_vector])
    score = float(np.clip(rf_model.predict(features)[0], 0, 10))
    return Prediction(
        engagement_score=round(score, 2),
        weekday=weekday,
        hour=hour,
        text_length=len(text),
    )
