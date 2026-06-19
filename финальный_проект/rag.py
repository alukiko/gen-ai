from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schema import RetrievedExample


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Tweets" / "Twitterdatainsheets.csv"


def compute_engagement_score(frame: pd.DataFrame) -> pd.Series:
    likes = pd.to_numeric(frame["Likes"], errors="coerce").fillna(0)
    retweets = pd.to_numeric(frame["RetweetCount"], errors="coerce").fillna(0)
    reach = pd.to_numeric(frame["Reach"], errors="coerce").replace(0, np.nan)
    raw = (np.log1p(likes) + 2 * np.log1p(retweets)) / (np.log1p(reach).fillna(1) ** 0.5)
    scaled = np.log1p(raw)
    p1 = scaled.quantile(0.01)
    p99 = scaled.quantile(0.99)
    return (10 * (scaled - p1) / (p99 - p1)).clip(0, 10).fillna(0)


@lru_cache(maxsize=1)
def _load_index():
    columns = ["TweetID", " text", " Weekday", " Hour", " Reach", " RetweetCount", " Likes"]
    df = pd.read_csv(DATA_PATH, usecols=columns, low_memory=False).dropna(subset=[" text"]).head(8000).copy()
    df.columns = [c.strip() for c in df.columns]
    df["Engagement_score"] = compute_engagement_score(df)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=6000)
    matrix = vectorizer.fit_transform(df["text"].astype(str))
    return df, vectorizer, matrix


def retrieve_examples(query: str, k: int = 4) -> list[RetrievedExample]:
    df, vectorizer, matrix = _load_index()
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).ravel()
    top = similarities.argsort()[::-1][: max(k * 3, k)]
    examples: list[RetrievedExample] = []
    for idx in top:
        row = df.iloc[int(idx)]
        if len(examples) >= k:
            break
        examples.append(
            RetrievedExample(
                tweet_id=str(row["TweetID"]),
                text=str(row["text"])[:280],
                weekday=str(row["Weekday"]),
                hour=int(row["Hour"]),
                engagement_score=round(float(row["Engagement_score"]), 2),
                similarity=round(float(similarities[idx]), 3),
            )
        )
    return examples
