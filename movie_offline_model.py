import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DATA_ROOT = Path(r"E:\anime-recommender\movie data")

_MOVIE_DF = None
_MOVIE_MATRIX = None
_MOVIE_VECTORIZER = None

_TV_DF = None
_TV_MATRIX = None
_TV_VECTORIZER = None


def _load_provider_xls(fname: str, provider_name: str) -> pd.DataFrame:
    path = DATA_ROOT / fname
    df = pd.read_excel(path)

    df = df.copy()
    df["provider"] = provider_name

    needed = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "director", "cast", "country", "listed_in", "description"]:
        df[col] = df[col].fillna("").astype(str)

    return df[needed + ["provider"]]


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["title"],
        df["listed_in"],
        df["description"],
        df["cast"],
        df["director"],
        df["country"],
    ]
    return (
        parts[0].str.lower()
        + " "
        + parts[1].str.lower()
        + " "
        + parts[2].str.lower()
        + " "
        + parts[3].str.lower()
        + " "
        + parts[4].str.lower()
        + " "
        + parts[5].str.lower()
    )


def _load_all() -> pd.DataFrame:
    providers = [
        ("netflix_titles.XLS", "Netflix"),
        ("amazon_prime_titles.XLS", "Prime Video"),
        ("disney_plus_titles.XLS", "Disney+"),
    ]
    frames = []
    for fname, name in providers:
        frames.append(_load_provider_xls(fname, name))
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def build_offline_movie_model() -> Tuple[pd.DataFrame, any]:
    global _MOVIE_DF, _MOVIE_MATRIX, _MOVIE_VECTORIZER

    if _MOVIE_DF is not None and _MOVIE_MATRIX is not None:
        return _MOVIE_DF, _MOVIE_MATRIX

    df_all = _load_all()
    df_movies = df_all[df_all["type"].str.lower() == "movie"].reset_index(drop=True)
    df_movies["text"] = _build_text_column(df_movies)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(df_movies["text"])

    _MOVIE_DF = df_movies
    _MOVIE_MATRIX = matrix
    _MOVIE_VECTORIZER = vectorizer
    return _MOVIE_DF, _MOVIE_MATRIX


def build_offline_tv_model() -> Tuple[pd.DataFrame, any]:
    global _TV_DF, _TV_MATRIX, _TV_VECTORIZER

    if _TV_DF is not None and _TV_MATRIX is not None:
        return _TV_DF, _TV_MATRIX

    df_all = _load_all()
    df_tv = df_all[df_all["type"].str.lower() != "movie"].reset_index(drop=True)
    df_tv["text"] = _build_text_column(df_tv)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(df_tv["text"])

    _TV_DF = df_tv
    _TV_MATRIX = matrix
    _TV_VECTORIZER = vectorizer
    return _TV_DF, _TV_MATRIX


def get_offline_similar(df: pd.DataFrame, matrix, title: str, top_n: int = 30) -> pd.DataFrame:
    title_lc = title.strip().lower()
    mask = df["title"].str.lower().str.contains(title_lc)
    candidates = df[mask]

    if candidates.empty:
        return pd.DataFrame()

    idx = candidates.index[0]
    sims = linear_kernel(matrix[idx], matrix).flatten()
    out = df.copy()
    out["similarity"] = sims
    out = out[out.index != idx]
    out = out.sort_values("similarity", ascending=False).head(top_n)

    cols = [
        "title",
        "provider",
        "type",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "similarity",
    ]
    return out[cols]
