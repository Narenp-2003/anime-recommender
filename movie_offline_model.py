import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DATA_ROOT = Path(r"E:\anime-recommender\movie data")

_MOVIE_DF = None
_MOVIE_TFIDF = None
_MOVIE_MATRIX = None

_TV_DF = None
_TV_TFIDF = None
_TV_MATRIX = None


def _load_provider_xls(fname: str, provider_name: str) -> pd.DataFrame:
    path = DATA_ROOT / fname
    df = pd.read_excel(path)

    df = df.copy()
    df["provider"] = provider_name

    # Normalize columns
    for col in ["title", "listed_in", "description", "cast", "director", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    if "release_year" not in df.columns:
        df["release_year"] = None

    if "type" not in df.columns:
        df["type"] = "Movie"

    return df[
        [
            "show_id",
            "provider",
            "type",
            "title",
            "director",
            "cast",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in",
            "description",
        ]
    ]


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


def build_offline_movie_model():
    global _MOVIE_DF, _MOVIE_TFIDF, _MOVIE_MATRIX

    if _MOVIE_DF is not None and _MOVIE_MATRIX is not None:
        return _MOVIE_DF, _MOVIE_MATRIX

    providers = [
        ("netflix_titles.xlsx", "Netflix"),
        ("amazon_prime_titles.xlsx", "Prime"),
        ("disney_plus_titles.xlsx", "Disney+"),
    ]

    frames = []
    for fname, name in providers:
        frames.append(_load_provider_xls(fname, name))

    df_all = pd.concat(frames, ignore_index=True)

    # Movies only
    df_movies = df_all[df_all["type"].str.lower() == "movie"].reset_index(drop=True)
    df_movies["text"] = _build_text_column(df_movies)

    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = tfidf.fit_transform(df_movies["text"])

    _MOVIE_DF = df_movies
    _MOVIE_TFIDF = tfidf
    _MOVIE_MATRIX = matrix
    return _MOVIE_DF, _MOVIE_MATRIX


def build_offline_tv_model():
    global _TV_DF, _TV_TFIDF, _TV_MATRIX

    if _TV_DF is not None and _TV_MATRIX is not None:
        return _TV_DF, _TV_MATRIX

    providers = [
        ("netflix_titles.xlsx", "Netflix"),
        ("amazon_prime_titles.xlsx", "Prime"),
        ("disney_plus_titles.xlsx", "Disney+"),
    ]

    frames = []
    for fname, name in providers:
        frames.append(_load_provider_xls(fname, name))

    df_all = pd.concat(frames, ignore_index=True)

    # TV Shows only
    df_tv = df_all[df_all["type"].str.lower() != "movie"].reset_index(drop=True)
    df_tv["text"] = _build_text_column(df_tv)

    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = tfidf.fit_transform(df_tv["text"])

    _TV_DF = df_tv
    _TV_TFIDF = tfidf
    _TV_MATRIX = matrix
    return _TV_DF, _TV_MATRIX


def get_offline_similar(df: pd.DataFrame, matrix, title: str, top_n: int = 30):
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

    return out[
        [
            "title",
            "provider",
            "type",
            "release_year",
            "rating",
            "duration",
            "listed_in",
            "similarity",
        ]
    ]
