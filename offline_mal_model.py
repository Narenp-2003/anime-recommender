import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Global caches
_ANIME_DF = None
_TFIDF = None
_TFIDF_MATRIX = None

# Folder with anime_cleaned.csv
DATASET_PATH = r"E:\ani_list"  # make sure E:\ani_list\anime_cleaned.csv exists


def _load_anime_df():
    """
    Load anime_cleaned.csv and prepare text features.
    """
    global _ANIME_DF
    if _ANIME_DF is not None:
        return _ANIME_DF

    anime_path = os.path.join(DATASET_PATH, "anime_cleaned.csv")
    if not os.path.exists(anime_path):
        raise FileNotFoundError(f"anime_cleaned.csv not found at {anime_path}")

    df = pd.read_csv(anime_path)

    # Keep only columns that actually exist in your file
    wanted = ["mal_id", "title_english", "title", "synopsis", "genre", "score", "episodes"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

    # Unified title: prefer English + original title
    if "title_english" in df.columns:
        title_eng = df["title_english"].fillna("").astype(str)
    else:
        title_eng = ""

    if "title" in df.columns:
        title_jp = df["title"].fillna("").astype(str)
    else:
        title_jp = ""

    df["title_display"] = (title_eng + " " + title_jp).str.strip()
    if "title" in df.columns:
        empty_mask = df["title_display"] == ""
        df.loc[empty_mask, "title_display"] = df.loc[empty_mask, "title"].astype(str)

    # Build text field for TF‑IDF using whatever you have
    if "synopsis" in df.columns:
        synopsis = df["synopsis"].fillna("").astype(str)
    else:
        synopsis = ""

    if "genre" in df.columns:
        genre = df["genre"].fillna("").astype(str)
    else:
        genre = ""

    if isinstance(synopsis, str) and synopsis == "":
        if isinstance(genre, str) and genre == "":
            df["text"] = df["title_display"].fillna("").astype(str)
        else:
            df["text"] = genre
    else:
        df["text"] = synopsis.astype(str) + " " + genre.astype(str)

    _ANIME_DF = df
    return _ANIME_DF


def _build_tfidf():
    """
    Build (or reuse) TF-IDF model on df['text'].
    """
    global _TFIDF, _TFIDF_MATRIX
    if _TFIDF is not None and _TFIDF_MATRIX is not None:
        return _TFIDF, _TFIDF_MATRIX

    df = _load_anime_df()
    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    tfidf_matrix = tfidf.fit_transform(df["text"])

    _TFIDF = tfidf
    _TFIDF_MATRIX = tfidf_matrix
    return _TFIDF, _TFIDF_MATRIX


def get_offline_similar_by_title(input_title: str, top_n: int = 30) -> pd.DataFrame:
    """
    Given an anime title string, find similar shows using MAL dataset (content-based).
    Returns a DataFrame with columns: title_display, score, episodes, similarity (where available).
    """
    if not input_title or not isinstance(input_title, str):
        return pd.DataFrame()

    df = _load_anime_df()
    tfidf, tfidf_matrix = _build_tfidf()

    # Find candidate matches by title_display contains
    t = input_title.strip().lower()
    mask = df["title_display"].astype(str).str.lower().str.contains(t)
    candidates = df[mask]
    if candidates.empty:
        return pd.DataFrame()

    # Choose best index: highest score if present, otherwise first candidate
    if "score" in candidates.columns:
        best_idx = candidates["score"].fillna(0).astype(float).idxmax()
    else:
        best_idx = candidates.index[0]

    if best_idx not in df.index:
        return pd.DataFrame()

    # Cosine similarity to all others
    cosine_sim = linear_kernel(tfidf_matrix[best_idx], tfidf_matrix).flatten()
    df_out = df.copy()
    df_out["similarity"] = cosine_sim

    # Drop the main show itself and sort
    df_out = df_out[df_out.index != best_idx]
    df_out = df_out.sort_values("similarity", ascending=False).head(top_n)

    # Only select columns that exist
    out_cols = []
    for c in ["title_display", "score", "episodes", "similarity"]:
        if c in df_out.columns:
            out_cols.append(c)

    return df_out[out_cols].reset_index(drop=True)
