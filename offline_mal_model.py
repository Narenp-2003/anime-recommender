import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Global caches (used if you call this module outside Streamlit)
_ANIME_DF = None

# Folder with anime_cleaned.csv
DATASET_PATH = r"E:\ani_list"  # make sure E:\ani_list\anime_cleaned.csv exists


def _load_anime_df() -> pd.DataFrame:
    """
    Load anime_cleaned.csv and prepare text features.
    Limited to first N rows for speed on 8GB RAM.
    """
    global _ANIME_DF
    if _ANIME_DF is not None:
        return _ANIME_DF

    anime_path = os.path.join(DATASET_PATH, "anime_cleaned.csv")
    if not os.path.exists(anime_path):
        raise FileNotFoundError(f"anime_cleaned.csv not found at {anime_path}")

    df = pd.read_csv(anime_path)

    # Limit size for speed / memory
    df = df.head(15000)

    # Keep only columns that actually exist
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

    # Build text field for TF‑IDF using whatever exists
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


def build_offline_model():
    """
    Build TF-IDF model and matrix for offline recommendations.
    Returns (df, tfidf_matrix).
    """
    df = _load_anime_df()
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df["text"])
    return df, tfidf_matrix
