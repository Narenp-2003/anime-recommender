from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Global cache
_ANIME_DF = None

# repo_root/offline_mal_model.py
# repo_root/data/anime_cleaned.csv
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data"
ANIME_FILENAME = "anime_cleaned.csv"
ANIME_PATH = DATASET_PATH / ANIME_FILENAME


def _load_anime_df() -> pd.DataFrame:
    """
    Load anime_cleaned.csv and prepare text features.
    Limited to first N rows for speed on 8GB RAM.
    """
    global _ANIME_DF
    if _ANIME_DF is not None:
        return _ANIME_DF

    if not ANIME_PATH.exists():
        raise FileNotFoundError(
            f"{ANIME_FILENAME} not found at {ANIME_PATH}. "
            "Make sure it is committed under data/."
        )

    df = pd.read_csv(ANIME_PATH)

    # Limit size for speed / memory
    df = df.head(15000)

    wanted = ["mal_id", "title_english", "title", "synopsis", "genre", "score", "episodes"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

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
