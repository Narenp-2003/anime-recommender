"""
Content-based anime recommendation engine.

Replaces naive genre-overlap-count ranking with TF-IDF + cosine similarity
over genre text (this dataset — data/anime_cleaned.csv — has no synopsis
column, so similarity is computed over genre tags only). This is still a
meaningful improvement over raw overlap counting: TF-IDF down-weights
common genres (e.g. "Comedy") and up-weights rare, distinctive ones
(e.g. "Mahou Shoujo"), and cosine similarity normalizes for shows that
simply have more tags — both of which the old shared_genres count ignored.

If you want true synopsis-based semantic similarity, the dataset needs a
synopsis column populated (e.g. backfilled from Jikan) — happy to help
with that as a follow-up.

Drop-in usage:
    from content_based_recommender import ContentRecommender

    rec = ContentRecommender()  # loads/builds index once, cached
    results_df = rec.recommend_by_title("Attack on Titan", top_n=20)
    # or, if you already have a row with genre text:
    results_df = rec.recommend_from_text(genres, synopsis, exclude_title="...", top_n=20)
"""

from pathlib import Path
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "anime_cleaned.csv"


class ContentRecommender:
    """
    Loads the anime dataset once, builds a TF-IDF matrix over a combined
    text field (genres + themes/demographics if present + synopsis),
    and answers similarity queries with cosine similarity instead of
    raw genre-tag counting.
    """

    def __init__(self, dataset_path: Path = DATASET_PATH, max_rows: int = 15000):
        self.dataset_path = dataset_path
        self.max_rows = max_rows
        self._df = None
        self._matrix = None
        self._vectorizer = None
        self._load()

    def _load(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Make sure data/anime_cleaned.csv is committed."
            )

        df = pd.read_csv(self.dataset_path)
        df = df.head(self.max_rows).copy()

        wanted = ["anime_id", "title_english", "title", "genre", "score", "episodes"]
        cols = [c for c in wanted if c in df.columns]
        df = df[cols]

        title_eng = df.get("title_english", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        title_main = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        # Prefer the English title; fall back to the native/romaji title.
        # Don't concatenate both — when they're identical (common) that
        # produced "Death Note Death Note", which corrupted fuzzy-match
        # scores against user search input.
        df["title_display"] = title_eng.where(title_eng != "", title_main)

        genre = df.get("genre", pd.Series("", index=df.index)).fillna("").astype(str)

        # No synopsis column in this dataset — similarity runs on genre
        # text only. Repeating each genre token doesn't change relative
        # TF-IDF weighting within a pure-genre corpus, so just use the
        # genre string directly.
        df["text"] = genre
        df["genre"] = genre

        self._df = df.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self._vectorizer.fit_transform(self._df["text"])

    def _find_title_index(self, title: str):
        """Fuzzy-match a user-provided title to a row in the dataset.
        Uses token_sort_ratio (not WRatio, which has a false-positive
        problem: it scores short unrelated titles like "K K" or "Canaan"
        as 85%+ matches for "Jujutsu Kaisen"/"Chainsaw Man" via aggressive
        partial-ratio boosting). Threshold 90 was chosen because testing
        showed genuine matches scoring 98-100 and false matches capping
        at ~76 with this scorer+dataset. This offline dataset is old and
        doesn't cover many newer titles, so returning no match (rather
        than a bad guess) lets the caller fall back to the live search
        result's real genre data instead of generating irrelevant recs
        from a completely unrelated show."""
        choices = self._df["title_display"].tolist()
        match = process.extractOne(title, choices, scorer=fuzz.token_sort_ratio)
        if not match or match[1] < 90:
            return None
        return match[2]  # index into choices / self._df

    def recommend_by_title(self, title: str, top_n: int = 20) -> pd.DataFrame:
        idx = self._find_title_index(title)
        if idx is None:
            return pd.DataFrame()
        return self._recommend_from_index(idx, top_n=top_n)

    def _recommend_from_index(self, idx: int, top_n: int = 20) -> pd.DataFrame:
        sims = cosine_similarity(self._matrix[idx], self._matrix).flatten()

        result = self._df.copy()
        result["similarity"] = sims

        score_norm = pd.to_numeric(result.get("score"), errors="coerce").fillna(0) / 10.0
        # Similarity does the heavy lifting; score only breaks ties /
        # nudges away from obscure near-duplicates.
        result["rec_score"] = 0.85 * result["similarity"] + 0.15 * score_norm

        result = result.drop(index=idx)  # exclude the source title itself
        result = result.sort_values("rec_score", ascending=False)
        return result.head(top_n).reset_index(drop=True)

    def recommend_from_text(
        self, genres: str, synopsis: str = "", exclude_title: str = "", top_n: int = 20
    ) -> pd.DataFrame:
        """
        For cases where the source title isn't in the local dataset
        (e.g. it came from a live API search) — vectorize its genre
        text on the fly and compare against the corpus. `synopsis` is
        accepted for interface compatibility but currently unused,
        since the offline dataset has no synopsis vocabulary to match
        against.
        """
        vec = self._vectorizer.transform([genres or ""])
        sims = cosine_similarity(vec, self._matrix).flatten()

        result = self._df.copy()
        result["similarity"] = sims
        score_norm = pd.to_numeric(result.get("score"), errors="coerce").fillna(0) / 10.0
        result["rec_score"] = 0.85 * result["similarity"] + 0.15 * score_norm

        if exclude_title:
            result = result[
                result["title_display"].str.lower() != exclude_title.strip().lower()
            ]

        result = result.sort_values("rec_score", ascending=False)
        return result.head(top_n).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_recommender() -> ContentRecommender:
    """Cache a single instance so Streamlit doesn't rebuild the TF-IDF
    index on every rerun."""
    return ContentRecommender()
