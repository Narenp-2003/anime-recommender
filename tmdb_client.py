import os
from typing import Optional

import pandas as pd
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"


def _tmdb_get(path: str, params: Optional[dict] = None, timeout: int = 10):
    """Internal helper to call TMDb with API key."""
    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY is not set.")
        return None

    params = params or {}
    params["api_key"] = TMDB_API_KEY
    # Default language for all requests (you can change to "en-US" or "hi-IN")
    if "language" not in params:
        params["language"] = "en-IN"

    try:
        resp = requests.get(f"{TMDB_BASE_URL}{path}", params=params, timeout=timeout)
        # Temporary debug – you can comment these out later
        print("DEBUG TMDB REQUEST:", resp.url, "STATUS:", resp.status_code)
        if resp.status_code != 200:
            print("DEBUG TMDB BODY:", resp.text[:300])
            return None
        return resp.json()
    except Exception as e:
        print("EXCEPTION while calling TMDb:", e)
        return None


def _poster_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"https://image.tmdb.org/t/p/w342{path}"


def search_movies_tmdb(query: str, limit: int = 20) -> pd.DataFrame:
    """Search movies by title (biased to Indian releases)."""
    if not query or not isinstance(query, str):
        return pd.DataFrame()

    data = _tmdb_get(
        "/search/movie",
        {
            "query": query,
            "page": 1,
            "include_adult": False,
            "region": "IN",          # prefer Indian releases
            # "language" comes from _tmdb_get ("en-IN")
        },
    )
    if not data or "results" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["results"][:limit]:
        rows.append(
            {
                "provider": "TMDbMovie",
                "provider_id": item.get("id"),
                "title": item.get("title") or "",
                "type": "Movie",
                "score": item.get("vote_average"),
                "vote_count": item.get("vote_count"),
                "release_date": item.get("release_date"),
                "overview": item.get("overview") or "",
                "poster_url": _poster_url(item.get("poster_path")),
            }
        )
    return pd.DataFrame(rows)


def search_tv_tmdb(query: str, limit: int = 20) -> pd.DataFrame:
    """Search TV shows (including sitcoms) by title."""
    if not query or not isinstance(query, str):
        return pd.DataFrame()

    data = _tmdb_get(
        "/search/tv",
        {
            "query": query,
            "page": 1,
            "include_adult": False,
            # language = "en-IN" from _tmdb_get
        },
    )
    if not data or "results" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["results"][:limit]:
        rows.append(
            {
                "provider": "TMDbTV",
                "provider_id": item.get("id"),
                "title": item.get("name") or "",
                "type": "TV",
                "score": item.get("vote_average"),
                "vote_count": item.get("vote_count"),
                "first_air_date": item.get("first_air_date"),
                "overview": item.get("overview") or "",
                "poster_url": _poster_url(item.get("poster_path")),
            }
        )
    return pd.DataFrame(rows)


def get_movie_recommendations(main_row: pd.Series, limit: int = 20) -> pd.DataFrame:
    """Get TMDb movie recommendations for a given movie row."""
    tmdb_id = main_row.get("provider_id")
    if not tmdb_id:
        return pd.DataFrame()

    data = _tmdb_get(f"/movie/{tmdb_id}/recommendations", {"page": 1})
    if not data or "results" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["results"][:limit]:
        rows.append(
            {
                "provider": "TMDbMovieRec",
                "provider_id": item.get("id"),
                "title": item.get("title") or "",
                "type": "Movie",
                "score": item.get("vote_average"),
                "vote_count": item.get("vote_count"),
                "release_date": item.get("release_date"),
                "overview": item.get("overview") or "",
                "poster_url": _poster_url(item.get("poster_path")),
            }
        )
    return pd.DataFrame(rows)


def get_tv_recommendations(main_row: pd.Series, limit: int = 20) -> pd.DataFrame:
    """Get TMDb TV recommendations for a given TV show row."""
    tmdb_id = main_row.get("provider_id")
    if not tmdb_id:
        return pd.DataFrame()

    data = _tmdb_get(f"/tv/{tmdb_id}/recommendations", {"page": 1})
    if not data or "results" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["results"][:limit]:
        rows.append(
            {
                "provider": "TMDbTVRec",
                "provider_id": item.get("id"),
                "title": item.get("name") or "",
                "type": "TV",
                "score": item.get("vote_average"),
                "vote_count": item.get("vote_count"),
                "first_air_date": item.get("first_air_date"),
                "overview": item.get("overview") or "",
                "poster_url": _poster_url(item.get("poster_path")),
            }
        )
    return pd.DataFrame(rows)
