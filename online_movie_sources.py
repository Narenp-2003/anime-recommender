import os
from typing import Optional, Dict, Any

import requests


TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")          # e.g. 4c49e026
STREAMING_API_KEY = os.getenv("STREAMING_API_KEY")  # e.g. 9dnsjgw52ad2

TMDB_BASE = "https://api.themoviedb.org/3"
OMDB_BASE = "https://www.omdbapi.com/"
STREAMING_BASE = "https://streaming-availability.p.rapidapi.com"


def _safe_get(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_omdb_for_tmdb(title: str, year: Optional[int] = None) -> Optional[dict]:
    """
    IMDb-style data for a movie/series from OMDb using title+year.
    """
    if not OMDB_API_KEY:
        return None

    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
    }
    if year:
        params["y"] = str(year)

    data = _safe_get(OMDB_BASE, params)
    if not data or data.get("Response") != "True":
        return None
    return data


def fetch_streaming_availability(title: str, year: Optional[int] = None, country: str = "IN") -> Optional[dict]:
    """
    Where to watch a title using Streaming Availability API (RapidAPI).
    """
    if not STREAMING_API_KEY:
        return None

    headers = {
        "x-rapidapi-key": STREAMING_API_KEY,
        "x-rapidapi-host": "streaming-availability.p.rapidapi.com",
    }

    params = {
        "title": title,
        "country": country,
        "show_type": "movie_or_series",
    }
    if year:
        params["year"] = str(year)

    url = f"{STREAMING_BASE}/search/title"
    data = _safe_get(url, params, headers=headers)
    if not data:
        return None
    return data


def summarize_streaming_providers(streaming_data: dict) -> str:
    """
    Turn streaming-availability JSON into a short string like:
    'Netflix, Prime Video, Disney+'.
    """
    if not streaming_data:
        return ""

    items = streaming_data.get("result") or streaming_data.get("titles") or []
    if not items:
        return ""

    first = items[0]
    offers = first.get("streamingInfo") or {}
    providers = set()

    # Example schema: {"in": {"netflix": {...}, "prime": {...}}}
    for country_info in offers.values():
        for provider_name in country_info.keys():
            providers.add(provider_name.title())

    if not providers:
        return ""

    return ", ".join(sorted(providers))
