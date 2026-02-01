import os
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
STREAMING_API_KEY = os.getenv("STREAMING_API_KEY")


def fetch_omdb_for_tmdb(title: str, year: int | None):
    if not OMDB_API_KEY:
        return None

    params = {
        "t": title,
        "apikey": OMDB_API_KEY,
    }
    if year:
        params["y"] = year

    try:
        resp = requests.get("https://www.omdbapi.com/", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") != "True":
            return None
        return data
    except Exception:
        return None


def fetch_streaming_availability(title: str, year: int | None, country: str = "IN"):
    """
    Uses a Streaming Availability–style API (RapidAPI) to find legal platforms.
    Adjust `url`, headers, and JSON keys if your provider is slightly different.
    """
    if not STREAMING_API_KEY:
        return []

    url = "https://streaming-availability.p.rapidapi.com/search/title"
    headers = {
        "X-RapidAPI-Key": STREAMING_API_KEY,
        "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com",
    }
    params = {
        "title": title,
        "country": country,
        "output_language": "en",
        "show_type": "all",
        "series_granularity": "show",
    }
    if year:
        params["year"] = year

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = data.get("result") or []
    if not results:
        return []

    item = results[0]
    streaming_info = item.get("streamingInfo") or {}
    providers = []

    # streamingInfo structure: { "netflix": { "in": [ { "link": "https://..." }, ... ] }, ... }
    for provider_name, regions in streaming_info.items():
        region_info = regions.get(country.lower()) or regions.get(country.upper())
        if not region_info:
            continue
        for entry in region_info:
            url_link = entry.get("link") or entry.get("url")
            if url_link:
                providers.append(
                    {
                        "name": provider_name.capitalize(),
                        "url": url_link,
                    }
                )

    return providers


def summarize_streaming_providers(providers: list[dict]):
    """
    Deduplicate by provider name and return:
    - list of {name, url}
    - simple comma-separated label (optional)
    """
    if not providers:
        return [], ""

    seen = {}
    for p in providers:
        name = p.get("name")
        url = p.get("url")
        if not name:
            continue
        if name not in seen:
            seen[name] = url

    out_list = [{"name": n, "url": u} for n, u in seen.items()]
    label = ", ".join(p["name"] for p in out_list)
    return out_list, label
