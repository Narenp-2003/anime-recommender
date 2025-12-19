# anime_recommender.py
import requests
import pandas as pd

# --- Base URLs for providers ---
JIKAN_BASE_URL = "https://api.jikan.moe/v4/anime"
KITSU_BASE_URL = "https://kitsu.io/api/edge/anime"
ANIMEDB_BASE_URL = "https://animedb.docs.apiary.io"  # placeholder, structure only


# ---------- JIKAN IMPLEMENTATION ----------
def jikan_search_anime(query, limit=10):
    """
    Search anime by title using Jikan (MyAnimeList unofficial).
    Handles network/SSL errors gracefully.
    Docs: https://docs.api.jikan.moe
    """
    params = {"q": query, "limit": limit}
    try:
        response = requests.get(JIKAN_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[JIKAN] Request failed: {e}")
        return []

    data = response.json()
    return data.get("data", [])


def jikan_results_to_df(results):
    """
    Convert Jikan results to unified DataFrame.
    Fields: title, score, total_episodes, synopsis, status,
            genres, start_date, end_date, type.
    """
    rows = []
    for item in results:
        title = item.get("title")
        score = item.get("score")
        episodes = item.get("episodes")
        synopsis = item.get("synopsis")
        status = item.get("status")  # "Finished Airing", "Currently Airing", etc.[web:106][web:143]

        genres_list = [g["name"] for g in item.get("genres", [])] or []
        type_ = item.get("type")  # TV, Movie, OVA, etc.[web:222]

        aired = item.get("aired") or {}
        start_date = (aired.get("from") or "").split("T")[0]  # "YYYY-MM-DD"
        end_date = (aired.get("to") or "").split("T")[0]

        rows.append({
            "title": title,
            "score": score,
            "total_episodes": episodes,
            "synopsis": synopsis,
            "status": status,
            "genres": ", ".join(genres_list),
            "start_date": start_date,
            "end_date": end_date,
            "type": type_,
            "provider": "jikan",
        })

    return pd.DataFrame(rows)


# ---------- KITSU IMPLEMENTATION ----------
def kitsu_search_anime(query, limit=10):
    """
    Basic Kitsu search with text filter.
    Handles network/SSL errors gracefully.
    Docs: https://api-docs.kitsu.cloud
    """
    params = {
        "filter[text]": query,
        "page[limit]": limit
    }
    try:
        response = requests.get(KITSU_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[KITSU] Request failed: {e}")
        return []

    data = response.json()
    return data.get("data", [])


def kitsu_results_to_df(results):
    """
    Convert Kitsu JSON:API results to unified DataFrame.
    Attributes: canonicalTitle, averageRating, episodeCount, synopsis, status,
                startDate, endDate.[web:107][web:104][web:101]
    """
    rows = []
    for item in results:
        attrs = item.get("attributes", {})
        title = attrs.get("canonicalTitle")
        avg_rating = attrs.get("averageRating")
        synopsis = attrs.get("synopsis")
        episode_count = attrs.get("episodeCount")
        status = attrs.get("status")  # "finished", "current", etc.

        start_date = attrs.get("startDate")  # "YYYY-MM-DD"
        end_date = attrs.get("endDate")

        genres_str = ""  # Kitsu genres via relationships; left empty for now

        rows.append({
            "title": title,
            "score": avg_rating,
            "total_episodes": episode_count,
            "synopsis": synopsis,
            "status": status,
            "genres": genres_str,
            "start_date": start_date,
            "end_date": end_date,
            "type": attrs.get("subtype"),  # TV, movie, etc. if present
            "provider": "kitsu",
        })

    return pd.DataFrame(rows)


# ---------- THIRD PROVIDER (GENERIC / PLACEHOLDER) ----------
def animedb_search_anime(query, limit=10):
    """
    Placeholder for a third provider (e.g., AnimeDb / AnimeAPI style).
    Many such APIs expose fields like total_episodes, status, genres, dates.[web:137][web:140]
    For now, return empty so schema is ready for future extension.
    """
    return []


def animedb_results_to_df(results):
    """
    Map AnimeDb-style results to unified schema when you add a real API.
    """
    rows = []
    for item in results:
        title = item.get("title")
        synopsis = item.get("description")
        total_eps = item.get("total_episodes")
        status = item.get("status")
        genres_str = ", ".join(item.get("genres", [])) if item.get("genres") else ""
        start_date = item.get("start_date")
        end_date = item.get("end_date")
        type_ = item.get("type")

        rows.append({
            "title": title,
            "score": item.get("average_rating"),
            "total_episodes": total_eps,
            "synopsis": synopsis,
            "status": status,
            "genres": genres_str,
            "start_date": start_date,
            "end_date": end_date,
            "type": type_,
            "provider": "animedb",
        })

    return pd.DataFrame(rows)


# ---------- GENERIC INTERFACE ----------
def get_provider_dataframe(query, limit=10, provider="jikan"):
    """
    Get a unified DataFrame for a single provider.
    Columns: title, score, total_episodes, synopsis, status,
             genres, start_date, end_date, type, provider.
    """
    provider = provider.lower()
    if provider == "jikan":
        res = jikan_search_anime(query, limit)
        return jikan_results_to_df(res)
    elif provider == "kitsu":
        res = kitsu_search_anime(query, limit)
        return kitsu_results_to_df(res)
    elif provider == "animedb":
        res = animedb_search_anime(query, limit)
        return animedb_results_to_df(res)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _title_match_score(title: str, query: str) -> float:
    """
    Title similarity:

    - 1.0: exact match (case-insensitive, trimmed)
    - 0.8: title equals query ignoring small punctuation differences
    - 0.5: fallback partial overlap (used only if nothing matches exactly)
    - 0.0: otherwise
    """
    if not isinstance(title, str):
        return 0.0

    t_raw = title.strip()
    q_raw = query.strip()

    t = t_raw.lower()
    q = q_raw.lower()

    if not t or not q:
        return 0.0

    # Exact match
    if t == q:
        return 1.0

    # Soft exact: remove punctuation like ":" and "–"
    def normalize(s: str) -> str:
        return (
            s.replace(":", "")
             .replace("–", "-")
             .replace("—", "-")
             .replace("!", "")
             .replace("?", "")
             .strip()
             .lower()
        )

    if normalize(t_raw) == normalize(q_raw):
        return 0.8

    # Fallback: partial word overlap
    t_words = set(t.replace("?", "").replace("!", "").split())
    q_words = set(q.replace("?", "").replace("!", "").split())
    if not q_words:
        return 0.0

    overlap = len(t_words & q_words) / len(q_words)
    if overlap >= 0.5:
        return 0.5
    else:
        return 0.0


def search_all_providers(query, limit=5):
    """
    Search Jikan, Kitsu, and placeholder AnimeDb; combine and
    keep only reasonably matching titles.

    Priority:
    - First, keep only exact/soft matches (score >= 0.8).
    - If none, fall back to partial matches (score >= 0.5).
    """
    providers = ["jikan", "kitsu", "animedb"]
    dfs = []

    for prov in providers:
        df = get_provider_dataframe(query, limit=limit, provider=prov)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[
            "title", "score", "total_episodes", "synopsis", "status",
            "genres", "start_date", "end_date", "type",
            "provider", "simple_match"
        ])

    combined = pd.concat(dfs, ignore_index=True)

    # Compute title match score
    combined["simple_match"] = combined["title"].apply(
        lambda t: _title_match_score(t, query)
    )

    # Prefer true/soft exact matches
    exact_mask = combined["simple_match"] >= 0.8
    if exact_mask.any():
        combined = combined[exact_mask].reset_index(drop=True)
    else:
        # Fallback: allow decent partial matches
        combined = combined[combined["simple_match"] >= 0.5].reset_index(drop=True)

    return combined


def get_genre_based_recommendations(all_results_df, main_row, top_n=30):
    """
    From a larger DataFrame of related anime, return top_n other anime
    that share at least one genre with the main anime, sorted by
    number of shared genres and score descending.
    """
    main_genres = str(main_row.get("genres") or "")
    if not main_genres.strip():
        return pd.DataFrame()

    main_genre_set = set(g.strip().lower() for g in main_genres.split(",") if g.strip())

    if not main_genre_set:
        return pd.DataFrame()

    def shared_genre_count(genres_str):
        if not isinstance(genres_str, str):
            return 0
        gset = set(g.strip().lower() for g in genres_str.split(",") if g.strip())
        return len(main_genre_set & gset)

    df = all_results_df.copy()
    # Exclude the main row itself (by title & provider)
    df = df[(df["title"] != main_row["title"]) | (df["provider"] != main_row["provider"])]

    df["shared_genres"] = df["genres"].apply(shared_genre_count)
    df = df[df["shared_genres"] > 0]

    if df.empty:
        return df

    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.sort_values(by=["shared_genres", "score_num"], ascending=[False, False])

    return df.head(top_n)


def get_same_title_group_sorted(combined_df, main_row):
    """
    For anime with multiple seasons/movies/specials (same or very similar title),
    group them and sort by release date (start_date ascending).
    """
    main_title = str(main_row["title"]).strip().lower()

    def is_same_series(title):
        if not isinstance(title, str):
            return False
        t = title.strip().lower()
        # Basic heuristic: main title is substring or vice versa
        return main_title in t or t in main_title

    group_df = combined_df[combined_df["title"].apply(is_same_series)].copy()

    if group_df.empty:
        return group_df

    # Normalize dates so we can sort
    group_df["start_date_parsed"] = pd.to_datetime(group_df["start_date"], errors="coerce")
    group_df = group_df.sort_values(
        by=["start_date_parsed", "type", "title"],
        ascending=[True, True, True]
    ).drop(columns=["start_date_parsed"])

    return group_df


def cli_demo():
    """
    Terminal test, transparent output.
    """
    query = input("Enter an anime name: ")
    df = search_all_providers(query, limit=5)

    if df.empty:
        print("No results found.")
        return

    for i, row in df.iterrows():
        print(f"{i+1}. {row['title']} [{row['provider']}]")
        print(f"   Score: {row['score']}, Episodes: {row['total_episodes']}")
        print(f"   Status: {row['status']}")
        if isinstance(row["synopsis"], str):
            print(f"   Synopsis: {row['synopsis']}")
        print()


if __name__ == "__main__":
    cli_demo()
