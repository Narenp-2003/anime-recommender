# anime_recommender.py
import requests
import pandas as pd
from rapidfuzz import fuzz  # fuzzy matching


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
    Fields: mal_id, title, all_titles, score, total_episodes, synopsis, status,
            genres, start_date, end_date, type, provider.
    """
    rows = []
    for item in results:
        title = item.get("title")
        score = item.get("score")
        episodes = item.get("episodes")
        synopsis = item.get("synopsis")
        status = item.get("status")

        genres_list = [g["name"] for g in item.get("genres", [])] or []
        type_ = item.get("type")  # TV, Movie, OVA, etc.

        aired = item.get("aired") or {}
        start_date = (aired.get("from") or "").split("T")[0]
        end_date = (aired.get("to") or "").split("T")[0]

        # Collect all known titles (default, English, Japanese, synonyms)
        all_titles = set()
        if isinstance(title, str):
            all_titles.add(title)
        for t in item.get("titles", []) or []:
            t_str = t.get("title")
            if isinstance(t_str, str):
                all_titles.add(t_str)
        for alt in item.get("title_synonyms", []) or []:
            if isinstance(alt, str):
                all_titles.add(alt)

        rows.append({
            "mal_id": item.get("mal_id"),
            "title": title,
            "all_titles": list(all_titles),
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


def jikan_fetch_relations(mal_id):
    """
    Fetch related anime for a given MAL ID using Jikan relations endpoint.
    Returns a list of dicts with minimal fields needed for the series table.
    """
    if mal_id is None:
        return []

    url = f"{JIKAN_BASE_URL}/{int(mal_id)}/relations"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[JIKAN] Relations request failed for {mal_id}: {e}")
        return []

    data = resp.json()
    relations = data.get("data", []) or []

    rows = []
    for rel in relations:
        rel_type = rel.get("relation")  # e.g. "Sequel", "Prequel", "Side story"
        entries = rel.get("entry", []) or []
        for ent in entries:
            title = ent.get("name")
            r_mal_id = ent.get("mal_id")
            if not isinstance(title, str):
                continue

            rows.append({
                "mal_id": r_mal_id,
                "title": title,
                "all_titles": [title],
                "score": None,
                "total_episodes": None,
                "synopsis": None,
                "status": rel_type,   # store relation type if nothing better
                "genres": "",
                "start_date": "",
                "end_date": "",
                "type": None,         # unknown from relations-only call
                "provider": "jikan-rel",  # mark that it came from relations
            })

    return rows


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
    Attributes: canonicalTitle, titles.{en,en_jp,ja_jp}, averageRating,
                episodeCount, synopsis, status, startDate, endDate.
    """
    rows = []
    for item in results:
        attrs = item.get("attributes", {})
        title = attrs.get("canonicalTitle")
        avg_rating = attrs.get("averageRating")
        synopsis = attrs.get("synopsis")
        episode_count = attrs.get("episodeCount")
        status = attrs.get("status")

        start_date = attrs.get("startDate")
        end_date = attrs.get("endDate")

        genres_str = ""  # Kitsu genres via relationships; left empty for now

        # Collect title variants
        all_titles = set()
        if isinstance(title, str):
            all_titles.add(title)
        tdict = attrs.get("titles") or {}
        for key in ("en", "en_jp", "ja_jp"):
            t_str = tdict.get(key)
            if isinstance(t_str, str):
                all_titles.add(t_str)

        rows.append({
            "mal_id": None,
            "title": title,
            "all_titles": list(all_titles),
            "score": avg_rating,
            "total_episodes": episode_count,
            "synopsis": synopsis,
            "status": status,
            "genres": genres_str,
            "start_date": start_date,
            "end_date": end_date,
            "type": attrs.get("subtype"),
            "provider": "kitsu",
        })

    return pd.DataFrame(rows)


# ---------- THIRD PROVIDER (GENERIC / PLACEHOLDER) ----------
def animedb_search_anime(query, limit=10):
    """
    Placeholder for a third provider.
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

        all_titles = []
        if isinstance(title, str):
            all_titles = [title]

        rows.append({
            "mal_id": None,
            "title": title,
            "all_titles": all_titles,
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


# ---------- TITLE MATCHING (FUZZY) ----------
def _norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (
        s.lower()
         .replace("’", "'")
         .replace("–", "-")
         .replace("—", "-")
         .replace(":", " ")
         .replace("!", "")
         .replace("?", "")
         .strip()
    )


def _fuzzy_title_score(user_title: str, candidate_title: str) -> float:
    """
    Fuzzy similarity between user query and a single candidate title (0.0–1.0).
    """
    u = _norm_title(user_title)
    c = _norm_title(candidate_title)
    if not u or not c:
        return 0.0
    return fuzz.token_set_ratio(u, c) / 100.0


def _title_match_score(row_title: str, query: str, all_titles=None) -> float:
    """
    Best fuzzy match score between query and all known titles for this anime.
    """
    candidates = []

    if isinstance(row_title, str):
        candidates.append(row_title)

    if isinstance(all_titles, (list, tuple, set)):
        for t in all_titles:
            if isinstance(t, str):
                candidates.append(t)

    if not candidates:
        return 0.0

    best = 0.0
    for cand in candidates:
        s = _fuzzy_title_score(query, cand)
        if s > best:
            best = s
    return best


# ---------- GENERIC INTERFACE ----------
def get_provider_dataframe(query, limit=10, provider="jikan"):
    """
    Get a unified DataFrame for a single provider.
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


def search_all_providers(query, limit=5):
    """
    Search Jikan, Kitsu, and placeholder AnimeDb; combine and
    keep only reasonably matching titles.

    Priority:
    - First, keep strong fuzzy matches (score >= 0.80).
    - If none, fall back to moderate matches (score >= 0.60).
    """
    providers = ["jikan", "kitsu", "animedb"]
    dfs = []

    for prov in providers:
        df = get_provider_dataframe(query, limit=limit, provider=prov)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[
            "mal_id", "title", "all_titles", "score", "total_episodes",
            "synopsis", "status", "genres", "start_date", "end_date",
            "type", "provider", "simple_match"
        ])

    combined = pd.concat(dfs, ignore_index=True)

    # Compute title match score using all title variants
    combined["simple_match"] = combined.apply(
        lambda row: _title_match_score(
            row.get("title"), query, row.get("all_titles")
        ),
        axis=1,
    )

    # Prefer strong matches; if none, allow moderate ones
    strong_mask = combined["simple_match"] >= 0.80
    if strong_mask.any():
        combined = combined[strong_mask].reset_index(drop=True)
    else:
        combined = combined[combined["simple_match"] >= 0.60].reset_index(drop=True)

    return combined


# ---------- RECOMMENDATION HELPERS ----------
def get_genre_based_recommendations(all_results_df, main_row, top_n=30):
    """
    From a larger DataFrame of related anime, return top_n other anime
    that share at least one genre with the main anime, sorted by
    number of shared genres and score descending.

    If the displayed main_row has weak/missing genre info (e.g. Kitsu or relations),
    try to fall back to a Jikan row with the same/similar title that has genres.
    """
    main_genres = str(main_row.get("genres") or "")

    if not main_genres.strip():
        same_title_mask = (
            (all_results_df["provider"] == "jikan") &
            (all_results_df["title"].astype(str).str.lower() ==
             str(main_row.get("title") or "").strip().lower()) &
            (all_results_df["genres"].astype(str).str.strip() != "")
        )
        if same_title_mask.any():
            fallback_row = all_results_df[same_title_mask].iloc[0]
            main_genres = str(fallback_row.get("genres") or "")

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
        return main_title in t or t in main_title

    group_df = combined_df[combined_df["title"].apply(is_same_series)].copy()

    if group_df.empty:
        return group_df

    group_df["start_date_parsed"] = pd.to_datetime(group_df["start_date"], errors="coerce")
    group_df = group_df.sort_values(
        by=["start_date_parsed", "type", "title"],
        ascending=[True, True, True]
    ).drop(columns=["start_date_parsed"])

    return group_df


def get_series_group_with_relations(combined_df, main_row):
    """
    Build a series group (seasons/movies/specials) for the main anime, and
    enrich it with Jikan relations when MAL ID is available.
    """
    base_group = get_same_title_group_sorted(combined_df, main_row)

    mal_id = main_row.get("mal_id") if "mal_id" in main_row else None
    if main_row.get("provider") != "jikan" or mal_id in (None, "", 0):
        return base_group

    related_rows = jikan_fetch_relations(mal_id)
    if not related_rows:
        return base_group

    rel_df = pd.DataFrame(related_rows)

    combined_group = pd.concat([base_group, rel_df], ignore_index=True)
    combined_group = combined_group.drop_duplicates(
        subset=["title", "provider"],
        keep="first"
    )

    combined_group["start_date_parsed"] = pd.to_datetime(
        combined_group.get("start_date"), errors="coerce"
    )
    combined_group = combined_group.sort_values(
        by=["start_date_parsed", "type", "title"],
        ascending=[True, True, True]
    ).drop(columns=["start_date_parsed"])

    return combined_group


def cli_demo():
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
