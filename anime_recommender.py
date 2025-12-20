# anime_recommender.py

import re
import requests
import pandas as pd
from rapidfuzz import fuzz  # fuzzy matching

# --- Base URLs for providers ---
JIKAN_BASE_URL = "https://api.jikan.moe/v4/anime"
KITSU_BASE_URL = "https://kitsu.io/api/edge/anime"
ANIMEDB_BASE_URL = "https://animedb.docs.apiary.io"  # placeholder, structure only
ANILIST_URL = "https://graphql.anilist.co"


# ========== Utility: robust requests ==========

def _safe_get(url, params=None, headers=None, timeout=10):
    try:
        resp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


# ========== Jikan helpers ==========

def jikan_search_anime(query, limit=10):
    params = {
        "q": query,
        "limit": limit,
        "order_by": "score",
        "sort": "desc",
    }
    data = _safe_get(JIKAN_BASE_URL, params=params)
    if not data or "data" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["data"]:
        mal_id = item.get("mal_id")
        titles = item.get("titles") or []
        title_str = item.get("title") or ""

        # Collect title variants
        all_titles = set()
        if isinstance(titles, list):
            for t in titles:
                val = t.get("title")
                if isinstance(val, str):
                    all_titles.add(val.strip())
        if isinstance(title_str, str) and title_str.strip():
            all_titles.add(title_str.strip())

        type_ = item.get("type") or ""
        episodes = item.get("episodes")
        status = item.get("status") or ""
        score = item.get("score")
        synopsis = item.get("synopsis") or ""

        # Genres
        genres = item.get("genres") or []
        explicit_genres = item.get("explicit_genres") or []
        themes = item.get("themes") or []
        demographics = item.get("demographics") or []

        genre_names = []
        for g_block in (genres, explicit_genres, themes, demographics):
            for g in g_block:
                name = g.get("name")
                if isinstance(name, str):
                    genre_names.append(name.strip())
        genres_str = ", ".join(sorted(set(genre_names)))

        start_date = None
        aired = item.get("aired") or {}
        if isinstance(aired, dict):
            start_date = (aired.get("from") or "")[:10]

        rows.append(
            {
                "provider": "Jikan",
                "provider_id": mal_id,
                "title": title_str,
                "all_titles": list(all_titles),
                "type": type_,
                "total_episodes": episodes,
                "status": status,
                "score": score,
                "genres": genres_str,
                "start_date": start_date,
                "synopsis": synopsis,
                "raw": item,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def jikan_fetch_relations(mal_id):
    if not mal_id:
        return pd.DataFrame()

    url = f"{JIKAN_BASE_URL}/{mal_id}/full"
    data = _safe_get(url)
    if not data or "data" not in data:
        return pd.DataFrame()
    data = data["data"]

    main_title = data.get("title") or ""
    main_titles_block = data.get("titles") or []
    main_all_titles = set()
    for t in main_titles_block:
        v = t.get("title")
        if isinstance(v, str):
            main_all_titles.add(v.strip())
    if isinstance(main_title, str) and main_title.strip():
        main_all_titles.add(main_title.strip())

    relations = data.get("relations") or []
    rows = []

    def extract_genre_string(anime_obj):
        genres = anime_obj.get("genres") or []
        explicit_genres = anime_obj.get("explicit_genres") or []
        themes = anime_obj.get("themes") or []
        demographics = anime_obj.get("demographics") or []
        names = []
        for block in (genres, explicit_genres, themes, demographics):
            for g in block:
                nm = g.get("name")
                if isinstance(nm, str):
                    names.append(nm.strip())
        return ", ".join(sorted(set(names)))

    for rel in relations:
        relation_type = rel.get("relation") or ""
        entry_list = rel.get("entry") or []
        for entry in entry_list:
            e_id = entry.get("mal_id")
            if not e_id:
                continue
            # Call /anime/{id} for this related anime
            rel_data = _safe_get(f"{JIKAN_BASE_URL}/{e_id}")
            if not rel_data or "data" not in rel_data:
                continue
            item = rel_data["data"]

            title = item.get("title") or ""
            titles_block = item.get("titles") or []
            all_titles = set()
            for t in titles_block:
                v = t.get("title")
                if isinstance(v, str):
                    all_titles.add(v.strip())
            if isinstance(title, str) and title.strip():
                all_titles.add(title.strip())

            type_ = item.get("type") or ""
            episodes = item.get("episodes")
            status = item.get("status") or ""
            score = item.get("score")
            synopsis = item.get("synopsis") or ""
            start_date = None
            aired = item.get("aired") or {}
            if isinstance(aired, dict):
                start_date = (aired.get("from") or "")[:10]

            genres_str = extract_genre_string(item)

            rows.append(
                {
                    "provider": "JikanRel",
                    "provider_id": e_id,
                    "title": title,
                    "all_titles": list(all_titles),
                    "type": type_,
                    "total_episodes": episodes,
                    "status": status,
                    "score": score,
                    "genres": genres_str,
                    "start_date": start_date,
                    "synopsis": synopsis,
                    "relation_type": relation_type,
                    "raw": item,
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["provider_id", "title"])
    return df


def jikan_fetch_by_genres(genre_names, limit_per_genre=25, global_limit=200):
    """
    Fetch anime from Jikan by genre names, returning a DataFrame with genres filled.
    """
    if not genre_names:
        return pd.DataFrame()

    genre_list_url = "https://api.jikan.moe/v4/genres/anime"
    genre_data = _safe_get(genre_list_url)
    if not genre_data or "data" not in genre_data:
        return pd.DataFrame()

    name_to_id = {}
    for g in genre_data["data"]:
        nm = g.get("name")
        gid = g.get("mal_id")
        if isinstance(nm, str) and gid is not None:
            name_to_id[nm.strip().lower()] = gid

    genre_ids = []
    for name in genre_names:
        gid = name_to_id.get(name.strip().lower())
        if gid is not None:
            genre_ids.append(str(gid))

    if not genre_ids:
        return pd.DataFrame()

    seen_ids = set()
    rows = []

    for gid in genre_ids:
        params = {
            "genres": gid,
            "order_by": "score",
            "sort": "desc",
            "limit": limit_per_genre,
        }
        data = _safe_get(JIKAN_BASE_URL, params=params)
        if not data or "data" not in data:
            continue

        for item in data["data"]:
            mal_id = item.get("mal_id")
            if not mal_id or mal_id in seen_ids:
                continue
            seen_ids.add(mal_id)

            titles = item.get("titles") or []
            title_str = item.get("title") or ""
            all_titles = set()
            for t in titles:
                v = t.get("title")
                if isinstance(v, str):
                    all_titles.add(v.strip())
            if isinstance(title_str, str) and title_str.strip():
                all_titles.add(title_str.strip())

            type_ = item.get("type") or ""
            episodes = item.get("episodes")
            status = item.get("status") or ""
            score = item.get("score")
            synopsis = item.get("synopsis") or ""

            genres = item.get("genres") or []
            explicit_genres = item.get("explicit_genres") or []
            themes = item.get("themes") or []
            demographics = item.get("demographics") or []

            genre_names_local = []
            for block in (genres, explicit_genres, themes, demographics):
                for g in block:
                    nm = g.get("name")
                    if isinstance(nm, str):
                        genre_names_local.append(nm.strip())
            genres_str = ", ".join(sorted(set(genre_names_local)))

            start_date = None
            aired = item.get("aired") or {}
            if isinstance(aired, dict):
                start_date = (aired.get("from") or "")[:10]

            rows.append(
                {
                    "provider": "JikanGenre",
                    "provider_id": mal_id,
                    "title": title_str,
                    "all_titles": list(all_titles),
                    "type": type_,
                    "total_episodes": episodes,
                    "status": status,
                    "score": score,
                    "genres": genres_str,
                    "start_date": start_date,
                    "synopsis": synopsis,
                    "raw": item,
                }
            )

            if len(rows) >= global_limit:
                break
        if len(rows) >= global_limit:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ========== Kitsu helpers ==========

def kitsu_search_anime(query, limit=10):
    params = {
        "filter[text]": query,
        "page[limit]": limit,
    }
    data = _safe_get(KITSU_BASE_URL, params=params)
    if not data or "data" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["data"]:
        attrs = item.get("attributes") or {}
        titles = attrs.get("titles") or {}
        canonical_title = attrs.get("canonicalTitle") or ""

        title_str = (
            canonical_title
            or titles.get("en")
            or titles.get("en_jp")
            or titles.get("ja_jp")
            or ""
        )

        all_titles = set()
        for key in ("en", "en_jp", "ja_jp"):
            val = titles.get(key)
            if isinstance(val, str):
                all_titles.add(val.strip())
        if isinstance(canonical_title, str) and canonical_title.strip():
            all_titles.add(canonical_title.strip())

        t_type = attrs.get("showType") or ""
        episodes = attrs.get("episodeCount")
        status = attrs.get("status") or ""
        score = attrs.get("averageRating")
        synopsis = attrs.get("synopsis") or ""
        start_date = attrs.get("startDate")

        genres_str = ""  # Kitsu genres not fetched yet

        rows.append(
            {
                "provider": "Kitsu",
                "provider_id": item.get("id"),
                "title": title_str,
                "all_titles": list(all_titles),
                "type": t_type,
                "total_episodes": episodes,
                "status": status,
                "score": float(score) / 10.0 if score else None,
                "genres": genres_str,
                "start_date": start_date,
                "synopsis": synopsis,
                "raw": item,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ========== AniList helper ==========

def anilist_search_anime(query, limit=10):
    """
    Search anime by title using AniList GraphQL.
    """
    if not query or not isinstance(query, str):
        return pd.DataFrame()

    graphql_query = """
    query ($search: String!, $perPage: Int!) {
      Page(page: 1, perPage: $perPage) {
        media(search: $search, type: ANIME) {
          id
          idMal
          title {
            romaji
            english
            native
          }
          format
          episodes
          status
          averageScore
          genres
          startDate {
            year
            month
            day
          }
          description(asHtml: false)
        }
      }
    }
    """

    variables = {"search": query, "perPage": int(limit)}

    try:
        resp = requests.post(
            ANILIST_URL,
            json={"query": graphql_query, "variables": variables},
            timeout=10,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    page = (data or {}).get("data", {}).get("Page", {})
    media_list = page.get("media") or []
    if not media_list:
        return pd.DataFrame()

    rows = []
    for m in media_list:
        t_romaji = (m.get("title") or {}).get("romaji") or ""
        t_english = (m.get("title") or {}).get("english") or ""
        t_native = (m.get("title") or {}).get("native") or ""

        primary_title = t_english or t_romaji or t_native

        all_titles = set()
        for t in (t_romaji, t_english, t_native):
            if isinstance(t, str) and t.strip():
                all_titles.add(t.strip())

        fmt = m.get("format") or ""
        episodes = m.get("episodes")
        status_raw = (m.get("status") or "").upper()
        score = m.get("averageScore")
        score_10 = float(score) / 10.0 if score is not None else None

        genres_list = m.get("genres") or []
        genres_str = ", ".join(
            sorted({g.strip() for g in genres_list if isinstance(g, str)})
        )

        start = m.get("startDate") or {}
        year, month, day = start.get("year"), start.get("month"), start.get("day")
        if all(v is not None for v in (year, month, day)):
            start_date = f"{year:04d}-{month:02d}-{day:02d}"
        elif year is not None:
            start_date = f"{year:04d}-01-01"
        else:
            start_date = None

        desc = m.get("description") or ""
        synopsis = desc if isinstance(desc, str) else ""

        rows.append(
            {
                "provider": "AniList",
                "provider_id": m.get("id"),
                "title": primary_title,
                "all_titles": list(all_titles),
                "type": fmt,
                "total_episodes": episodes,
                "status": status_raw,
                "score": score_10,
                "genres": genres_str,
                "start_date": start_date,
                "synopsis": synopsis,
                "raw": m,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ========== Title matching (RapidFuzz) ==========

def _norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (
        s.lower()
         .replace("’", "'")
         .replace("–", "-")
         .replace("—", "-")
         .replace(":", " ")
         .replace("!", " ")
         .replace("?", " ")
         .replace(".", " ")
         .replace(",", " ")
         .replace("  ", " ")
         .strip()
    )


def _extract_season_number(title: str) -> int | None:
    """
    Extract a season number from a title if present.
    Examples: 'season 1', '2nd season', 'S3'.
    """
    if not isinstance(title, str):
        return None
    t = title.lower()

    m = re.search(r"season\s+(\d+)", t)
    if m:
        return int(m.group(1))

    m = re.search(r"\bs(\d)\b", t)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)(st|nd|rd|th)\s+season", t)
    if m:
        return int(m.group(1))

    return None


def _title_match_score(user_title: str, all_titles):
    """
    Best fuzzy score between user_title and any title variant in all_titles,
    with a season-aware adjustment so S1 vs S2 gets penalized.
    """
    if not isinstance(user_title, str):
        return 0.0
    if not isinstance(all_titles, (list, tuple, set)):
        all_titles = [all_titles]

    user_norm = _norm_title(user_title)
    user_season = _extract_season_number(user_title)

    best = 0.0
    for t in all_titles:
        t_norm = _norm_title(t)
        if not t_norm:
            continue

        base_score = fuzz.ratio(user_norm, t_norm) / 100.0

        title_season = _extract_season_number(t)
        if user_season is not None and title_season is not None:
            if user_season == title_season:
                base_score += 0.05
            else:
                base_score -= 0.20

        if base_score < 0:
            base_score = 0.0
        elif base_score > 1:
            base_score = 1.0

        if base_score > best:
            best = base_score

    return best


# ========== High-level search over providers ==========

def search_all_providers(query: str, limit=10):
    """
    Search Jikan + Kitsu + AniList, combine into one DataFrame with a simple_match column.
    """
    jikan_df = jikan_search_anime(query, limit=limit)
    kitsu_df = kitsu_search_anime(query, limit=limit)
    anilist_df = anilist_search_anime(query, limit=limit)

    frames = []
    if not jikan_df.empty:
        frames.append(jikan_df)
    if not kitsu_df.empty:
        frames.append(kitsu_df)
    if not anilist_df.empty:
        frames.append(anilist_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    combined["all_titles"] = combined["all_titles"].apply(
        lambda v: v if isinstance(v, (list, tuple, set)) else [v]
    )
    combined["simple_match"] = combined["all_titles"].apply(
        lambda titles: _title_match_score(query, titles)
    )

    exact_mask = combined["simple_match"] >= 0.85
    if exact_mask.any():
        combined = combined[exact_mask].reset_index(drop=True)
    else:
        combined = combined[combined["simple_match"] >= 0.65].reset_index(drop=True)

    return combined


# ========== Grouping helpers (series / same title) ==========

def get_same_title_group_sorted(all_results_df, main_row):
    if all_results_df.empty or main_row is None:
        return pd.DataFrame()

    main_title = str(main_row.get("title") or "").strip().lower()

    def is_same_series(title):
        if not isinstance(title, str):
            return False
        t = title.strip().lower()
        return main_title in t or t in main_title

    group_df = all_results_df[all_results_df["title"].apply(is_same_series)].copy()

    if group_df.empty:
        return group_df

    group_df["start_date_parsed"] = pd.to_datetime(group_df["start_date"], errors="coerce")
    group_df = group_df.sort_values(
        by=["start_date_parsed", "type", "title"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    group_df.drop(columns=["start_date_parsed"], inplace=True, errors="ignore")
    return group_df


def get_series_group_with_relations(all_results_df, best_row):
    if all_results_df.empty or best_row is None:
        return pd.DataFrame()

    same_title_group = get_same_title_group_sorted(all_results_df, best_row)

    mal_id = None
    if best_row.get("provider") == "Jikan" and best_row.get("provider_id"):
        mal_id = best_row.get("provider_id")
    else:
        jikan_rows = all_results_df[all_results_df["provider"] == "Jikan"]
        if not jikan_rows.empty:
            mal_id = jikan_rows.iloc[0].get("provider_id")

    if not mal_id:
        return same_title_group

    rel_df = jikan_fetch_relations(mal_id)
    if rel_df.empty:
        return same_title_group

    combined = pd.concat([same_title_group, rel_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["provider", "provider_id", "title"])

    combined["start_date_parsed"] = pd.to_datetime(combined["start_date"], errors="coerce")
    combined = combined.sort_values(
        by=["start_date_parsed", "type", "title"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    combined.drop(columns=["start_date_parsed"], inplace=True, errors="ignore")

    return combined


# ========== Best synopsis helper ==========

def get_best_synopsis(all_results_df, best_row):
    """
    Prefer the synopsis on best_row; if empty, fall back to a Jikan row
    with the same title, if available.
    """
    syn = best_row.get("synopsis")
    if isinstance(syn, str) and syn.strip():
        return syn

    title = str(best_row.get("title") or "").strip().lower()
    if not title or all_results_df.empty:
        return ""

    jikan_mask = (
        (all_results_df["provider"] == "Jikan")
        & (all_results_df["title"].astype(str).str.strip().str.lower() == title)
    )
    jikan_rows = all_results_df[jikan_mask]
    if not jikan_rows.empty:
        js = jikan_rows.iloc[0].get("synopsis")
        if isinstance(js, str):
            return js
    return ""


# ========== Genre-based recommendations (More Like This) ==========

def get_genre_based_recommendations(all_results_df, main_row, top_n=30):
    if all_results_df.empty or main_row is None:
        return pd.DataFrame()

    main_genres = str(main_row.get("genres") or "").strip()

    if not main_genres:
        title_str = str(main_row.get("title") or "").strip().lower()
        if title_str:
            same_title_mask = (
                (all_results_df["provider"] == "Jikan")
                & (all_results_df["title"].astype(str).str.strip().str.lower() == title_str)
                & (all_results_df["genres"].astype(str).str.strip() != "")
            )
            fallback_df = all_results_df[same_title_mask]
            if not fallback_df.empty:
                main_genres = str(fallback_df.iloc[0].get("genres") or "")

        if not main_genres.strip():
            same_title_mask = (
                (all_results_df["title"].astype(str).str.strip().str.lower() ==
                 str(main_row.get("title") or "").strip().lower())
                & (all_results_df["genres"].astype(str).str.strip() != "")
            )
            if same_title_mask.any():
                fallback_row = all_results_df[same_title_mask].iloc[0]
                main_genres = str(fallback_row.get("genres") or "")

    if not main_genres.strip():
        return pd.DataFrame()

    main_genre_set = set(g.strip().lower() for g in main_genres.split(",") if g.strip())
    if not main_genre_set:
        return pd.DataFrame()

    genre_df = jikan_fetch_by_genres(main_genre_set, limit_per_genre=25, global_limit=200)

    frames = [all_results_df]
    if not genre_df.empty:
        frames.append(genre_df)

    combined = pd.concat(frames, ignore_index=True)

    main_title = str(main_row.get("title") or "").strip().lower()

    def is_main_series(title):
        if not isinstance(title, str):
            return False
        t = title.strip().lower()
        return main_title in t or t in main_title

    combined = combined[~combined["title"].apply(is_main_series)].copy()

    def shared_genre_count(genres_str):
        if not isinstance(genres_str, str) or not genres_str.strip():
            return 0
        g_set = set(g.strip().lower() for g in genres_str.split(",") if g.strip())
        return len(main_genre_set.intersection(g_set))

    combined["shared_genres"] = combined["genres"].apply(shared_genre_count)
    combined = combined[combined["shared_genres"] > 0]

    if combined.empty:
        return combined

    combined["start_date_parsed"] = pd.to_datetime(combined["start_date"], errors="coerce")
    combined = combined.sort_values(
        by=["shared_genres", "score", "start_date_parsed"],
        ascending=[False, False, True],
        na_position="last",
    )

    combined.drop(columns=["start_date_parsed"], inplace=True, errors="ignore")
    combined = combined.reset_index(drop=True)

    return combined.head(top_n)


# ========== CLI demo (optional) ==========

def cli_demo():
    q = input("Enter anime title: ").strip()
    if not q:
        print("Empty query.")
        return

    results = search_all_providers(q, limit=15)
    if results.empty:
        print("No reasonably matching results found.")
        return

    print("\nTop matches:")
    for idx, row in results.head(10).iterrows():
        print(f"{idx}: {row['title']} [{row['provider']}] score={row['simple_match']:.2f}")

    best = results.iloc[0]
    print("\nBest match details:")
    print(f"Title: {best['title']}")
    print(f"Provider: {best['provider']}")
    print(f"Score: {best.get('score')}")
    print(f"Episodes: {best.get('total_episodes')}")
    print(f"Status: {best.get('status')}")
    syn = get_best_synopsis(results, best)
    if syn:
        print(f"Synopsis: {syn[:200]}...")

    print("\nSeries group (including relations):")
    series_df = get_series_group_with_relations(results, best)
    if not series_df.empty:
        for _, row in series_df.iterrows():
            print(f"- {row['title']} [{row['provider']}] type={row['type']} eps={row['total_episodes']}")

    print("\nMore Like This (genre-based):")
    genre_recs = get_genre_based_recommendations(results, best, top_n=10)
    if genre_recs.empty:
        print("Not enough genre information to generate recommendations.")
    else:
        for _, row in genre_recs.iterrows():
            print(f"- {row['title']} [{row['provider']}] shared_genres={row['shared_genres']}")


if __name__ == "__main__":
    cli_demo()
