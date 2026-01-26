import re
import requests
import pandas as pd
from rapidfuzz import fuzz  # fuzzy matching

JIKAN_BASE_URL = "https://api.jikan.moe/v4/anime"
KITSU_BASE_URL = "https://kitsu.io/api/edge/anime"
ANIMEDB_BASE_URL = "https://animedb.docs.apiary.io"  # placeholder
ANILIST_URL = "https://graphql.anilist.co"


# ========= small utilities =========


def _safe_get(url, params=None, headers=None, timeout=10):
    try:
        resp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


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
    if not isinstance(title, str):
        return None
    t = title.lower()
    for pat in [
        r"season\s+(\d+)",
        r"\bs(\d)\b",
        r"(\d+)(st|nd|rd|th)\s+season",
    ]:
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return None


def _collect_titles(primary: str | None, titles_iterable) -> list[str]:
    all_titles = set()
    if titles_iterable:
        for t in titles_iterable:
            v = t.get("title") if isinstance(t, dict) else t
            if isinstance(v, str) and v.strip():
                all_titles.add(v.strip())
    if isinstance(primary, str) and primary.strip():
        all_titles.add(primary.strip())
    return list(all_titles)


def _extract_genres_from_blocks(*blocks) -> str:
    names = []
    for block in blocks:
        for g in (block or []):
            nm = g.get("name")
            if isinstance(nm, str):
                names.append(nm.strip())
    return ", ".join(sorted(set(names)))


def _parse_start_date_from_aired(aired: dict | None) -> str | None:
    if not isinstance(aired, dict):
        return None
    val = (aired.get("from") or "")[:10]
    return val or None


def _title_match_score(user_title: str, all_titles):
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

        base_score = max(0.0, min(1.0, base_score))
        best = max(best, base_score)

    return best


# ========= Jikan =========


def jikan_search_anime(query, limit=10):
    params = {"q": query, "limit": limit, "order_by": "score", "sort": "desc"}
    data = _safe_get(JIKAN_BASE_URL, params=params)
    if not data or "data" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["data"]:
        title_str = item.get("title") or ""
        titles = item.get("titles") or []
        all_titles = _collect_titles(title_str, titles)

        genres_str = _extract_genres_from_blocks(
            item.get("genres"),
            item.get("explicit_genres"),
            item.get("themes"),
            item.get("demographics"),
        )

        img = (item.get("images") or {}).get("jpg", {})  # image block
        image_url = img.get("image_url")

        rows.append(
            {
                "provider": "Jikan",
                "provider_id": item.get("mal_id"),
                "title": title_str,
                "all_titles": all_titles,
                "type": item.get("type") or "",
                "total_episodes": item.get("episodes"),
                "status": item.get("status") or "",
                "score": item.get("score"),
                "genres": genres_str,
                "start_date": _parse_start_date_from_aired(item.get("aired")),
                "synopsis": item.get("synopsis") or "",
                "image_url": image_url,
                "raw": item,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def jikan_fetch_relations(mal_id):
    if not mal_id:
        return pd.DataFrame()

    data = _safe_get(f"{JIKAN_BASE_URL}/{mal_id}/full")
    if not data or "data" not in data:
        return pd.DataFrame()
    data = data["data"]

    relations = data.get("relations") or []
    rows = []

    for rel in relations:
        relation_type = rel.get("relation") or ""
        for entry in rel.get("entry") or []:
            e_id = entry.get("mal_id")
            if not e_id:
                continue
            rel_data = _safe_get(f"{JIKAN_BASE_URL}/{e_id}")
            if not rel_data or "data" not in rel_data:
                continue
            item = rel_data["data"]

            title = item.get("title") or ""
            titles_block = item.get("titles") or []
            all_titles = _collect_titles(title, titles_block)
            genres_str = _extract_genres_from_blocks(
                item.get("genres"),
                item.get("explicit_genres"),
                item.get("themes"),
                item.get("demographics"),
            )

            img = (item.get("images") or {}).get("jpg", {})
            image_url = img.get("image_url")

            rows.append(
                {
                    "provider": "JikanRel",
                    "provider_id": e_id,
                    "title": title,
                    "all_titles": all_titles,
                    "type": item.get("type") or "",
                    "total_episodes": item.get("episodes"),
                    "status": item.get("status") or "",
                    "score": item.get("score"),
                    "genres": genres_str,
                    "start_date": _parse_start_date_from_aired(item.get("aired")),
                    "synopsis": item.get("synopsis") or "",
                    "relation_type": relation_type,
                    "image_url": image_url,
                    "raw": item,
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["provider_id", "title"])


def jikan_fetch_by_genres(genre_names, limit_per_genre=25, global_limit=200):
    if not genre_names:
        return pd.DataFrame()

    genre_list_url = "https://api.jikan.moe/v4/genres/anime"
    genre_data = _safe_get(genre_list_url)
    if not genre_data or "data" not in genre_data:
        return pd.DataFrame()

    name_to_id = {
        (g.get("name") or "").strip().lower(): g.get("mal_id")
        for g in genre_data["data"]
        if isinstance(g.get("name"), str) and g.get("mal_id") is not None
    }

    genre_ids = [
        str(name_to_id[n.strip().lower()])
        for n in genre_names
        if name_to_id.get(n.strip().lower()) is not None
    ]

    if not genre_ids:
        return pd.DataFrame()

    seen_ids, rows = set(), []
    for gid in genre_ids:
        params = {"genres": gid, "order_by": "score", "sort": "desc", "limit": limit_per_genre}
        data = _safe_get(JIKAN_BASE_URL, params=params)
        if not data or "data" not in data:
            continue

        for item in data["data"]:
            mal_id = item.get("mal_id")
            if not mal_id or mal_id in seen_ids:
                continue
            seen_ids.add(mal_id)

            title_str = item.get("title") or ""
            titles = item.get("titles") or []
            all_titles = _collect_titles(title_str, titles)
            genres_str = _extract_genres_from_blocks(
                item.get("genres"),
                item.get("explicit_genres"),
                item.get("themes"),
                item.get("demographics"),
            )

            img = (item.get("images") or {}).get("jpg", {})
            image_url = img.get("image_url")

            rows.append(
                {
                    "provider": "JikanGenre",
                    "provider_id": mal_id,
                    "title": title_str,
                    "all_titles": all_titles,
                    "type": item.get("type") or "",
                    "total_episodes": item.get("episodes"),
                    "status": item.get("status") or "",
                    "score": item.get("score"),
                    "genres": genres_str,
                    "start_date": _parse_start_date_from_aired(item.get("aired")),
                    "synopsis": item.get("synopsis") or "",
                    "image_url": image_url,
                    "raw": item,
                }
            )

            if len(rows) >= global_limit:
                break
        if len(rows) >= global_limit:
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ========= Kitsu =========


def kitsu_search_anime(query, limit=10):
    params = {"filter[text]": query, "page[limit]": limit}
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
            if isinstance(val, str) and val.strip():
                all_titles.add(val.strip())
        if canonical_title:
            all_titles.add(canonical_title.strip())

        poster = attrs.get("posterImage") or {}
        image_url = poster.get("small") or poster.get("original")

        score = attrs.get("averageRating")
        rows.append(
            {
                "provider": "Kitsu",
                "provider_id": item.get("id"),
                "title": title_str,
                "all_titles": list(all_titles),
                "type": attrs.get("showType") or "",
                "total_episodes": attrs.get("episodeCount"),
                "status": attrs.get("status") or "",
                "score": float(score) / 10.0 if score else None,
                "genres": "",  # not fetched
                "start_date": attrs.get("startDate"),
                "synopsis": attrs.get("synopsis") or "",
                "image_url": image_url,
                "raw": item,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ========= AniList =========


def anilist_search_anime(query, limit=10):
    if not query or not isinstance(query, str):
        return pd.DataFrame()

    graphql_query = """
    query ($search: String!, $perPage: Int!) {
      Page(page: 1, perPage: $perPage) {
        media(search: $search, type: ANIME) {
          id
          idMal
          title { romaji english native }
          format
          episodes
          status
          averageScore
          genres
          startDate { year month day }
          description(asHtml: false)
          coverImage { large extraLarge }
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

    media_list = (data or {}).get("data", {}).get("Page", {}).get("media") or []
    if not media_list:
        return pd.DataFrame()

    rows = []
    for m in media_list:
        titles = m.get("title") or {}
        t_romaji = titles.get("romaji") or ""
        t_english = titles.get("english") or ""
        t_native = titles.get("native") or ""
        primary_title = t_english or t_romaji or t_native

        all_titles = {
            t.strip()
            for t in (t_romaji, t_english, t_native)
            if isinstance(t, str) and t.strip()
        }

        score = m.get("averageScore")
        genres_str = ", ".join(
            sorted({g.strip() for g in (m.get("genres") or []) if isinstance(g, str)})
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

        cover = m.get("coverImage") or {}
        image_url = cover.get("extraLarge") or cover.get("large")

        rows.append(
            {
                "provider": "AniList",
                "provider_id": m.get("id"),
                "title": primary_title,
                "all_titles": list(all_titles),
                "type": m.get("format") or "",
                "total_episodes": m.get("episodes"),
                "status": (m.get("status") or "").upper(),
                "score": float(score) / 10.0 if score is not None else None,
                "genres": genres_str,
                "start_date": start_date,
                "synopsis": desc if isinstance(desc, str) else "",
                "image_url": image_url,
                "raw": m,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ========= high-level search =========


def search_all_providers(query: str, limit=10):
    jikan_df = jikan_search_anime(query, limit=limit)
    kitsu_df = kitsu_search_anime(query, limit=limit)
    anilist_df = anilist_search_anime(query, limit=limit)

    frames = [df for df in (jikan_df, kitsu_df, anilist_df) if not df.empty]
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


# ========= grouping / relations =========


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


# ========= synopsis helper =========


def get_best_synopsis(all_results_df, best_row):
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


# ========= genre-based recs =========


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
                all_results_df["title"].astype(str).str.strip().str.lower()
                == str(main_row.get("title") or "").strip().lower()
            ) & (all_results_df["genres"].astype(str).str.strip() != "")
            if same_title_mask.any():
                main_genres = str(all_results_df[same_title_mask].iloc[0].get("genres") or "")

    if not main_genres.strip():
        return pd.DataFrame()

    main_genre_set = {g.strip().lower() for g in main_genres.split(",") if g.strip()}
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
        g_set = {g.strip().lower() for g in genres_str.split(",") if g.strip()}
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
    ).reset_index(drop=True)
    combined.drop(columns=["start_date_parsed"], inplace=True, errors="ignore")

    return combined.head(top_n)


# ========= CLI demo =========


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
