import streamlit as st
import pandas as pd

from anime_recommender import (
    search_all_providers,
    get_genre_based_recommendations,
    get_same_title_group_sorted,
    get_series_group_with_relations,
    get_best_synopsis,
)

from offline_mal_model import build_offline_model
from sklearn.metrics.pairwise import linear_kernel

from movie_recommender import (
    search_movies_tmdb,
    search_tv_tmdb,
    get_movie_recommendations,
    get_tv_recommendations,
)

from online_movie_sources import (
    fetch_omdb_for_tmdb,
    fetch_streaming_availability,
    summarize_streaming_providers,
)

from movie_offline_model import (
    build_offline_movie_model,
    build_offline_tv_model,
    get_offline_similar,
)

CONTENT_TYPES = ["Anime", "Movies", "TV"]

st.set_page_config(page_title="Watch Recommender", page_icon="🎬", layout="wide")

st.title("Watch Recommender 🎬")
st.caption("Find anime, movies, and TV shows similar to what you're watching.")

# ---- Session state ----
if "current_anime" not in st.session_state:
    st.session_state.current_anime = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "current_content_type" not in st.session_state:
    st.session_state.current_content_type = "Anime"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "content_type_radio" not in st.session_state:
    st.session_state.content_type_radio = st.session_state.current_content_type

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Search & settings")

    content_type = st.radio(
        "Content type",
        CONTENT_TYPES,
        index=CONTENT_TYPES.index(st.session_state.current_content_type),
        key="content_type_radio",
    )

    placeholder = {
        "Anime": "Search anime title",
        "Movies": "Search movie title",
        "TV": "Search TV show title",
    }[content_type]

    query = st.text_input(placeholder, st.session_state.last_query)
    search_limit = st.slider("Max results per provider", 5, 30, 15, 5)
    st.markdown("---")
    theme_pref = st.radio(
        "Theme preference (for future)", ["System", "Light", "Dark"], index=0
    )
    st.caption("Use the main area for timelines and recommendations.")

    if st.button("Search", type="primary"):
        st.session_state.last_query = query
        st.session_state.current_anime = None
        st.session_state.current_content_type = st.session_state.content_type_radio
        if query.strip() and query not in st.session_state.search_history:
            st.session_state.search_history.append(query)

    if st.session_state.search_history:
        st.subheader("Recent searches")
        for q in reversed(st.session_state.search_history[-5:]):
            if st.button(q, key=f"hist_{q}"):
                st.session_state.last_query = q
                st.session_state.current_anime = None
                st.experimental_rerun()


def _normalize_status_col(s):
    return (
        s.fillna("Currently airing")
        .replace(
            {
                "None": "Currently airing",
                "Ongoing": "Currently airing",
                "On going": "Currently airing",
            }
        )
    )


@st.cache_resource
def get_offline_model():
    return build_offline_model()


@st.cache_data(show_spinner=False)
def cached_genre_recs(results_df: pd.DataFrame, best_row: dict, top_n: int):
    df = results_df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: ", ".join(map(str, x))
                if isinstance(x, (list, tuple, set))
                else x
            )

    core_cols = [
        "provider",
        "provider_id",
        "title",
        "all_titles",
        "genres",
        "score",
        "status",
        "type",
        "total_episodes",
        "start_date",
    ]
    existing = [c for c in core_cols if c in df.columns]
    safe_df = df[existing].copy()
    return get_genre_based_recommendations(safe_df, best_row, top_n=top_n)


# ---- Main ----
active_query = st.session_state.last_query or query
active_type = st.session_state.get("content_type_radio", "Anime")

if active_query.strip():
    with st.spinner("Searching..."):
        if active_type == "Anime":
            results = search_all_providers(active_query, limit=search_limit)
        elif active_type == "Movies":
            results = search_movies_tmdb(active_query, limit=search_limit)
        else:
            results = search_tv_tmdb(active_query, limit=search_limit)

    if results.empty:
        st.error("No reasonably matching results found. Try another title or spelling.")
    else:
        if active_type == "Anime":
            # ---- Anime flow ----
            if st.session_state.current_anime is None:
                best = results.iloc[0]
                st.session_state.current_anime = best.to_dict()
            else:
                best = pd.Series(st.session_state.current_anime)

            st.markdown("---")
            st.header("Best match & series")

            col_left, col_right = st.columns([2, 2])

            with col_left:
                st.subheader("Best match")

                img_url = None
                if "image_url" in best.index and pd.notna(best["image_url"]):
                    img_url = best["image_url"]
                if img_url:
                    st.image(img_url, width=220)

                st.write(f"**Title:** {best['title']}")
                st.write(f"**Provider:** {best['provider']}")
                if pd.notna(best.get("score")):
                    st.write(f"**Score:** {best['score']}")
                if pd.notna(best.get("total_episodes")):
                    st.write(f"**Total episodes:** {best['total_episodes']}")

                status_val = best.get("status")
                if (
                    not isinstance(status_val, str)
                    or not status_val.strip()
                    or status_val == "None"
                ):
                    status_val = "Currently airing"
                st.write(f"**Status:** {status_val}")

                info_bits = []
                if pd.notna(best.get("type")):
                    info_bits.append(str(best["type"]))

                if pd.notna(best.get("start_date")):
                    year = str(best["start_date"]).split("-")[0]
                    if year and year != "NaT":
                        info_bits.append(year)

                status_short = status_val
                if isinstance(status_short, str):
                    status_short = status_short.replace("Currently airing", "Airing")
                    info_bits.append(status_short)

                if info_bits:
                    st.write(" · ".join(info_bits))

                synopsis_text = get_best_synopsis(results, best)
                if isinstance(synopsis_text, str) and synopsis_text.strip():
                    with st.expander("Synopsis", expanded=False):
                        st.write(synopsis_text)

                if pd.notna(best.get("simple_match")):
                    st.write(f"**Title match score:** {best['simple_match']:.2f}")
                    if best["simple_match"] < 0.8:
                        st.warning(
                            "Match score is not very high. "
                            "Check this is the right show before trusting timeline & recommendations."
                        )

            with col_right:
                st.subheader("Series timeline")
                series_df = get_series_group_with_relations(results, best)

                if series_df.empty:
                    st.info("Not enough information to build a series timeline.")
                else:
                    base_cols = [
                        "title",
                        "type",
                        "total_episodes",
                        "status",
                        "score",
                        "provider",
                    ]
                    cols = base_cols
                    if "image_url" in series_df.columns:
                        cols = ["image_url"] + base_cols

                    series_table = series_df[cols].copy()
                    series_table["status"] = _normalize_status_col(
                        series_table["status"]
                    )
                    series_table.rename(
                        columns={
                            "title": "Title",
                            "type": "Type",
                            "total_episodes": "Episodes",
                            "status": "Status",
                            "score": "Score",
                            "provider": "Provider",
                            "image_url": "Image",
                        },
                        inplace=True,
                    )
                    series_table = series_table.reset_index(drop=True)
                    series_table.insert(0, "Rank", series_table.index + 1)
                    st.dataframe(
                        series_table,
                        column_config={
                            "Image": st.column_config.ImageColumn(
                                "Image", width="small"
                            ),
                        },
                        width="stretch",
                    )

            # ---- Anime search results ----
            st.markdown("---")
            st.header("Search results")

            provider_options = ["All"] + sorted(
                results["provider"].astype(str).unique().tolist()
            )
            provider_filter = st.selectbox(
                "Provider filter",
                options=provider_options,
                index=0,
            )

            filtered_results = results.copy()
            if provider_filter != "All":
                filtered_results = filtered_results[
                    filtered_results["provider"].astype(str) == provider_filter
                ]

            if filtered_results.empty:
                st.info("No results match the selected provider.")
            else:
                table_df = filtered_results[
                    [
                        "title",
                        "type",
                        "total_episodes",
                        "status",
                        "score",
                        "provider",
                        "simple_match",
                    ]
                ].copy()
                table_df["status"] = _normalize_status_col(table_df["status"])
                table_df.rename(
                    columns={
                        "title": "Title",
                        "type": "Type",
                        "total_episodes": "Episodes",
                        "status": "Status",
                        "score": "Score",
                        "provider": "Provider",
                        "simple_match": "Title match score",
                    },
                    inplace=True,
                )
                table_df = table_df.reset_index(drop=True)
                table_df.insert(0, "Rank", table_df.index + 1)
                st.dataframe(table_df, width="stretch")

            # ---- Anime: More like this ----
            st.markdown("---")
            st.header("More like this")

            mode_tab = st.radio(
                "Recommendation source",
                ["API-based (genres)", "Offline MAL model (TF-IDF)"],
                index=0,
            )

            if mode_tab == "API-based (genres)":
                top_col1, top_col2 = st.columns([2, 1])
                with top_col1:
                    rec_mode = st.radio(
                        "Recommendation mode",
                        ["Cross-genre (default)", "Same-franchise first", "Hidden gems"],
                        index=0,
                    )
                with top_col2:
                    top_n = st.slider("Max recommendations", 10, 50, 30, 5)

                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                with col_f1:
                    type_filter = st.selectbox(
                        "Type",
                        options=["All", "TV", "Movie", "OVA", "ONA", "Special"],
                        index=0,
                    )
                with col_f2:
                    min_score = st.slider("Min score", 0.0, 10.0, 0.0, 0.5)
                with col_f3:
                    min_eps = st.number_input("Min episodes", 1, 300, 1)
                with col_f4:
                    only_airing = st.checkbox("Only currently airing", value=False)

                genre_recs = cached_genre_recs(results, best.to_dict(), top_n)

                if genre_recs.empty:
                    st.info("Not enough genre information to generate recommendations.")
                else:
                    filtered = genre_recs.copy()

                    all_genres = (
                        results["genres"]
                        .dropna()
                        .astype(str)
                        .str.split(",")
                        .explode()
                        .str.strip()
                    )
                    all_genres = sorted(g for g in all_genres.unique() if g)

                    preferred_genres = st.multiselect(
                        "Preferred genres (optional)",
                        options=all_genres,
                        default=[],
                    )

                    if preferred_genres:
                        pg_set = {g.lower() for g in preferred_genres}

                        def matches_pref(gstr):
                            if not isinstance(gstr, str):
                                return False
                            cur = {
                                g.strip().lower()
                                for g in gstr.split(",")
                                if g.strip()
                            }
                            return bool(pg_set & cur)

                        filtered = filtered[filtered["genres"].apply(matches_pref)]

                    rec_provider_opts = ["All"] + sorted(
                        filtered["provider"].astype(str).unique().tolist()
                    )
                    rec_provider = st.selectbox(
                        "Recommendation providers",
                        options=rec_provider_opts,
                        index=0,
                        key="rec_provider",
                    )
                    if rec_provider != "All":
                        filtered = filtered[
                            filtered["provider"].astype(str) == rec_provider
                        ]

                    if type_filter != "All":
                        filtered = filtered[
                            filtered["type"]
                            .astype(str)
                            .str.upper()
                            == type_filter.upper()
                        ]

                    if min_score > 0:
                        filtered = filtered[
                            filtered["score"].fillna(0) >= min_score
                        ]

                    filtered = filtered[
                        filtered["total_episodes"].fillna(0) >= min_eps
                    ]

                    if only_airing:
                        status_series = filtered["status"].astype(str).str.lower()
                        mask_airing = status_series.str.contains(
                            "airing"
                        ) | status_series.str.contains("ongoing")
                        filtered = filtered[mask_airing]

                    if rec_mode == "Same-franchise first":
                        main_title_lc = best["title"].strip().lower()
                        filtered["same_franchise"] = (
                            filtered["title"]
                            .astype(str)
                            .str.lower()
                            .apply(
                                lambda t: main_title_lc in t or t in main_title_lc
                            )
                        )
                        filtered = filtered.sort_values(
                            by=["same_franchise", "shared_genres", "score"],
                            ascending=[False, False, False],
                        )
                    elif rec_mode == "Hidden gems":
                        if "score" in filtered.columns:
                            filtered = filtered.sort_values(
                                by=["score"], ascending=True
                            )

                    if filtered.empty:
                        st.info("No recommendations match the selected filters.")
                    else:
                        if "rec_score" in filtered.columns:
                            base_rank = filtered["rec_score"].astype(float)
                        else:
                            score_norm = filtered["score"].fillna(0) / 10.0
                            max_shared = max(
                                filtered["shared_genres"].max(), 1
                            )
                            shared_norm = (
                                filtered["shared_genres"] / max_shared
                            )
                            base_rank = 0.6 * shared_norm + 0.4 * score_norm

                        filtered["rank_score"] = base_rank

                        filtered = filtered.sort_values(
                            "rank_score", ascending=False
                        )

                        cols = [
                            "title",
                            "score",
                            "total_episodes",
                            "status",
                            "genres",
                            "provider",
                            "shared_genres",
                            "rank_score",
                        ]
                        if "title_sim" in filtered.columns:
                            cols.insert(-1, "title_sim")

                        recs_table = filtered[cols].copy()
                        recs_table["status"] = _normalize_status_col(
                            recs_table["status"]
                        )
                        rename_map = {
                            "title": "Title",
                            "score": "Score",
                            "total_episodes": "Episodes",
                            "status": "Status",
                            "genres": "Genres",
                            "provider": "Provider",
                            "shared_genres": "Shared genres",
                            "rank_score": "Rank score",
                        }
                        if "title_sim" in filtered.columns:
                            rename_map["title_sim"] = "Title similarity"

                        recs_table.rename(columns=rename_map, inplace=True)
                        recs_table = recs_table.reset_index(drop=True)
                        recs_table.insert(0, "Rank", recs_table.index + 1)
                        st.dataframe(recs_table, width="stretch")

            else:
                st.subheader(
                    "Offline MAL content-based recommendations (MyAnimeList dataset)"
                )
                top_n_offline = st.slider(
                    "Max recommendations (offline)", 10, 50, 30, 5
                )

                with st.spinner(
                    "Loading offline model (first time may take a bit)..."
                ):
                    df_off, tfidf_matrix = get_offline_model()

                t = best["title"].strip().lower()
                mask = (
                    df_off["title_display"]
                    .astype(str)
                    .str.lower()
                    .str.contains(t)
                )
                candidates = df_off[mask]
                if candidates.empty:
                    st.info(
                        "No offline recommendations found for this title in the MAL dataset."
                    )
                else:
                    if "score" in candidates.columns:
                        best_idx = (
                            candidates["score"]
                            .fillna(0)
                            .astype(float)
                            .idxmax()
                        )
                    else:
                        best_idx = candidates.index[0]

                    if best_idx not in df_off.index:
                        st.info(
                            "No offline recommendations found for this title in the MAL dataset."
                        )
                    else:
                        cosine_sim = linear_kernel(
                            tfidf_matrix[best_idx], tfidf_matrix
                        ).flatten()
                        df_out = df_off.copy()
                        df_out["similarity"] = cosine_sim
                        df_out = df_out[df_out.index != best_idx]
                        df_out = df_out.sort_values(
                            "similarity", ascending=False
                        ).head(top_n_offline)

                        table = df_out.rename(
                            columns={
                                "title_display": "Title",
                                "score": "MAL Score",
                                "episodes": "Episodes",
                                "similarity": "Similarity",
                            }
                        )
                        table = table.reset_index(drop=True)
                        table.insert(0, "Rank", table.index + 1)
                        st.dataframe(table, width="stretch")

        else:
            # ---- Movie / TV flow ----
            best = results.iloc[0]

            year_val = None
            if active_type == "Movies" and pd.notna(best.get("release_date")):
                year_val = str(best["release_date"]).split("-")[0]
            elif active_type != "Movies" and pd.notna(best.get("first_air_date")):
                year_val = str(best["first_air_date"]).split("-")[0]
            year_int = int(year_val) if year_val and year_val.isdigit() else None

            omdb_data = fetch_omdb_for_tmdb(best["title"], year_int)
            streaming_data = fetch_streaming_availability(
                best["title"], year_int, country="IN"
            )
            providers, streaming_summary = summarize_streaming_providers(streaming_data)

            st.markdown("---")
            st.header(f"Best match ({active_type})")

            col1, col2 = st.columns([1, 2])
            with col1:
                if pd.notna(best.get("poster_url")):
                    st.image(best["poster_url"], width=220)
            with col2:
                st.write(f"**Title:** {best['title']}")
                if active_type == "Movies":
                    if pd.notna(best.get("release_date")):
                        st.write(f"**Release date:** {best['release_date']}")
                else:
                    if pd.notna(best.get("first_air_date")):
                        st.write(f"**First air date:** {best['first_air_date']}")
                if pd.notna(best.get("score")):
                    st.write(f"**TMDb score:** {best['score']:.1f}")
                if pd.notna(best.get("vote_count")):
                    st.write(f"**Votes:** {int(best['vote_count'])}")

                if omdb_data:
                    imdb_rating = omdb_data.get("imdbRating")
                    imdb_votes = omdb_data.get("imdbVotes")
                    if imdb_rating and imdb_rating != "N/A":
                        st.write(f"**IMDb rating:** {imdb_rating}")
                    if imdb_votes and imdb_votes != "N/A":
                        st.write(f"**IMDb votes:** {imdb_votes}")

                if providers:
                    links = []
                    for p in providers:
                        name = p.get("name")
                        url = p.get("url")
                        if not name:
                            continue
                        if url:
                            links.append(f"[{name}]({url})")
                        else:
                            links.append(name)
                    st.markdown("**Available on:** " + " · ".join(links))

                if isinstance(best.get("overview"), str) and best["overview"].strip():
                    with st.expander("Overview", expanded=False):
                        st.write(best["overview"])

            # ---- Movie/TV search results ----
            st.markdown("---")
            st.header("Search results")

            table_cols = ["title", "type", "score", "vote_count"]
            date_col = "release_date" if active_type == "Movies" else "first_air_date"
            if date_col in results.columns:
                table_cols.append(date_col)

            table_df = results[table_cols].copy()
            table_df.rename(
                columns={
                    "title": "Title",
                    "type": "Type",
                    "score": "TMDb score",
                    "vote_count": "Votes",
                    "release_date": "Release date",
                    "first_air_date": "First air date",
                },
                inplace=True,
            )
            table_df = table_df.reset_index(drop=True)
            table_df.insert(0, "Rank", table_df.index + 1)
            st.dataframe(table_df, width="stretch")

            # ---- Movie/TV: More like this ----
            st.markdown("---")
            st.header("More like this")

            mode_mv = st.radio(
                "Recommendation source",
                ["TMDb API", "TMDb + IMDb (top 10 only)", "Offline streaming catalog (TF-IDF)"],
                index=0,
            )

            if mode_mv == "TMDb API":
                if active_type == "Movies":
                    recs = get_movie_recommendations(best, limit=50)
                else:
                    recs = get_tv_recommendations(best, limit=50)

                if recs.empty:
                    st.info("No recommendations found from TMDb.")
                else:
                    colf1, colf2, colf3 = st.columns(3)
                    with colf1:
                        min_score = st.slider("Min TMDb score", 0.0, 10.0, 6.0, 0.5)
                    with colf2:
                        min_votes = st.number_input("Min votes", 0, 50000, 100)
                    with colf3:
                        year_filter = st.text_input("Year (optional)", "")

                    recs_f = recs.copy()
                    recs_f = recs_f[recs_f["score"].fillna(0) >= min_score]
                    recs_f = recs_f[recs_f["vote_count"].fillna(0) >= min_votes]

                    date_col_rec = (
                        "release_date" if active_type == "Movies" else "first_air_date"
                    )
                    if year_filter.strip() and date_col_rec in recs_f.columns:
                        y = year_filter.strip()
                        recs_f = recs_f[
                            recs_f[date_col_rec].astype(str).str.startswith(y)
                        ]

                    if recs_f.empty:
                        st.info("No recommendations match the selected filters.")
                    else:
                        score_norm = recs_f["score"].fillna(0) / 10.0
                        votes_raw = recs_f["vote_count"].fillna(0)
                        vmax = max(votes_raw.max(), 1)
                        votes_norm = votes_raw / vmax

                        recs_f["rank_score"] = 0.7 * score_norm + 0.3 * votes_norm
                        recs_f = recs_f.sort_values("rank_score", ascending=False)

                        rec_table_cols = [
                            "title",
                            "type",
                            "score",
                            "vote_count",
                            "rank_score",
                        ]
                        if date_col_rec in recs_f.columns:
                            rec_table_cols.append(date_col_rec)

                        recs_table = recs_f[rec_table_cols].copy()
                        recs_table.rename(
                            columns={
                                "title": "Title",
                                "type": "Type",
                                "score": "TMDb score",
                                "vote_count": "Votes",
                                "release_date": "Release date",
                                "first_air_date": "First air date",
                                "rank_score": "Rank score",
                            },
                            inplace=True,
                        )
                        recs_table = recs_table.reset_index(drop=True)
                        recs_table.insert(0, "Rank", recs_table.index + 1)
                        st.dataframe(recs_table, width="stretch")

            elif mode_mv == "TMDb + IMDb (top 10 only)":
                if active_type == "Movies":
                    recs = get_movie_recommendations(best, limit=50)
                else:
                    recs = get_tv_recommendations(best, limit=50)

                if recs.empty:
                    st.info("No recommendations found from TMDb.")
                else:
                    date_col_rec = (
                        "release_date" if active_type == "Movies" else "first_air_date"
                    )

                    top10 = recs.sort_values("score", ascending=False).head(10).copy()
                    imdb_ratings = []
                    for _, row in top10.iterrows():
                        y_val = None
                        if date_col_rec in row and pd.notna(row[date_col_rec]):
                            y_s = str(row[date_col_rec]).split("-")[0]
                            y_val = int(y_s) if y_s.isdigit() else None
                        od = fetch_omdb_for_tmdb(row["title"], y_val)
                        if od and od.get("imdbRating") and od.get("imdbRating") != "N/A":
                            imdb_ratings.append(od["imdbRating"])
                        else:
                            imdb_ratings.append(None)

                    top10["imdb_rating"] = imdb_ratings

                    rec_table_cols = [
                        "title",
                        "type",
                        "score",
                        "vote_count",
                        "imdb_rating",
                    ]
                    if date_col_rec in top10.columns:
                        rec_table_cols.append(date_col_rec)

                    recs_table = top10[rec_table_cols].copy()
                    recs_table.rename(
                        columns={
                            "title": "Title",
                            "type": "Type",
                            "score": "TMDb score",
                            "vote_count": "Votes",
                            "release_date": "Release date",
                            "first_air_date": "First air date",
                            "imdb_rating": "IMDb rating",
                        },
                        inplace=True,
                    )
                    recs_table = recs_table.reset_index(drop=True)
                    recs_table.insert(0, "Rank", recs_table.index + 1)
                    st.dataframe(recs_table, width="stretch")

            else:
                # Offline streaming catalog mode
                with st.spinner("Loading offline streaming catalog model..."):
                    if active_type == "Movies":
                        df_off, mat = build_offline_movie_model()
                    else:
                        df_off, mat = build_offline_tv_model()

                top_n_off = st.slider("Max recommendations (offline)", 10, 50, 30, 5)

                recs_off = get_offline_similar(df_off, mat, best["title"], top_n=top_n_off)
                if recs_off.empty:
                    st.info("No offline recommendations found for this title in the catalog.")
                else:
                    table = recs_off.rename(
                        columns={
                            "title": "Title",
                            "provider": "Provider",
                            "type": "Type",
                            "release_year": "Year",
                            "rating": "Rating",
                            "duration": "Duration",
                            "listed_in": "Genres",
                            "similarity": "Similarity",
                        }
                    )
                    table = table.reset_index(drop=True)
                    table.insert(0, "Rank", table.index + 1)
                    st.dataframe(table, width="stretch")

else:
    st.info("Type a title above to get started.")
