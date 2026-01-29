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


st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

st.title("Anime Recommender 🎌")
st.caption("Find the best entry point, series timeline, and similar-genre anime across multiple sources.")


# ---- Sidebar controls ----
with st.sidebar:
    st.header("Search & settings")
    query = st.text_input("Search anime title", "")
    search_limit = st.slider("Max results per provider", 5, 30, 15, 5)
    st.markdown("---")
    theme_pref = st.radio("Theme preference (for future)", ["System", "Light", "Dark"], index=0)
    st.caption("Use the main area for timelines and recommendations.")


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
    existing = [c for c in core_cols if c in results_df.columns]
    safe_df = results_df[existing].copy()
    return get_genre_based_recommendations(safe_df, best_row, top_n=top_n)


if query.strip():
    with st.spinner("Searching across providers..."):
        results = search_all_providers(query, limit=search_limit)

    if results.empty:
        st.error("No reasonably matching results found. Try another title or spelling.")
    else:
        best = results.iloc[0]

        # ---- Best match + series ----
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
            if not isinstance(status_val, str) or not status_val.strip() or status_val == "None":
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
                base_cols = ["title", "type", "total_episodes", "status", "score", "provider"]
                cols = base_cols
                if "image_url" in series_df.columns:
                    cols = ["image_url"] + base_cols

                series_table = series_df[cols].copy()
                series_table["status"] = _normalize_status_col(series_table["status"])
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
                st.dataframe(
                    series_table,
                    column_config={
                        "Image": st.column_config.ImageColumn("Image", width="small"),
                    },
                    use_container_width=True,
                )

        # ---- Search results ----
        st.markdown("---")
        st.header("Search results")

        provider_options = ["All"] + sorted(results["provider"].astype(str).unique().tolist())
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
                ["title", "type", "total_episodes", "status", "score", "provider", "simple_match"]
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
            st.dataframe(table_df, use_container_width=True)

        # ---- More like this (API + offline) ----
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
                        cur = {g.strip().lower() for g in gstr.split(",") if g.strip()}
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
                        filtered["type"].astype(str).str.upper() == type_filter.upper()
                    ]

                if min_score > 0:
                    filtered = filtered[filtered["score"].fillna(0) >= min_score]

                filtered = filtered[filtered["total_episodes"].fillna(0) >= min_eps]

                if only_airing:
                    status_series = filtered["status"].astype(str).str.lower()
                    mask_airing = status_series.str.contains("airing") | status_series.str.contains(
                        "ongoing"
                    )
                    filtered = filtered[mask_airing]

                if rec_mode == "Same-franchise first":
                    main_title_lc = best["title"].strip().lower()
                    filtered["same_franchise"] = filtered["title"].astype(str).str.lower().apply(
                        lambda t: main_title_lc in t or t in main_title_lc
                    )
                    filtered = filtered.sort_values(
                        by=["same_franchise", "shared_genres", "score"],
                        ascending=[False, False, False],
                    )
                elif rec_mode == "Hidden gems":
                    if "score" in filtered.columns:
                        filtered = filtered.sort_values(by=["score"], ascending=True)

                if filtered.empty:
                    st.info("No recommendations match the selected filters.")
                else:
                    if "rec_score" in filtered.columns:
                        base_rank = filtered["rec_score"].astype(float)
                    else:
                        score_norm = filtered["score"].fillna(0) / 10.0
                        max_shared = max(filtered["shared_genres"].max(), 1)
                        shared_norm = filtered["shared_genres"] / max_shared
                        base_rank = 0.6 * shared_norm + 0.4 * score_norm

                    filtered["rank_score"] = base_rank

                    filtered = filtered.sort_values("rank_score", ascending=False)

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
                    recs_table["status"] = _normalize_status_col(recs_table["status"])
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
                    st.dataframe(recs_table, use_container_width=True)

        else:
            st.subheader("Offline MAL content-based recommendations (MyAnimeList dataset)")
            top_n_offline = st.slider("Max recommendations (offline)", 10, 50, 30, 5)

            with st.spinner("Loading offline model (first time may take a bit)..."):
                df_off, tfidf_matrix = get_offline_model()

            t = best["title"].strip().lower()
            mask = df_off["title_display"].astype(str).str.lower().str.contains(t)
            candidates = df_off[mask]
            if candidates.empty:
                st.info("No offline recommendations found for this title in the MAL dataset.")
            else:
                if "score" in candidates.columns:
                    best_idx = candidates["score"].fillna(0).astype(float).idxmax()
                else:
                    best_idx = candidates.index[0]

                if best_idx not in df_off.index:
                    st.info("No offline recommendations found for this title in the MAL dataset.")
                else:
                    cosine_sim = linear_kernel(tfidf_matrix[best_idx], tfidf_matrix).flatten()
                    df_out = df_off.copy()
                    df_out["similarity"] = cosine_sim
                    df_out = df_out[df_out.index != best_idx]
                    df_out = df_out.sort_values("similarity", ascending=False).head(top_n_offline)

                    table = df_out.rename(
                        columns={
                            "title_display": "Title",
                            "score": "MAL Score",
                            "episodes": "Episodes",
                            "similarity": "Similarity",
                        }
                    )
                    st.dataframe(table, use_container_width=True)

else:
    st.info("Type an anime title above to get started.")
