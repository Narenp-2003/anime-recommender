# app.py
import streamlit as st
import pandas as pd

from anime_recommender import (
    search_all_providers,
    get_genre_based_recommendations,
    get_same_title_group_sorted,
    get_series_group_with_relations,
)

st.set_page_config(page_title="Anime Recommender", page_icon="🎌")

st.title("Anime Recommender 🎌")
st.write("Search for an anime title to see best matches, timeline, and similar-genre recommendations.")

query = st.text_input("Enter anime title", "")

search_limit = st.slider("Max results per provider", min_value=5, max_value=30, value=15, step=5)

if query.strip():
    with st.spinner("Searching across providers..."):
        results = search_all_providers(query, limit=search_limit)

    if results.empty:
        st.error("No reasonably matching results found. Try another title or spelling.")
    else:
        # Best match
        best = results.iloc[0]

        st.subheader("Best Match")
        st.write(f"**Title:** {best['title']}")
        st.write(f"**Provider:** {best['provider']}")
        if pd.notna(best.get("score")):
            st.write(f"**Score:** {best['score']}")
        if pd.notna(best.get("total_episodes")):
            st.write(f"**Total episodes:** {best['total_episodes']}")
        if isinstance(best.get("status"), str) and best["status"].strip():
            # Normalize status here for the detail block as well
            status_val = best["status"]
            if not isinstance(status_val, str) or not status_val.strip() or status_val == "None":
                status_val = "Currently airing"
            st.write(f"**Status:** {status_val}")
        if isinstance(best.get("synopsis"), str) and best["synopsis"].strip():
            st.write(f"**Synopsis:** {best['synopsis']}")

        # Show fuzzy match score, with warning if low
        if pd.notna(best.get("simple_match")):
            st.write(f"**Title match score:** {best['simple_match']:.2f}")
            if best["simple_match"] < 0.8:
                st.warning(
                    "The match score is not very high. "
                    "Please check that this is the correct show before trusting the timeline & recommendations."
                )

        st.markdown("---")

        # Full results table
        st.subheader("All reasonably matching results")
        table_df = results[[
            "title", "type", "total_episodes", "status", "score", "provider", "simple_match"
        ]].copy()

        # Normalize status for the results table
        table_df["status"] = (
            table_df["status"]
            .fillna("Currently airing")
            .replace({
                "None": "Currently airing",
                "Ongoing": "Currently airing",
                "On going": "Currently airing",
            })
        )

        table_df.rename(columns={
            "title": "Title",
            "type": "Type",
            "total_episodes": "Episodes",
            "status": "Status",
            "score": "Score",
            "provider": "Provider",
            "simple_match": "Title match score",
        }, inplace=True)

        st.dataframe(table_df)

        st.markdown("---")

        # Series / timeline block (with relations)
        st.subheader("Series timeline (including related entries)")
        series_df = get_series_group_with_relations(results, best)

        if series_df.empty:
            st.info("Not enough information to build a series timeline.")
        else:
            series_table = series_df[[
                "title", "type", "total_episodes", "status", "score", "provider"
            ]].copy()

            # Normalize status for series timeline table
            series_table["status"] = (
                series_table["status"]
                .fillna("Currently airing")
                .replace({
                    "None": "Currently airing",
                    "Ongoing": "Currently airing",
                    "On going": "Currently airing",
                })
            )

            series_table.rename(columns={
                "title": "Title",
                "type": "Type",
                "total_episodes": "Episodes",
                "status": "Status",
                "score": "Score",
                "provider": "Provider",
            }, inplace=True)

            st.dataframe(series_table)

        st.markdown("---")

        # More Like This: genre-based
        st.subheader("More Like This (Similar Genre, Top 30)")
        genre_recs = get_genre_based_recommendations(results, best, top_n=30)

        if genre_recs.empty:
            st.info("Not enough genre information to generate recommendations.")
        else:
            recs_table = genre_recs[[
                "title", "score", "total_episodes", "status", "genres", "provider"
            ]].copy()

            # Normalize status for More Like This table
            recs_table["status"] = (
                recs_table["status"]
                .fillna("Currently airing")
                .replace({
                    "None": "Currently airing",
                    "Ongoing": "Currently airing",
                    "On going": "Currently airing",
                })
            )

            recs_table.rename(columns={
                "title": "Title",
                "score": "Score",
                "total_episodes": "Episodes",
                "status": "Status",
                "genres": "Genres",
                "provider": "Provider",
            }, inplace=True)

            # Reset index so numbering starts at 0,1,2,... in the UI
            recs_table = recs_table.reset_index(drop=True)

            st.dataframe(recs_table)
else:
    st.info("Type an anime title above to get started.")
