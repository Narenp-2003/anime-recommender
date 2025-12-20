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
        # --- Best match + timeline side by side ---
        st.markdown("---")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Best Match")

            best = results.iloc[0]

            st.write(f"**Title:** {best['title']}")
            st.write(f"**Provider:** {best['provider']}")
            if pd.notna(best.get("score")):
                st.write(f"**Score:** {best['score']}")
            if pd.notna(best.get("total_episodes")):
                st.write(f"**Total episodes:** {best['total_episodes']}")

            # Normalize status for best match
            status_val = best.get("status")
            if (
                not isinstance(status_val, str)
                or not status_val.strip()
                or status_val == "None"
            ):
                status_val = "Currently airing"
            st.write(f"**Status:** {status_val}")

            if isinstance(best.get("synopsis"), str) and best["synopsis"].strip():
                st.write(f"**Synopsis:** {best['synopsis']}")

            # Fuzzy match score + warning
            if pd.notna(best.get("simple_match")):
                st.write(f"**Title match score:** {best['simple_match']:.2f}")
                if best["simple_match"] < 0.8:
                    st.warning(
                        "The match score is not very high. "
                        "Please confirm this is the correct show before trusting the timeline & recommendations."
                    )

        with col_right:
            st.subheader("Series timeline")

            series_df = get_series_group_with_relations(results, best)

            if series_df.empty:
                st.info("Not enough information to build a series timeline.")
            else:
                series_table = series_df[[
                    "title", "type", "total_episodes", "status", "score", "provider"
                ]].copy()

                # Normalize status for series table
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

        # All reasonably matching results
        st.subheader("All reasonably matching results")
        table_df = results[[
            "title", "type", "total_episodes", "status", "score", "provider", "simple_match"
        ]].copy()

        # Normalize status for results table
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

        # --- More Like This with filters ---
        st.markdown("---")
        st.subheader("More Like This (Similar Genre, Top 30)")

        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            type_filter = st.selectbox(
                "Type filter",
                options=["All", "TV", "Movie", "OVA", "ONA", "Special"],
                index=0,
            )

        with col_f2:
            min_score = st.slider(
                "Minimum score",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
            )

        with col_f3:
            only_airing = st.checkbox("Only currently airing", value=False)

        # Compute genre-based recommendations
        genre_recs = get_genre_based_recommendations(results, best, top_n=30)

        if genre_recs.empty:
            st.info("Not enough genre information to generate recommendations.")
        else:
            filtered = genre_recs.copy()

            # Apply type filter
            if type_filter != "All":
                filtered = filtered[
                    filtered["type"].astype(str).str.upper() == type_filter.upper()
                ]

            # Apply score filter
            if pd.notna(min_score) and min_score > 0:
                filtered = filtered[filtered["score"].fillna(0) >= min_score]

            # Apply airing filter
            if only_airing:
                filtered_status = filtered["status"].astype(str).str.lower()
                mask_airing = (
                    filtered_status.str.contains("airing")
                    | filtered_status.str.contains("ongoing")
                )
                filtered = filtered[mask_airing]

            if filtered.empty:
                st.info("No recommendations match the selected filters.")
            else:
                recs_table = filtered[[
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

                # Reset index so numbering is 0,1,2,...
                recs_table = recs_table.reset_index(drop=True)

                st.dataframe(recs_table)
else:
    st.info("Type an anime title above to get started.")
