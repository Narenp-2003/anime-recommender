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
st.write("Enter an anime name. The app will query multiple sources and show what is actually known.")

# Single text input with 200 character limit
query = st.text_input(
    "Anime name:",
    value="That Time I Got Reincarnated as a Slime",
    max_chars=200
)

search_limit = st.slider(
    "Number of results per provider",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

if st.button("Search"):
    if not query.strip():
        st.error("Please enter an anime name.")
    else:
        with st.spinner("Searching across multiple providers..."):
            combined = search_all_providers(query, limit=search_limit)

        if combined.empty:
            st.error("No reasonably matching results found.")
        else:
            combined["score_num"] = pd.to_numeric(combined["score"], errors="coerce")

            combined_sorted = combined.sort_values(
                by=["simple_match", "score_num"],
                ascending=[False, False]
            )

            best = combined_sorted.iloc[0]

            st.subheader("Best Match")

            st.write(f"**Title:** {best['title']}")
            st.write(f"**Provider:** {best['provider']}")
            st.write(f"**Score:** {best['score']}")

            match_strength = float(best.get("simple_match") or 0)
            st.write(f"**Title match score:** {match_strength:.2f} (1.0 = perfect match)")
            if match_strength < 0.85:
                st.info(
                    "This is a partial/approximate title match. "
                    "Check the title and provider before trusting the timeline and recommendations."
                )

            total_eps = best.get("total_episodes")
            if pd.isna(total_eps) or total_eps is None:
                st.write("**Total episodes:** Not provided by source (often unknown for ongoing shows)")
            else:
                st.write(f"**Total episodes:** {total_eps}")

            status_text = str(best.get("status") or "").strip()
            if status_text:
                st.write(f"**Status:** {status_text}")
            else:
                st.write("**Status:** Not provided by source")

            st.write("**Next episode / season info:** This project does not guess dates; use official announcements or schedule sites.")

            synopsis = best.get("synopsis")
            if isinstance(synopsis, str) and synopsis.strip():
                st.markdown("**Summary:**")
                st.write(synopsis)
            else:
                st.write("Summary not available from these APIs.")

            # --- Same series entries + relations ---
            st.markdown("---")
            st.subheader("This Series: Seasons / Movies / Specials (by release date)")

            same_series_df = get_series_group_with_relations(combined_sorted, best)
            if same_series_df.empty:
                st.write("No additional entries for this series found across providers.")
            else:
                same_series_df = same_series_df.reset_index(drop=True)
                series_table = same_series_df[[
                    "title", "type", "total_episodes", "status", "start_date", "end_date", "provider"
                ]].copy()
                series_table.rename(columns={
                    "title": "Title",
                    "type": "Type",
                    "total_episodes": "Total episodes (raw)",
                    "status": "Status (raw)",
                    "start_date": "Start date",
                    "end_date": "End date",
                    "provider": "Provider"
                }, inplace=True)
                st.dataframe(series_table)

            # --- Genre-based recommendations (top 30) ---
            st.markdown("---")
            st.subheader("More Like This (Similar Genre, Top 30)")

            genre_recs = get_genre_based_recommendations(combined_sorted, best, top_n=30)
            if genre_recs.empty:
                st.write(
                    "Not enough usable genre information across providers for this title. "
                    "Try another anime or a more popular series."
                )
            else:
                recs_table = genre_recs[[
                    "title", "score", "total_episodes", "status", "genres", "provider"
                ]].copy()
                recs_table.rename(columns={
                    "title": "Title",
                    "score": "Score",
                    "total_episodes": "Total episodes (raw)",
                    "status": "Status (raw)",
                    "genres": "Genres",
                    "provider": "Provider"
                }, inplace=True)
                st.dataframe(recs_table)

            # --- All close matches (overview) ---
            st.markdown("---")
            st.subheader("All Close Matches (All Providers)")

            table_df = combined_sorted[[
                "title", "score", "total_episodes", "status", "provider", "simple_match"
            ]].copy()
            table_df = table_df.reset_index(drop=True)
            table_df.rename(columns={
                "title": "Title",
                "score": "Score",
                "total_episodes": "Total episodes (raw)",
                "status": "Status (raw)",
                "provider": "Provider",
                "simple_match": "Title match score"
            }, inplace=True)
            st.dataframe(table_df)
