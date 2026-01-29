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
    safe_df = results_df[core_cols].copy()
    return get_genre_based_recommendations(safe_df, best_row, top_n=top_n)
