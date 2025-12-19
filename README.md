# Anime Recommender (Multi-API, Python + Streamlit)

A simple web app that searches anime titles across multiple public APIs (Jikan + Kitsu + placeholder 3rd source), picks the best match, and shows related entries and similar-genre recommendations.

## Features

- Multi-provider search:
  - Jikan (MyAnimeList unofficial) for detailed titles, genres, and dates.
  - Kitsu for additional metadata.
  - Third provider hook ready (AnimeDb-style) for future extension.[web:99][web:106][web:101][web:104][web:140]

- Exact title matching:
  - Prioritizes exact and soft-exact matches (case-insensitive, ignores small punctuation).
  - Filters out unrelated entries like "Pokémon Violet" when searching "Violet Evergarden".[web:208][web:162]

- Series timeline:
  - Groups entries belonging to the same series (TV + movies/OVAs/specials) using title heuristics.
  - Sorts by start date to show a simple release-order view.

- Similar-genre recommendations:
  - Uses genres from the best match (when available).
  - Returns up to 30 other anime that share one or more genres, ranked by shared-genre count and score.

- Transparent data:
  - Does not guess total episodes or next-season dates.
  - Clearly labels missing data (e.g., ongoing long-runners where episode counts are not fixed in APIs).[web:106][web:107][web:138][web:173][web:182]

## Tech Stack

- Python (requests, pandas, numpy)
- Streamlit for the web UI
- Public anime APIs (Jikan, Kitsu, future extension to others)

## How to Run

git clone https://github.com/Narenp-2003/anime-recommender.git

cd anime-recommender

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt # or install manually

streamlit run app.py


Then open the URL shown in the terminal (usually http://localhost:8501).

## Phase 2 (Planned Improvements)

- Use per-anime ID endpoints (e.g., Jikan "full anime by ID") to fetch all related movies/OVAs/sequels for a franchise using official relations fields like "sequel", "side story", etc.[web:240][web:243][web:247]
- Integrate another rich source (e.g., AniList) for better tags and episode info.
- Improve recommendation quality with content-based or tag-based similarity.
