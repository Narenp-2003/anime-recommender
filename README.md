# Watch Recommender 🎬

Streamlit app to discover **anime, movies, and TV shows** similar to what you already like.

- Anime: multiple providers + series timeline + genre-based and offline MAL TF‑IDF recommendations.
- Movies/TV: TMDb search + recommendations, enriched with **IMDb ratings (OMDb)** and **where to watch** via a streaming-availability API.
- Offline movie/TV model (optional): Netflix / Prime / Disney+ catalogues for future extensions.

## Features

### Anime
- Search across multiple anime providers.
- Best-match card with image, status, score, and synopsis.
- Series timeline table with rank and thumbnails.
- Search results with provider filter and rank.
- “More like this”:
  - API-based (genres) with advanced filters and provider selection.
  - Offline MAL TF‑IDF model for content-based recommendations.

### Movies & TV
- TMDb-based search for movies and TV shows.
- Best-match card with:
  - TMDb score and votes.
  - IMDb rating and votes (OMDb).
  - Streaming platforms (Streaming Availability API).
- Search results table with rank.
- “More like this”:
  - TMDb API mode with filters (score, votes, year) and a hybrid rank score.
  - TMDb + IMDb mode showing top 10 TMDb recs with IMDb rating.

## Project structure

Key files:

- `app.py` – Streamlit UI and main logic.
- `anime_recommender.py` – Anime API integrations and helpers.
- `offline_mal_model.py` – Offline anime TF‑IDF model (MAL dataset).
- `movie_recommender.py` – TMDb search and recommendation helpers for movies/TV.
- `movie_offline_model.py` – (optional) offline movie/TV TF‑IDF model.
- `online_movie_sources.py` – OMDb + streaming-availability API helpers.
- `requirements.txt` – Python dependencies.
- `content_based_recommender.py` – TF-IDF + cosine similarity engine for anime recommendations (replaces old genre-overlap counting).

## Setup

### 1. Create and activate virtualenv (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate

2. Install dependencies
bash
pip install -r requirements.txt

3. Environment variables
Set these locally (PowerShell):

setx TMDB_API_KEY "your_tmdb_key_here"
setx OMDB_API_KEY "your_omdb_key_here"
setx STREAMING_API_KEY "your_streaming_api_key_here"

For Streamlit Community Cloud, add the same keys in Secrets:

TMDB_API_KEY = "your_tmdb_key_here"
OMDB_API_KEY = "your_omdb_key_here"
STREAMING_API_KEY = "your_streaming_api_key_here"

Open the URL shown in the terminal (usually http://localhost:8501).

Deployment (Streamlit Community Cloud)
Push this project to GitHub.

Go to https://streamlit.io/cloud and create a new app.

Select your repo, branch, and app.py as the entry file.

Add the API keys in “Secrets”.

Deploy – the app will auto-update on every git push

