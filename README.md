# Anime Recommender 🎌

Streamlit web app to find good entry points, timelines, and similar anime using multiple public APIs plus an optional offline MyAnimeList TF-IDF model.

## Features

- Fuzzy title search across MyAnimeList (Jikan), AniList, and Kitsu APIs. 
- Series timeline that merges main entries with related works from Jikan relations (movies, OVAs, specials). 
- Cover images and basic stats (score, episodes, status, year) for the best match and series entries. 
- Genre-based recommendations using Jikan’s genre endpoints and fuzzy title matching via RapidFuzz. 
- Optional offline content-based recommendations using a MAL CSV dataset and TF-IDF (cosine similarity). 

## Tech stack

- Python 3.10+
- Streamlit for the UI
- Requests + JSON for direct API calls (Jikan, AniList GraphQL, Kitsu REST). 
- RapidFuzz for fuzzy string matching between user input and titles.
- Pandas for data handling and basic ranking.
- scikit-learn for the offline TF-IDF model and cosine similarity. 

## Project structure

- `anime_recommender.py`  
  Core API integration and logic: Jikan/AniList/Kitsu search, series relations, fuzzy title scoring, and genre-based recommendations. 
- `offline_mal_model.py`  
  Loads a MyAnimeList anime CSV, builds a TF-IDF representation, and exposes the matrix plus metadata for offline recs. 
- `app.py`  
  Streamlit front-end: search bar, best-match card with image, series timeline table, and “More like this” (API + offline) views. 
- `requirements.txt`  
  Python dependencies for deployment.

## Setup

1. Clone the repository and create a virtual environment.

2. Install dependencies:

   
   pip install -r requirements.txt

Prepare the offline MAL dataset (optional but recommended):

Download a cleaned anime list CSV (e.g. from Kaggle or a MAL dump) with at least: id, title, synopsis/description, score, episodes.

Update the path and column names in offline_mal_model.py to match your CSV.

Run the app:

streamlit run app.py

Usage
Type an anime title (e.g. “Cowboy Bebop”) in the search box. git push -u origin main


Check the best match card and the series timeline to understand viewing order (TV, movie, specials). 

Use the “More like this” section:

API-based (genres) for cross-genre recs with filters (type, score, episodes, airing, preferred genres).

Offline MAL model for content-based recs purely from the local MAL dataset.

Notes and API limits
Jikan is an unofficial MyAnimeList REST API; respect their rate limits and terms of service. 

AniList GraphQL and Kitsu also have usage guidelines; heavy usage may require auth or backoff. 