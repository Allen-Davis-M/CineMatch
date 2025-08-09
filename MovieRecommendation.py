import pandas as pd
import numpy as np
import ast
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import random
import matplotlib.pyplot as plt


st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\allen\OneDrive\Desktop\Movie Recommendation\movies.csv")
    df = df[['budget', 'genres', 'keywords', 'original_language', 'title',
             'popularity', 'release_date', 'revenue', 'runtime',
             'vote_average', 'vote_count', 'cast', 'director']]
    df = df[df['genres'].notna()].fillna('')
    return df

df = load_data()


def extract_names(text):
    try:
        return ' '.join([d['name'].replace(" ", "") for d in ast.literal_eval(text)])
    except:
        return ''

for col in ['genres', 'keywords', 'cast']:
    df[col] = df[col].apply(extract_names)


for col in ['budget', 'popularity', 'release_date', 'revenue', 'runtime',
            'vote_average', 'vote_count']:
    df[col] = df[col].astype(str)


def create_feature_string(row):
    return f"{row['genres']} {row['budget']} {row['keywords']} {row['original_language']} {row['title']} {row['popularity']} {row['release_date']} {row['revenue']} {row['runtime']} {row['vote_average']} {row['vote_count']} {row['cast']} {row['director']}"

df['combined'] = df.apply(create_feature_string, axis=1)


@st.cache_resource
def compute_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(tfidf_matrix)

similarity_score = compute_similarity()


TMDB_API_KEY = "API_KEY"  # use your API KEY

def get_poster(title):
    if not TMDB_API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url).json()
        poster_path = res['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None


def recommend(movie_title, genre_filter='', lang_filter='', year_range=(1950, 2025), n=30):
    all_titles = df['title'].tolist()
    matches = difflib.get_close_matches(movie_title, all_titles)
    if not matches:
        return None, []

    close_match = matches[0]
    index = df[df['title'] == close_match].index[0]
    recs = list(enumerate(similarity_score[index]))
    recs = sorted(recs, key=lambda x: x[1], reverse=True)[1:200]

    recommended = []
    for idx, score in recs:
        movie = df.iloc[idx]
        try:
            year = int(movie['release_date'][:4])
        except:
            year = 0

        if genre_filter and genre_filter.lower() not in movie['genres'].lower():
            continue
        if lang_filter and lang_filter.lower() != movie['original_language'].lower():
            continue
        if not (year_range[0] <= year <= year_range[1]):
            continue

        recommended.append(movie)
        if len(recommended) == n:
            break
    return close_match, recommended


st.sidebar.header("📂 Filters")
selected_genre = st.sidebar.text_input("Genre Filter (e.g. Action, Comedy)")
selected_language = st.sidebar.text_input("Original Language (e.g. en, hi, fr)")
release_year_range = st.sidebar.slider("Release Year Range", 1950, 2025, (2000, 2025))
random_btn = st.sidebar.button("🎲 Surprise Me!")


st.title("🎬 Movie Recommender System")
st.write("Find movies similar to your favourites with filters and rich details.")

if random_btn:
    user_movie = random.choice(df['title'].tolist())
    st.info(f"Random pick: **{user_movie}**")
else:
    user_movie = st.text_input("Enter your favourite movie:")

if user_movie:
    close_match, recommendations = recommend(user_movie, selected_genre, selected_language, release_year_range)

    if close_match:
        st.subheader(f"Closest match found: **{close_match}**")
        if recommendations:
            st.subheader("🎥 Top Recommendations:")
            cols = st.columns(3)
            for i, movie in enumerate(recommendations):
                poster_url = get_poster(movie['title'])
                with cols[i % 3]:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{movie['title']}** ({movie['release_date'][:4]})")
                    st.caption(f"⭐ {movie['vote_average']} | 🎭 {movie['genres']} | 🌐 {movie['original_language']}")
        else:
            st.warning("No recommendations match the selected filters.")
    else:
        st.error("No close match found. Please check the spelling.")


st.subheader("📊 Recommendation Insights")

if user_movie and close_match and recommendations:
    recommendations_df = pd.DataFrame(recommendations)

    if 'genres' in recommendations_df.columns and not recommendations_df['genres'].empty:
        genres_list = ' '.join(recommendations_df['genres'].dropna())
        if genres_list.strip(): 
            genres_split = genres_list.split()
            genre_counts = pd.Series(genres_split).value_counts().head(10)

            if not genre_counts.empty:
                fig, ax = plt.subplots()
                genre_counts.plot(kind='bar', ax=ax)
                plt.title("Top Genres in Recommendations")
                st.pyplot(fig)
            else:
                st.info("No genre data available to plot.")
        else:
            st.info("No genre data available to plot.")
    else:
        st.info("No genre data available to plot.")

