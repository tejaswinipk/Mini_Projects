import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import streamlit as st
import random
import requests

# Load the movie dataset
movies = pd.read_csv('top10K-TMDB-movies.csv')
movies = movies[['id', 'title', 'overview', 'genre', 'popularity']]

# Load the book dataset
books = pd.read_csv('book.csv')
books = books[['title', 'name', 'genre', 'rating']]

# Define lists of keywords for different topics
# You can define keyword lists for various domains here
crime_keywords = ["Criminal", "Police", "Investigation", "Suspect", "Law", "Arrest", "Offense", "Homicide", "Robbery", "Theft", "Court", "Conviction", "Evidence", "Prison", "Witness", "Detective"]
comedy_keywords = ["funny", "humor", "hilarious", "joke", "laughter", "comedian", "witty", "humorous", "satire", "jest", "comical", "lighthearted", "amusing"]
history_keywords = ["historical", "past", "civilization", "archaeology", "timeline", "era", "chronicle", "museum", "artifact", "ancestry", "tradition", "culture", "legacy", "archival", "prehistoric", "historian", "relatives", "kin"]
family_keywords = ["parents","mother", "father","family","fam", "children", "siblings", "mother", "father",  "relationship", "bond", "together", "support", "unity", "generations", "upbringing", "household", "legacy", "generations", "relatives", "kin"]
war_keywords = ["battle", "soldier", "military", "troops", "violence", "weaponry", "strategy", "defense", "battlefield", "mission", "ceasefire", "generations", "resistance", "mission", "invasion"]
animation_keywords = ["cartoon", "animated", "Pixar", "Disney", "anime", "3D animation", "2D animation", "motion graphics", "stop motion", "frame", "storyboard", "voice acting", "animators", "animation studio", "CGI", "visual effects"]
fantasy_keywords = ["magical", "enchanted", "mythical", "surreal", "whimsical", "mystical", "otherworldly", "fantastical", "folklore", "dragon", "unicorns", "epic", "realms", "gripping storyline", "visual effects", "epic"]
thriller_keywords = ["suspense", "suspenseful", "tense", "dramatic", "chilling", "terrifying", "intense", "gripping", "heart-pounding", "edge-of-your-seat", "nail-biting", "thriller novel", "thriller movie", "psychological", "jump scare"]
horror_keywords = ["scary", "fright", "terrifying", "chilling", "blood-curdling", "haunted", "horror movie", "nightmare", "creepy", "gore", "macabre", "horror film", "robot", "zombies", "ghosts"]
science_fiction_keywords = ["futuristic", "space", "technology", "discovery", "extraterrestrial", "expedition", "cyberpunk", "advanced technology", "speculative fiction", "space exploration", "parallel universe", "speculative fiction"]
adventure_keywords = ["journey", "quest", "exploration", "time travel", "outdoor adventure", "wild", "wanderlust", "odyssey", "expedition"]
music_keywords = ["song","music","soundtrack","sound", "melody", "tune", "concert", "band", "rhythm", "performance", "album", "playlist", "vocalist", "epic sound"]
mystery_keywords = ["enigma", "puzzling", "clue", "riddle", "perplexing", "unsolved", "hidden", "detective", "suspense", "puzzle", "curious", "mysterious", "whodunit", "cryptic", "puzzling", "clue", "riddle"]
romance_keywords = ["love","you","me" "affection", "relationship", "kiss", "heart", "babe", "darling", "desire"]
# Define similar keyword lists for books (e.g., romance_keywords, thriller_keywords)

# Define the pre-trained model and labels for sentiment analysis
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
labels = ['Negative', 'Neutral', 'Positive']

# Load the model and tokenizer for sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Function to perform sentiment analysis on a single text
def analyze_sentiment(text):
    tweet_words = []

    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment_probabilities = {labels[i]: scores[i] for i in range(len(labels))}

    return sentiment_probabilities

# Function to extract movie genres based on keywords
def extract_movie_genres(text):
    cleaned_text = ' '.join([word for word in text.split() if not word.startswith('@')])

    keyword_lists = {
        "Crime": crime_keywords,
        "Comedy": comedy_keywords,
        "History": history_keywords,
        "Family": family_keywords,
        "War": war_keywords,
        "Animation": animation_keywords,
        "Fantasy": fantasy_keywords,
        "Thriller": thriller_keywords,
        "Horror": horror_keywords,
        "Science Fiction": science_fiction_keywords,
        "Adventure": adventure_keywords,
        "Music": music_keywords,
        "Mystery": mystery_keywords,
        "Romance": romance_keywords
        # Add more movie genres and their respective keyword lists
    }

    matched_genres = []
    for domain, keywords in keyword_lists.items():
        if any(keyword in cleaned_text for keyword in keywords):
            matched_genres.append(domain)

    return matched_genres

# recommend_random_books function
def recommend_random_books(genres):
    num_books_to_recommend = 5
    recommended_books = []

    # Iterate over the list of genres to collect books
    for genre in genres:
        genre_books = books[books['genre'].str.contains(genre, case=False, na=False)]
        if not genre_books.empty:
            recommended_books.extend(genre_books['title'].tolist())

    # If the number of collected books is less than required, fill with random books from the entire dataset
    if len(recommended_books) < num_books_to_recommend:
        remaining_books_needed = num_books_to_recommend - len(recommended_books)
        all_books = list(books['title'])
        recommended_books.extend(random.sample(all_books, remaining_books_needed))

    # Ensure we only return the required number of books
    if len(recommended_books) > num_books_to_recommend:
        recommended_books = random.sample(recommended_books, num_books_to_recommend)

    return recommended_books

def recommend_popular_books():
    # Sort movies by popularity (assuming there's a 'popularity' column)
    popular_books = books.sort_values(by='rating', ascending=False)

    # Get the top 5 most popular books
    top_popular_books = popular_books.head(5)

    return top_popular_books['title'].tolist()

def recommend_popular_movies():
    # Sort movies by popularity (assuming there's a 'popularity' column)
    popular_movies = movies.sort_values(by='popularity', ascending=False)

    # Get the top 5 most popular movies
    top_popular_movies = popular_movies.head(5)

    return top_popular_movies['title'].tolist()

# Function to extract book genres based on keywords
def extract_book_genres(text):
    cleaned_text = ' '.join([word for word in text.split() if not word.startswith('@')])

    keyword_lists = {
        "romance": romance_keywords,
        "thriller": thriller_keywords,
        "history": history_keywords,
        "horror": horror_keywords,
        "psychology": mystery_keywords,
        "science_fiction": science_fiction_keywords,
        "science": science_fiction_keywords,
        "sports": adventure_keywords,
        "thriller": thriller_keywords,
        "travel": adventure_keywords,
        "fantasy": fantasy_keywords,
        # Add more book genres and their respective keyword lists
    }

    matched_genres = []
    for domain, keywords in keyword_lists.items():
        if any(keyword in cleaned_text for keyword in keywords):
            matched_genres.append(domain)

    return matched_genres

# Streamlit app
st.title("Movie and Book Recommendation System")

user_input = st.text_input("Enter a text (or type 'exit' to quit):")

def fetch_poster(movie_id):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
     data=requests.get(url)
     data=data.json()
     poster_path = data['poster_path']
     full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
     return full_path

if user_input.lower() == 'exit':
    st.write("Exiting the program.")
else:
    sentiment_probs = analyze_sentiment(user_input)
    max_sentiment = max(sentiment_probs, key=sentiment_probs.get)

    st.subheader("Sentiment Analysis:")

    st.write(max_sentiment)

    # Extract movie genres based on keywords
    if st.button("Show Movie Recommend"):
        extracted_movie_genres = extract_movie_genres(user_input)

        if not extracted_movie_genres:
            popular_movies = recommend_popular_movies()
            if popular_movies:
                st.subheader("Recommended Popular Movies:")
                col1, col2, col3, col4, col5 = st.columns(5)
                for i, movie in enumerate(popular_movies[:5]):  # Display up to 5 popular movies
                    with col1 if i % 5 == 0 else col2 if i % 5 == 1 else col3 if i % 5 == 2 else col4 if i % 5 == 3 else col5:
                        st.text(movie)
                        movie_info = movies[movies['title'] == movie].iloc[0]
                        st.image(fetch_poster(movie_info['id']))
            else:
                st.write("No movies found in the dataset for recommendation.")
        else:
            st.write("Detected Movie Genres:", ", ".join(extracted_movie_genres))
            # Find movies in the dataset that match the extracted genres
            recommended_movies = movies[movies['genre'].isin(extracted_movie_genres)]['title'].tolist()
            recommended_moviess = movies[movies['genre'].isin(extracted_movie_genres)]
            movie_ids = recommended_moviess['id'].tolist()

            if recommended_movies:
                st.subheader("Recommended Movies:")
                col1, col2, col3, col4, col5 = st.columns(5)
                for i, movie in enumerate(recommended_movies[:5]):  # Display up to 5 recommended movies
                    with col1 if i % 5 == 0 else col2 if i % 5 == 1 else col3 if i % 5 == 2 else col4 if i % 5 == 3 else col5:
                        st.text(movie)
                        if i < len(movie_ids):
                            st.image(fetch_poster(movie_ids[i]))
            else:
                st.write("No movies found in the dataset matching the extracted movie genres.")


    extracted_movie_genres = extract_movie_genres(user_input)
    # Recommend popular books
    if st.button("Show Book Recommend"):
        popular_books = recommend_random_books(extracted_movie_genres)
        if popular_books:
            st.subheader("Recommended Popular Books:")
            for book in popular_books:
                st.write(book)
        else:
            st.write("No books found in the dataset for recommendation.")