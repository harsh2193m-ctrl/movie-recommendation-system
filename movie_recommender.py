import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "movies.csv"

df = pd.read_csv(DATA_FILE)
df["genres"] = df["genres"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["genres"])

similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

def recommend(title, n=5):
    key = title.lower().strip()
    if key not in indices:
        print("Movie not found. Try one of these:")
        print(", ".join(df["title"].head(10).tolist()))
        return

    idx = indices[key]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [x for x in scores if x[0] != idx][:n]

    print(f"\nRecommendations for: {df.iloc[idx]['title']}")
    for rank, (movie_idx, score) in enumerate(scores, 1):
        print(f"{rank}. {df.iloc[movie_idx]['title']}  (similarity: {score:.2f})")

if __name__ == "__main__":
    print("Movie Recommendation System")
    print("Available examples:", ", ".join(df["title"].head(10).tolist()))
    movie = input("\nEnter a movie title: ")
    recommend(movie)
