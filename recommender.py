import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("music.csv")

# Combine important features
df['features'] = df['artist'] + " " + df['genre']

# Convert text features to TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommend function
def recommend(track_name):
    if track_name not in df['track_name'].values:
        print("Track not found.")
        return

    idx = df[df['track_name'] == track_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\nTop recommendations for '{track_name}':")
    for i, score in similarity_scores[1:6]:  # Top 5 excluding itself
        print(f"- {df.iloc[i]['track_name']} by {df.iloc[i]['artist']} ({df.iloc[i]['genre']})")

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a track name: ")
    recommend(user_input)
