from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Example corpus of sentences
corpus = [
    ["I", "love", "to", "code"],
    ["Python", "is", "my", "favorite", "programming", "language"],
    ["Machine", "learning", "is", "an", "exciting", "field"],
    ["I", "enjoy", "working", "on", "data", "science", "projects"],
]

# Train Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=1)

# Get word embeddings
word_embeddings = model.wv

# Calculate cosine similarity
similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["enjoy"]])
print(similarity)
similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["favorite"]])
print(similarity)

similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["data"]])
print(similarity)

