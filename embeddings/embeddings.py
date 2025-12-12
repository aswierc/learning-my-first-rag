from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How to calculate VAT on imports?",
    "Customs duties and import taxes from China.",
    "Recipe for cooking pasta al dente.",
    "butter knife"
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True ## <-- L2 normalized
)

print("Embedding matrix shape:", embeddings.shape)

# Because embeddings are L2-normalized,
# dot product == cosine similarity
cosine_sim_matrix = np.dot(embeddings, embeddings.T)

print("Cosine similarity matrix:")
print(cosine_sim_matrix)
