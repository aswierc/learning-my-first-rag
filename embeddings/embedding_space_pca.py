from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentences = [
    "How to calculate VAT on imports?",
    "Customs duties and import taxes from China.",
    "Pasta cooking recipe.",
    "Cook pasta for 8 minutes.",
    "Imports from China require customs clearance.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

# Reduce high-dimensional embeddings to 2D for visualization
pca = PCA(n_components=2)
points_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(points_2d[:, 0], points_2d[:, 1])

for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (points_2d[i, 0], points_2d[i, 1]))

plt.title("PCA projection of sentence embeddings")
plt.show()
