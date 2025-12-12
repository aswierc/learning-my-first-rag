import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "VAT tax on imports from China.",
    "Customs duties are applied to goods from outside the EU.",
    "Pasta is cooked for about 8 minutes.",
    "Instructions for preparing spaghetti.",
    "Customs declaration for imported goods."
]

# Encode all documents once
doc_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

def retrieve(query: str, top_k: int = 3):
    # Encode query using the same model and normalization
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Because embeddings are normalized,
    # dot product == cosine similarity
    scores = np.dot(doc_embeddings, query_embedding)

    top_indices = np.argsort(-scores)[:top_k]

    return [
        (documents[i], float(scores[i]))
        for i in top_indices
    ]

query = "How to handle imports from China"
results = retrieve(query)

for doc, score in results:
    print(f"{score:.3f} | {doc}")
