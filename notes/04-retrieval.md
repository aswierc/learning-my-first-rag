# dense retrieval

- Dense retrieval represents both documents and queries
  as vectors in the same embedding space.

- Similarity is computed using cosine similarity.
  When embeddings are L2-normalized, cosine similarity
  can be computed as a simple dot product.

- This approach does not rely on keyword overlap.
  Semantic similarity is captured by the geometry
  of the embedding space.

- This implementation uses brute-force search.
  For larger datasets, vector databases or ANN indexes
  (e.g. FAISS, Qdrant) are required.


