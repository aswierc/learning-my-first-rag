### embeddings

- Sentence embeddings represent text as vectors in a high-dimensional space.
  Similar meanings are located closer to each other.

- Normalization (e.g. L2 normalization) scales vectors to unit length.
  This allows cosine similarity to be computed using a simple dot product.

- Dimensionality reduction techniques such as PCA, t-SNE, or UMAP
  are **not normalization methods**.
  They are used for visualization or analysis,
  not for similarity computation in production systems.

- Semantic similarity emerges from the geometry of the embedding space,
  not from individual dimensions.

### Embedding visualization

- Techniques like PCA, t-SNE, or UMAP are used only for visualization
  and exploratory analysis.

- They reduce high-dimensional embeddings (e.g. 384D)
  to 2D or 3D by discarding most information.

- Distances and clusters in the reduced space
  are only approximations of the real embedding space.

- These methods must NOT be used for similarity search
  or retrieval in production systems.
