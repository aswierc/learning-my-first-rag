# Pooling and Embeddings

## Embedding pipeline
1. Tokenization  
2. Token embeddings (one embedding per token)  
3. Pooling → sentence embedding  
4. Sentence embedding is stored in the vector database  

Only the **sentence embedding** is persisted and indexed.

---

## Pooling
Pooling reduces **token embeddings** into a single **sentence embedding**.

- Tokenization produces subword tokens (not words)
- Each token has its own embedding
- Pooling aggregates them into one vector

---

## Pooling strategies

### CLS pooling
Uses only the first token embedding (`[CLS]`).
- Simple
- Often not trained for semantic similarity

### Mean pooling (recommended)
Averages all token embeddings (with attention mask).
- Default in SentenceTransformers
- More stable for semantic similarity search

---

## Vector database record (example)
```json
{
  "id": "chunk_123",
  "vector": [0.12, -0.33, 0.91],
  "payload": {
    "text": "VAT on imports depends on the customs value...",
    "doc_id": "import_guide.pdf",
    "page": 12
  }
}
```

---

## Notes
- Token embeddings are not stored in the database
- Pooling belongs to the embedding space, not retrieval
- Retrieval operates on pooled sentence embeddings
