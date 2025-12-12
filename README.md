# learning-my-first-rag

This repository is a **personal learning and experimentation space** where I explore how modern
LLM-based systems work under the hood.

It is **not** a production-ready RAG implementation.

The goal of this repo is to understand the building blocks of retrieval-augmented generation
from first principles — without hype, shortcuts, or “AI magic”.

---

## Learning outline

The list below is a rough outline of topics I plan to study over time.
It will likely change as I learn, experiment, and discover gaps in my understanding.

1. Fundamentals: tokens, embeddings, attention, encoding  
2. Model architectures: encoder-only, decoder-only, encoder–decoder  
3. Embedding space and semantic similarity  
4. Retrieval strategies: dense, sparse, hybrid  
5. Reranking: math basics and practical use  
6. RAG: from simple pipelines to more advanced and agentic setups  
7. Vector databases (Qdrant) and retrieval pipelines  
8. Prompt engineering in real-world scenarios  
9. Evaluation and testing of LLM-based systems  
10. Deployment basics (vLLM, SGLang, llama.cpp)  
11. Fine-tuning basics (LoRA, QLoRA)

---

## High-level RAG flow (conceptual)

```text
User Query
    ↓
Tokenization
    ↓
Embedding model (text → vector)
    ↓
Retriever (dense / sparse / hybrid)
    ↓
Reranker (e.g. cross-encoder)
    ↓
Top context chunks
    ↓
LLM (GPT / Claude / LLaMA)
    ↓
Final answer
```

This diagram represents the conceptual flow, not a specific implementation.

Repository structure

The repository is organized by learning stages rather than production layers:

```
tokens/       → tokenization basics
embeddings/   → sentence embeddings and embedding space analysis
retrieval/    → dense retrieval and ranking
notes/        → theory, observations, and open questions
```


### Why this repo exists

Many AI demos look impressive but hide the underlying complexity.
This repository exists to close the gap between:

“I can use AI”
and
“I understand how AI systems are built and why they behave the way they do.”

