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
rag/          → end-to-end POC RAG pipeline (chunk → store → prompt → LLM)
notes/        → theory, observations, and open questions
```

---

## POC RAG pipeline (runnable)

This repo contains a small, **learning-oriented** RAG proof-of-concept in `rag/`.

Key idea: each stage is runnable as a separate module so you can see what’s happening.

### 0) Setup: virtualenv + deps

Create venv and install dependencies:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### 1) Configure `.env`

Copy the example and fill in what you need:

```bash
cp .env.example .env
```

Important variables:

- `QDRANT_LOCATION=./qdrant_data` stores Qdrant locally on disk (no server needed).
- `QDRANT_COLLECTION=rag_chunks` is the collection name used by the pipeline.
- `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` for dense embeddings.
- `RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking (optional).
- `OPENAI_API_KEY=...` needed only for the final LLM call (`rag/llm.py`).

### 2) (Optional) Preview chunking in a “SQL-like” table

Show how Markdown files are chunked, without storing anything:

```bash
./.venv/bin/python -m rag.chunk_preview notes
```

### 3) Chunk + embed + store into Qdrant

This reads `*.md` recursively from the directory you pass, chunks them, creates embeddings,
and upserts them into Qdrant (local on-disk storage by default).

```bash
./.venv/bin/python -m rag.chunk_store notes ./qdrant_data rag_chunks
```

Data will be stored under `./qdrant_data/`.

### 4) Build and inspect the prompt (dense retrieval + rerank)

This queries Qdrant, prints:

1) dense retrieval results (all rows returned by Qdrant),
2) reranked results (if reranker model is available),
3) the final prompt that would be sent to an LLM.

```bash
./.venv/bin/python -m rag.prompt "what is attention" ./qdrant_data rag_chunks
```

### 5) Full “RAG” run with an LLM (OpenAI)

This builds the prompt (same as `rag.prompt`) and then calls OpenAI, printing:

- `PROMPT` (what is sent to the model)
- `ANSWER` (model output)

```bash
./.venv/bin/python -m rag.llm "what is attention" ./qdrant_data rag_chunks
```

Notes:

- If you run without network access, SentenceTransformers cannot download models.
  Pre-download the embedding/reranker models or set `RAG_OFFLINE=1` to force local-only loading.
- This is intentionally a POC: the goal is visibility and learning, not production hardening.


### Why this repo exists

Many AI demos look impressive but hide the underlying complexity.
This repository exists to close the gap between:

“I can use AI”
and
“I understand how AI systems are built and why they behave the way they do.”
