from __future__ import annotations

import os
from typing import Iterable

from .qdrant import collection_exists, default_location, make_client


def _require_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it (e.g. `pip install sentence-transformers`)."
        ) from e
    return SentenceTransformer, CrossEncoder


def _default_local_files_only() -> bool:
    return bool(os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE") or os.getenv("RAG_OFFLINE"))


def _cell(text: object, width: int) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    if len(s) > width:
        s = s[: max(0, width - 1)] + "…"
    return s.ljust(width)


def print_results_table(
    rows: list[dict[str, object]],
    *,
    score_key: str,
    score_label: str,
    max_source_width: int = 60,
    max_text_width: int = 80,
) -> None:
    headers = (score_label, "source", "chunk_index", "text")

    score_w = max(len(headers[0]), 8)
    source_w = max((len(str(r.get("source", ""))) for r in rows), default=0)
    source_w = max(len(headers[1]), min(max_source_width, source_w))
    index_w = max((len(str(r.get("chunk_index", ""))) for r in rows), default=0)
    index_w = max(len(headers[2]), index_w)
    text_w = max((len(str(r.get("text", ""))) for r in rows), default=0)
    text_w = max(len(headers[3]), min(max_text_width, text_w))

    def sep() -> str:
        return (
            "+"
            + "-" * (score_w + 2)
            + "+"
            + "-" * (source_w + 2)
            + "+"
            + "-" * (index_w + 2)
            + "+"
            + "-" * (text_w + 2)
            + "+"
        )

    print(sep())
    print(
        "| "
        + _cell(headers[0], score_w)
        + " | "
        + _cell(headers[1], source_w)
        + " | "
        + _cell(headers[2], index_w)
        + " | "
        + _cell(headers[3], text_w)
        + " |"
    )
    print(sep())
    for r in rows:
        score = r.get(score_key)
        score_s = ""
        if isinstance(score, (int, float)):
            try:
                import math

                f = float(score)
                score_s = f"{f:.4f}" if math.isfinite(f) else ""
            except Exception:
                score_s = ""
        print(
            "| "
            + _cell(score_s, score_w)
            + " | "
            + _cell(r.get("source"), source_w)
            + " | "
            + _cell(r.get("chunk_index"), index_w)
            + " | "
            + _cell(r.get("text"), text_w)
            + " |"
        )
    print(sep())
    print(f"{len(rows)} rows in set")


def dense_retrieve(
    query: str,
    *,
    qdrant_location: str | None = None,
    collection: str | None = None,
    limit: int = 20,
    embedding_model_name: str | None = None,
    local_files_only: bool | None = None,
) -> list[dict[str, object]]:
    if qdrant_location is None:
        qdrant_location = default_location()
    if collection is None:
        collection = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    api_key = os.getenv("QDRANT_API_KEY")
    client = make_client(location=qdrant_location, api_key=api_key)
    if not collection_exists(client, collection):
        raise SystemExit(f"Collection not found: {collection} (location: {qdrant_location})")

    SentenceTransformer, _ = _require_sentence_transformers()
    if embedding_model_name is None:
        embedding_model_name = os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    if local_files_only is None:
        local_files_only = _default_local_files_only()

    try:
        model = SentenceTransformer(embedding_model_name, local_files_only=local_files_only)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Failed to load embedding model. If you're offline, pre-download the model "
            f"({embedding_model_name}) or set RAG_OFFLINE=1 to force local-only loading. "
            f"Original error: {e}"
        ) from e

    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=limit,
            with_payload=True,
        )
        hits = resp.points
    elif hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=qvec,
            limit=limit,
            with_payload=True,
        )
    elif hasattr(client, "search_points"):
        hits = client.search_points(
            collection_name=collection,
            query_vector=qvec,
            limit=limit,
            with_payload=True,
        )
    else:
        raise SystemExit("Unsupported qdrant-client version: no query/search method found")

    rows: list[dict[str, object]] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        rows.append(
            {
                "dense_score": getattr(hit, "score", None),
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "id": getattr(hit, "id", None),
            }
        )
    return rows


def rerank(
    query: str,
    rows: list[dict[str, object]],
    *,
    reranker_model_name: str | None = None,
    local_files_only: bool | None = None,
) -> list[dict[str, object]]:
    if not rows:
        return []

    _, CrossEncoder = _require_sentence_transformers()
    if reranker_model_name is None:
        reranker_model_name = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if local_files_only is None:
        local_files_only = _default_local_files_only()

    try:
        reranker = CrossEncoder(reranker_model_name, local_files_only=local_files_only)
    except Exception as e:  # pragma: no cover
        print(
            "Reranker not available (skipping rerank). "
            "Pre-download the reranker model or allow network access. "
            f"Model: {reranker_model_name}. Error: {e}"
        )
        return [dict(r, rerank_score=None) for r in rows]

    pairs = [(query, str(r.get("text", ""))) for r in rows]
    scores = reranker.predict(pairs)

    reranked: list[dict[str, object]] = []
    for r, s in zip(rows, scores, strict=True):
        rr = dict(r)
        rr["rerank_score"] = float(s)
        reranked.append(rr)

    def _rerank_key(x: dict[str, object]) -> float:
        s = x.get("rerank_score")
        return float(s) if isinstance(s, (int, float)) else float("-inf")

    reranked.sort(key=_rerank_key, reverse=True)
    return reranked


def build_prompt(
    query: str,
    rows: list[dict[str, object]],
    *,
    max_context_chars: int = 6000,
) -> str:
    parts: list[str] = []
    total = 0

    for r in rows:
        source = str(r.get("source") or "")
        chunk_index = r.get("chunk_index")
        chunk_ref = f"{source}#{chunk_index}" if source else str(chunk_index)
        text = str(r.get("text") or "").strip()
        if not text:
            continue

        entry = f"[{chunk_ref}]\n{text}\n"
        if parts and total + len(entry) > max_context_chars:
            break

        parts.append(entry)
        total += len(entry)

    context = "\n".join(parts).strip()
    return (
        "You are a helpful assistant.\n"
        "Answer the user question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know.\n"
        "Citations:\n"
        "- Cite sources inline using the exact bracketed IDs from the Context section (e.g., [notes/01.1-attention.md#2]).\n"
        "- Do NOT invent placeholders like [source#chunk_index].\n"
        "- Every factual claim should be backed by at least one citation.\n"
        "- After the answer, output a 'Sources:' section listing unique citations you used.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n"
    )


def select_prompt_rows(
    dense_rows: list[dict[str, object]],
    reranked_rows: list[dict[str, object]] | None,
    *,
    rerank_limit: int,
) -> list[dict[str, object]]:
    prompt_min_rows = int(os.getenv("RAG_PROMPT_MIN_ROWS", "3"))
    rerank_min_score = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.0"))

    def _finite_score(v: object) -> float | None:
        if not isinstance(v, (int, float)):
            return None
        try:
            import math

            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    if reranked_rows:
        good = []
        for r in reranked_rows:
            s = _finite_score(r.get("rerank_score"))
            if s is not None and s >= rerank_min_score:
                good.append(r)

        if len(good) >= prompt_min_rows:
            return good[:rerank_limit]

        top_reranked = reranked_rows[: max(prompt_min_rows, min(rerank_limit, len(reranked_rows)))]
        return top_reranked

    return dense_rows[: max(prompt_min_rows, min(rerank_limit, len(dense_rows)))]


def generate_prompt(
    query: str,
    *,
    qdrant_location: str | None = None,
    collection: str | None = None,
    retrieval_limit: int = 20,
    rerank_limit: int = 10,
    max_context_chars: int = 6000,
) -> dict[str, object]:
    dense_rows = dense_retrieve(
        query,
        qdrant_location=qdrant_location,
        collection=collection,
        limit=retrieval_limit,
    )
    reranked = rerank(query, dense_rows)
    prompt_rows = select_prompt_rows(dense_rows, reranked, rerank_limit=rerank_limit)
    prompt = build_prompt(query, prompt_rows, max_context_chars=max_context_chars)
    return {
        "dense_rows": dense_rows,
        "reranked_rows": reranked,
        "prompt_rows": prompt_rows,
        "prompt": prompt,
    }


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        import sys

        args = sys.argv[1:]
    else:
        args = list(argv)

    if not args:
        print(
            "Usage: python -m rag.prompt \"your query\" [qdrant_location] [collection] [limit] [rerank_limit]"
        )
        return 2

    query = args[0]
    qdrant_location = args[1] if len(args) >= 2 else None
    collection = args[2] if len(args) >= 3 else None
    retrieval_limit = int(args[3]) if len(args) >= 4 else int(os.getenv("RAG_RETRIEVAL_LIMIT", "20"))
    rerank_limit = int(args[4]) if len(args) >= 5 else int(os.getenv("RAG_RERANK_LIMIT", "10"))

    out = generate_prompt(
        query,
        qdrant_location=qdrant_location,
        collection=collection,
        retrieval_limit=retrieval_limit,
        rerank_limit=rerank_limit,
    )

    dense_rows = out["dense_rows"]
    reranked_rows = out["reranked_rows"]
    prompt = out["prompt"]

    print_results_table(dense_rows, score_key="dense_score", score_label="dense")
    if reranked_rows:
        print()
        print_results_table(
            reranked_rows[:rerank_limit], score_key="rerank_score", score_label="rerank"
        )
    print()
    print("PROMPT:")
    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
