from __future__ import annotations

import os
import uuid
from typing import Iterable

from .chunk import chunk_markdown_dir
from .qdrant import default_location, ensure_collection, make_client, require_qdrant_client


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it (e.g. `pip install sentence-transformers`)."
        ) from e
    return SentenceTransformer


def store_markdown_chunks_in_qdrant(
    root: str = "rag",
    *,
    qdrant_location: str = "./qdrant_data",
    collection: str = "rag_chunks",
    model_name: str | None = None,
    size: int = 500,
    overlap: int = 100,
    batch_size: int = 128,
    local_files_only: bool | None = None,
) -> int:
    records = chunk_markdown_dir(root, size=size, overlap=overlap)
    if not records:
        print(f"0 rows in set (no .md files found in {root})")
        return 0

    SentenceTransformer = _require_sentence_transformers()
    if model_name is None:
        model_name = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if local_files_only is None:
        local_files_only = bool(
            os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE") or os.getenv("RAG_OFFLINE")
        )

    try:
        model = SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Failed to load embedding model. If you're offline, pre-download the model "
            f"({model_name}) or unset offline env vars (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/RAG_OFFLINE). "
            f"Original error: {e}"
        ) from e

    api_key = os.getenv("QDRANT_API_KEY")
    client = make_client(location=qdrant_location, api_key=api_key)

    vector_size = int(model.get_sentence_embedding_dimension())
    ensure_collection(client=client, name=collection, vector_size=vector_size, distance="cosine")

    texts = [str(r["text"]) for r in records]
    vectors = model.encode(texts, normalize_embeddings=True)
    vectors_list = vectors.tolist()

    _, _, PointStruct, _ = require_qdrant_client()
    points: list[object] = []
    for rec, vec in zip(records, vectors_list, strict=True):
        source = str(rec.get("source", ""))
        chunk_index = int(rec.get("chunk_index", 0))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))
        payload = {"source": source, "chunk_index": chunk_index, "text": str(rec.get("text", ""))}
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    print(f"Upserted {total} points into {collection} @ {qdrant_location}")
    return total


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        import sys

        args = sys.argv[1:]
    else:
        args = list(argv)

    root = args[0] if len(args) >= 1 else "rag"
    qdrant_location = args[1] if len(args) >= 2 else default_location()
    collection = args[2] if len(args) >= 3 else os.getenv("QDRANT_COLLECTION", "rag_chunks")

    stored = store_markdown_chunks_in_qdrant(
        root, qdrant_location=qdrant_location, collection=collection
    )
    return 0 if stored > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
