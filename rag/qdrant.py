from __future__ import annotations

import os
from typing import Any


def require_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import (
            Distance,
            PointStruct,
            VectorParams,
        )
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: qdrant-client. Install it (e.g. `pip install qdrant-client`)."
        ) from e

    return QdrantClient, Distance, PointStruct, VectorParams


def default_location() -> str:
    return os.getenv("QDRANT_LOCATION") or os.getenv("QDRANT_PATH") or "./qdrant_data"


def make_client(*, location: str, api_key: str | None = None):
    QdrantClient, _, _, _ = require_qdrant_client()
    if location.startswith("http://") or location.startswith("https://"):
        return QdrantClient(url=location, api_key=api_key)

    if location not in (":memory:",):
        os.makedirs(location, exist_ok=True)
    return QdrantClient(path=location)


def collection_exists(client: Any, name: str) -> bool:
    exists_fn = getattr(client, "collection_exists", None)
    if callable(exists_fn):
        return bool(exists_fn(name))
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def ensure_collection(
    *,
    client: Any,
    name: str,
    vector_size: int,
    distance: str = "cosine",
) -> None:
    _, Distance, _, VectorParams = require_qdrant_client()

    if collection_exists(client, name):
        return

    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.COSINE
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=dist),
    )


def default_api_key() -> str | None:
    return os.getenv("QDRANT_API_KEY")
