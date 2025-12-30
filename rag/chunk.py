from __future__ import annotations

from pathlib import Path
from typing import Iterable


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(p for p in root.rglob("*.md") if p.is_file())


def chunk_markdown_dir(
    root: str | Path,
    *,
    size: int = 500,
    overlap: int = 100,
) -> list[dict[str, object]]:
    root_path = Path(root)
    md_files = list(iter_markdown_files(root_path))
    records: list[dict[str, object]] = []

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, size=size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            records.append({"source": str(path), "chunk_index": i, "text": chunk})

    return records

