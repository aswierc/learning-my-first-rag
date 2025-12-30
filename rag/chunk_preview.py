from __future__ import annotations

from typing import Iterable

from .chunk import chunk_markdown_dir


def _cell(text: object, width: int) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    if len(s) > width:
        s = s[: max(0, width - 1)] + "…"
    return s.ljust(width)


def print_chunks_table(
    records: list[dict[str, object]],
    *,
    max_source_width: int = 60,
    max_text_width: int = 80,
) -> None:
    headers = ("source", "chunk_index", "text")
    source_w = max((len(str(r.get("source", ""))) for r in records), default=0)
    source_w = max(len(headers[0]), min(max_source_width, source_w))

    index_w = max((len(str(r.get("chunk_index", ""))) for r in records), default=0)
    index_w = max(len(headers[1]), index_w)

    text_w = max((len(str(r.get("text", ""))) for r in records), default=0)
    text_w = max(len(headers[2]), min(max_text_width, text_w))

    def sep() -> str:
        return (
            "+"
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
        + _cell(headers[0], source_w)
        + " | "
        + _cell(headers[1], index_w)
        + " | "
        + _cell(headers[2], text_w)
        + " |"
    )
    print(sep())
    for rec in records:
        print(
            "| "
            + _cell(rec.get("source"), source_w)
            + " | "
            + _cell(rec.get("chunk_index"), index_w)
            + " | "
            + _cell(rec.get("text"), text_w)
            + " |"
        )
    print(sep())
    print(f"{len(records)} rows in set")


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        import sys

        args = sys.argv[1:]
    else:
        args = list(argv)

    root = args[0] if args else "rag"
    records = chunk_markdown_dir(root)
    if not records:
        print(f"0 rows in set (no .md files found in {root})")
        return 1

    print_chunks_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

