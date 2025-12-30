from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

from .prompt import generate_prompt


def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no external dependency).
    - Does not override already-set environment variables.
    - Supports simple KEY=VALUE lines and ignores comments/blank lines.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing env var: {name}")
    return value


def call_openai_chat(prompt: str) -> str:
    load_dotenv()
    api_key = _require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    timeout_s = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the instructions in the user prompt strictly. "
                    "Citations must use the exact bracketed IDs from Context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI HTTP error {e.code}: {detail}") from e
    except urllib.error.URLError as e:  # pragma: no cover
        raise SystemExit(f"OpenAI connection error: {e}") from e

    try:
        return str(payload["choices"][0]["message"]["content"])
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Unexpected OpenAI response: {payload}") from e


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        import sys

        args = sys.argv[1:]
    else:
        args = list(argv)

    if not args:
        print(
            "Usage: python -m rag.llm \"your query\" [qdrant_location] [collection] [limit] [rerank_limit]"
        )
        return 2

    query = args[0]
    qdrant_location = args[1] if len(args) >= 2 else None
    collection = args[2] if len(args) >= 3 else None
    retrieval_limit = int(args[3]) if len(args) >= 4 else int(os.getenv("RAG_RETRIEVAL_LIMIT", "20"))
    rerank_limit = int(args[4]) if len(args) >= 5 else int(os.getenv("RAG_RERANK_LIMIT", "10"))

    out: dict[str, Any] = generate_prompt(
        query,
        qdrant_location=qdrant_location,
        collection=collection,
        retrieval_limit=retrieval_limit,
        rerank_limit=rerank_limit,
    )

    prompt = str(out["prompt"])
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt.rstrip())
    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    answer = call_openai_chat(prompt)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
