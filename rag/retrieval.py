from __future__ import annotations

# Backward-compatible shim (use `rag.prompt` going forward).
from .prompt import (  # noqa: F401
    dense_retrieve,
    generate_prompt,
    main,
    print_results_table,
    rerank,
)

