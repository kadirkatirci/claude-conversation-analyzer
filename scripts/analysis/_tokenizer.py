"""
Tokenizer shim — uses tiktoken when available, falls back to char/4 approximation.

The char/4 heuristic matches the cl100k_base average of ~4 characters per token.
All token counts in this project are documented as approximate upper bounds,
so the fallback is acceptable for Pyodide / browser environments where
tiktoken (a C extension) is unavailable.
"""

from __future__ import annotations


def _try_tiktoken():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, Exception):
        return None


_enc = _try_tiktoken()


def encode(text: str) -> list[int]:
    """Return token IDs (or a dummy list of the right length)."""
    if _enc is not None:
        return _enc.encode(text, disallowed_special=())
    # Approximation: ~4 chars per token
    n = max(1, len(text) // 4) if text else 0
    return [0] * n


def encode_batch(texts: list[str], **kwargs) -> list[list[int]]:
    """Batch-encode a list of texts."""
    if _enc is not None:
        return _enc.encode_batch(texts, disallowed_special=())
    return [encode(t) for t in texts]
