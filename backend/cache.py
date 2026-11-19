"""Bounded in-process result cache for completed official runs."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
import os
import threading
from typing import Any

from .config import (
    RUN_CACHE_MAX_BYTES, RUN_CACHE_MAX_ENTRIES, RUN_CACHE_MAX_ENTRY_BYTES,
    RUN_CACHE_VERSION,
)

_run_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_run_cache_sizes: dict[str, int] = {}
_run_cache_lock = threading.Lock()


def _run_cache_key(dataset: dict[str, Any]) -> str:
    """Content digest of the fields that affect answers; never the reference.

    Each field is hashed separately so a large dataset is not duplicated as a
    second canonical JSON string in memory.
    """
    digest = sha256()
    digest.update(f"case4-run-cache\0{RUN_CACHE_VERSION}\0".encode())
    digest.update(os.environ.get("LLM_MODEL", "").encode())
    for name in ("requests", "experts", "chat_history", "articles", "users"):
        encoded = json.dumps(
            dataset.get(name),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(name.encode())
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_lookup(cache_key: str) -> dict[str, Any] | None:
    with _run_cache_lock:
        result = _run_cache.get(cache_key)
        if result is not None:
            _run_cache.move_to_end(cache_key)
        return result


def _cache_total_locked() -> int:
    return sum(
        size for key, size in _run_cache_sizes.items() if key in _run_cache
    )


def _cache_store(cache_key: str, result: dict[str, Any]) -> None:
    size = len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if size > RUN_CACHE_MAX_ENTRY_BYTES:
        return
    with _run_cache_lock:
        if cache_key in _run_cache:
            del _run_cache[cache_key]
            _run_cache_sizes.pop(cache_key, None)
        while _run_cache and (
            len(_run_cache) >= RUN_CACHE_MAX_ENTRIES
            or _cache_total_locked() + size > RUN_CACHE_MAX_BYTES
        ):
            old_key, _ = _run_cache.popitem(last=False)
            _run_cache_sizes.pop(old_key, None)
        if _cache_total_locked() + size > RUN_CACHE_MAX_BYTES:
            return
        _run_cache[cache_key] = result
        _run_cache_sizes[cache_key] = size


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


