"""Small accessors shared by run and chat diagnostics."""

from __future__ import annotations

from typing import Any


def client_calls(client: Any) -> int:
    return int(getattr(client, "calls", 0) or 0)


def client_tokens(client: Any) -> int:
    return int(getattr(client, "total_tokens", 0) or 0)
