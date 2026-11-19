"""Per-request correlation ID bound by the HTTP middleware."""

from __future__ import annotations

from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def request_id() -> str:
    return _request_id_var.get()


def bind_request_id(value: str):
    return _request_id_var.set(value)


def reset_request_id(token) -> None:
    _request_id_var.reset(token)
