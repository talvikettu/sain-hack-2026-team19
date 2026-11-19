"""Organizer chat proxy client with bounded calls, usage, and no secret logs."""

import json
import os
import threading
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from rag_preprocessing.embeddings import NoRedirect


class ChatError(RuntimeError):
    """Base class for failures while talking to the organizer proxy."""


class ChatTransientError(ChatError):
    """Proxy failure that may be retried after a bounded delay."""

    def __init__(self, message, retry_after=None, *, status_code=None, retryable=True):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code
        self.retryable = retryable


class ChatTerminalError(ChatError):
    """Proxy failure that a retry cannot repair (bad request or credentials)."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = False
        self.terminal = True


class BudgetError(ChatError):
    """The local per-process call or token budget is exhausted."""


class ProxyProtocolError(ChatError):
    """The proxy answered, but the response cannot be safely accounted for."""


class ChatClient:
    def __init__(self, base_url, api_key, model, *, timeout=120, max_calls=12,
                 token_budget=120000, min_interval=2.1):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Chat proxy must be an HTTPS base URL without credentials or query")
        if not api_key or not model or timeout <= 0 or max_calls < 1 or token_budget < 1 or min_interval < 0:
            raise ValueError("Invalid chat client settings")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key, self.model = api_key, model
        self.timeout, self.max_calls = timeout, max_calls
        self.token_budget, self.min_interval = token_budget, min_interval
        self.calls, self.total_tokens, self.last_request = 0, 0, 0.0
        self.lock = threading.Lock()
        self.opener = build_opener(NoRedirect())

    @classmethod
    def from_env(cls, **kwargs):
        names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        if any(not os.environ.get(name) for name in names):
            raise ValueError("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required")
        return cls(*(os.environ[name] for name in names), **kwargs)

    def complete(self, messages, *, deadline=None, max_tokens=1800,
                 call_ceiling=None, token_ceiling=None):
        # Byte count is a conservative guard, not a model-token measurement.
        reserve = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + max_tokens + 1000
        with self.lock:
            if self.calls >= self.max_calls or self.total_tokens + reserve > self.token_budget:
                raise BudgetError("Local LLM call/token budget exhausted")
            # A chat must be checked again at the actual generation call: the
            # first attempt of a turn can consume the last allowance and the
            # repair attempt must not cross the reserved official-run budget.
            if call_ceiling is not None and self.calls >= call_ceiling:
                raise BudgetError("Chat call allowance exhausted")
            if token_ceiling is not None and self.total_tokens + reserve > token_ceiling:
                raise BudgetError("Chat token allowance exhausted")
            wait = max(0, self.last_request + self.min_interval - time.monotonic())
            if deadline is not None and time.monotonic() + wait >= deadline:
                raise TimeoutError("LLM deadline exceeded")
            if wait:
                time.sleep(wait)
            self.calls += 1
            self.last_request = time.monotonic()
        remaining = self.timeout if deadline is None else min(self.timeout, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("LLM deadline exceeded")
        body = json.dumps({
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, data=body, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
        })
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=remaining) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ProxyProtocolError("Chat response too large; stop to protect budget")
                result = json.loads(raw)
        except HTTPError as exc:
            retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
            if not retryable:
                raise ChatTerminalError(
                    f"Chat proxy HTTP {exc.code}", status_code=exc.code
                ) from None
            retry_after = None
            header = exc.headers.get("Retry-After") if exc.headers else None
            if header:
                try:
                    retry_after = max(0.0, float(header))
                except ValueError:
                    retry_after = None
            raise ChatTransientError(
                f"Chat proxy HTTP {exc.code}; no automatic retry",
                retry_after, status_code=exc.code, retryable=True,
            ) from None
        except (URLError, TimeoutError, HTTPException):
            raise ChatTransientError("Chat proxy unavailable or timed out; no automatic retry") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProxyProtocolError("Chat proxy returned invalid JSON; stop to protect budget") from None
        if not isinstance(result, dict):
            raise ProxyProtocolError("Chat proxy response must be a JSON object; stop to protect budget")
        usage = result.get("usage")
        if not isinstance(usage, dict) or type(usage.get("total_tokens")) is not int or usage["total_tokens"] < 0:
            raise ProxyProtocolError("Proxy response lacks valid token usage; stop to protect budget")
        with self.lock:
            self.total_tokens += usage["total_tokens"]
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProxyProtocolError("Proxy returned no choices; stop to protect budget")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProxyProtocolError("Proxy returned an invalid choice; stop to protect budget")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProxyProtocolError("Proxy returned an invalid message; stop to protect budget")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProxyProtocolError("Proxy returned no text content; stop to protect budget")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProxyProtocolError("Proxy returned an invalid finish reason; stop to protect budget")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("LLM deadline exceeded")
        return {"content": content, "finish_reason": finish_reason,
                "usage": usage, "seconds": round(time.monotonic() - started, 3),
                "wait_seconds": round(wait, 3)}
