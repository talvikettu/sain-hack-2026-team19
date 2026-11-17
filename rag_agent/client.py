"""Organizer chat proxy client with bounded calls, usage, and no secret logs."""

import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from rag_preprocessing.embeddings import NoRedirect


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

    def complete(self, messages, *, deadline=None, max_tokens=1800):
        # Byte count is a conservative guard, not a model-token measurement.
        reserve = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + max_tokens + 1000
        with self.lock:
            if self.calls >= self.max_calls or self.total_tokens + reserve > self.token_budget:
                raise RuntimeError("Local LLM call/token budget exhausted")
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
        request = Request(self.endpoint, data=json.dumps({
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
        }, ensure_ascii=False).encode("utf-8"), headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
        })
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=remaining) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise RuntimeError("Chat response too large")
                result = json.loads(raw)
        except HTTPError as exc:
            raise RuntimeError(f"Chat proxy HTTP {exc.code}; no automatic retry") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Chat proxy unavailable or timed out; no automatic retry") from None
        usage = result.get("usage", {})
        if not isinstance(usage, dict) or type(usage.get("total_tokens")) is not int or usage["total_tokens"] < 0:
            raise RuntimeError("Proxy response lacks valid token usage; stop to protect budget")
        with self.lock:
            self.total_tokens += usage["total_tokens"]
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Proxy returned no choices")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("Proxy returned no text content")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("LLM deadline exceeded")
        return {"content": content, "finish_reason": choice.get("finish_reason"),
                "usage": usage, "seconds": round(time.monotonic() - started, 3)}
