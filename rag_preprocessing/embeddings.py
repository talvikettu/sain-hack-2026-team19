"""Explicitly enabled remote embeddings; never downloads model weights."""

import json
import math
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def normalize(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("Embedding must be a non-empty numeric vector")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in vector):
        raise ValueError("Embedding contains non-numeric values")
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains non-finite values")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding norm")
    return [x / norm for x in vector]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a team credential to a redirected URL.


class ProxyEmbeddings:
    """OpenAI-format adapter requiring an explicitly approved endpoint/model.

    No automatic retries: callers own the shared rate limit and execution budget.
    """

    def __init__(self, endpoint, model, api_key, allowed_base, timeout=30, min_interval=2.1):
        def origin(url):
            parts = urlsplit(url)
            if (parts.scheme != "https" or not parts.hostname or parts.username
                    or parts.password or parts.query or parts.fragment):
                raise ValueError("Embedding URLs must be HTTPS without credentials/query/fragment")
            return parts.hostname, parts.port or 443

        if origin(endpoint) != origin(allowed_base):
            raise ValueError("Embedding endpoint must use the organizer proxy origin")
        base_path = urlsplit(allowed_base).path.rstrip("/")
        if not urlsplit(endpoint).path.startswith(base_path + "/"):
            raise ValueError("Embedding endpoint must be inside the configured proxy base path")
        if not model or not api_key or timeout <= 0 or min_interval < 0:
            raise ValueError("Explicit embedding model, API key, and positive timeout required")
        self.endpoint, self.model = endpoint, model
        self.api_key, self.timeout = api_key, timeout
        self.identity = endpoint + "|" + model
        self.opener = build_opener(NoRedirect())
        self.min_interval, self.last_request = min_interval, 0.0
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls):
        if os.environ.get("RAG_EMBEDDINGS_APPROVED") != "1":
            raise ValueError("Embedding access must be confirmed: RAG_EMBEDDINGS_APPROVED=1")
        names = ("RAG_EMBEDDING_URL", "RAG_EMBEDDING_MODEL", "LLM_API_KEY", "LLM_BASE_URL")
        if any(not os.environ.get(name) for name in names):
            raise ValueError("Missing explicitly configured embedding/proxy environment variables")
        return cls(*(os.environ[name] for name in names))

    def embed(self, texts, *, deadline=None):
        if not texts:
            return []
        # Per adapter only. Backend must also coordinate the team's shared quota.
        with self.lock:
            wait = max(0, self.last_request + self.min_interval - time.monotonic())
            if deadline is not None and time.monotonic() + wait >= deadline:
                raise TimeoutError("Embedding deadline exceeded before next request")
            if wait:
                time.sleep(wait)
            self.last_request = time.monotonic()
        remaining = self.timeout if deadline is None else min(self.timeout, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Embedding deadline exceeded")
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(self.endpoint, data=body, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
        })
        try:
            with self.opener.open(request, timeout=remaining) as response:
                # Bound response memory; never print request URLs, inputs, or error bodies.
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("Embedding response exceeds memory guard")
                result = json.loads(raw)
        except HTTPError as exc:
            raise RuntimeError(f"Embedding proxy returned HTTP {exc.code}; no automatic retry") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Embedding proxy unavailable; no automatic retry") from None
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding response count mismatch")
        ordered = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if type(index) is not int or not 0 <= index < len(texts) or index in ordered:
                raise ValueError("Invalid or duplicate embedding response index")
            ordered[index] = normalize(item.get("embedding"))
        vectors = [ordered[i] for i in range(len(texts))]
        if len({len(v) for v in vectors}) != 1:
            raise ValueError("Inconsistent embedding dimensions")
        return vectors
