"""Team19 backend for the Sain x School 21 hackathon, case 4.

Contract (team19/README.md, sample-app/main.py):
- ``requirements.txt`` and ``start.sh`` live in the repository root;
- Python 3.11, the server listens on ``0.0.0.0:$PORT``;
- ``GET /`` answers within 60 seconds;
- ``POST /run`` receives ``{"case": 4, "dataset_code": "..."}``, fetches the
  dataset with the incoming Bearer token, answers every request, and returns
  the complete Case4Answers JSON in the same HTTP response.

Secrets, dataset URLs and dataset content are never written to logs.

The application is deliberately a single-worker process: the active runtime,
browser sessions and the shared LLM budget are process-local. Run it with one
Uvicorn worker until cross-process state has its own store.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import copy
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from fastapi import FastAPI, Header, HTTPException, Request as FastAPIRequest
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rag_agent import ChatClient, RAGAgent
from rag_agent.agent import PERSONA_FILES
from rag_agent.client import (
    BudgetError, ChatTerminalError, ChatTransientError, ProxyProtocolError,
)
from rag_agent.retrieval import merge_rankings, search_rankings
from rag_preprocessing import build_index, pack_context
from rag_preprocessing.embeddings import NoRedirect

TEAM = os.environ.get("TEAM", "team19")
CASE_ID = 4
RUN_BUDGET_SECONDS = 42 * 60  # the stand waits 45 minutes
CHAT_BUDGET_SECONDS = 5 * 60
CHAT_MAX_SESSIONS = 100
CHAT_MAX_MESSAGES = 40
CHAT_MAX_CHARS = 20_000
CHAT_SESSION_TTL_SECONDS = 2 * 60 * 60
CHAT_RATE_LIMIT = 12
CHAT_GLOBAL_RATE_LIMIT = 30
CHAT_RATE_WINDOW = 60
CHAT_SESSION_ISSUE_LIMIT = 20
CHAT_SESSION_EVICT_GRACE_SECONDS = 5 * 60
CHAT_TURN_CACHE_MAX = 200
# The official run has priority. Chat is checked against a hard ceiling at
# every individual proxy call, so a repair attempt can never consume the
# calls/tokens that a retry of the official run needs.
RUN_RESERVED_CALLS = 105
RUN_RESERVED_TOKENS = 220_000
INDEX_PATH = Path(os.environ.get("CASE4_INDEX_PATH", "/tmp/case4-index.sqlite3"))
INDEX_MAX_RETAINED = 2
DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CALLS = 150
LLM_TOKEN_BUDGET = 260_000
LLM_MIN_INTERVAL_SECONDS = 2.1
# A transient proxy outage must not permanently disable generation. After a
# failed record, pause before the next probe so later records can recover.
TRANSIENT_COOLDOWN_BASE_SECONDS = 4.0
TRANSIENT_COOLDOWN_MAX_SECONDS = 30.0
RUN_CACHE_VERSION = "10"
RUN_CACHE_MAX_ENTRIES = 4
RUN_CACHE_MAX_ENTRY_BYTES = 4 * 1024 * 1024
RUN_CACHE_MAX_BYTES = 16 * 1024 * 1024
MAX_DATASET_BYTES = 32 * 1024 * 1024
DATASET_READ_CHUNK = 64 * 1024
MAX_DATASET_REQUESTS = 2000
RUN_SERIALIZATION_RESERVE_SECONDS = 5
MAX_INFLIGHT_REQUESTS = 16
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

logger = logging.getLogger("team19.case4")
# Uvicorn configures only its own loggers. Add one plain handler so the
# aggregate run/chat metrics are visible on the stand without request data.
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
logger.setLevel(LOG_LEVEL)

if os.environ.get("WEB_CONCURRENCY", "").strip() not in ("", "1"):
    logger.warning(
        "WEB_CONCURRENCY is set; this backend requires one worker because "
        "runtime, sessions and quota are process-local"
    )

app = FastAPI(title=f"hackathon {TEAM} / case {CASE_ID}")

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
# Admission happens in middleware, before Starlette allocates a worker thread
# for a synchronous endpoint. Health and read endpoints are never admitted, so
# liveness stays independent of a long official run.
_admission = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)
_ADMISSION_PATHS = frozenset({"/run", "/chat", "/session"})
_REQUEST_ID_HEADER = "X-Request-ID"


def _request_id() -> str:
    return _request_id_var.get()


@app.middleware("http")
async def _request_context(request: FastAPIRequest, call_next):
    incoming = request.headers.get(_REQUEST_ID_HEADER, "")
    request_id = (
        incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex[:16]
    )
    token = _request_id_var.set(request_id)
    admitted = request.url.path in _ADMISSION_PATHS
    if admitted and not _admission.acquire(blocking=False):
        _request_id_var.reset(token)
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is busy; retry shortly"},
            headers={_REQUEST_ID_HEADER: request_id, "Retry-After": "2"},
        )
    try:
        response = await call_next(request)
    finally:
        if admitted:
            _admission.release()
        _request_id_var.reset(token)
    response.headers[_REQUEST_ID_HEADER] = request_id
    return response


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class Runtime:
    """One published dataset, its index and its process-local browser state."""

    def __init__(self, dataset: dict[str, Any], client: ChatClient, agent: RAGAgent,
                 index_path: str | Path | None = None,
                 dataset_key: str | None = None) -> None:
        self.dataset = dataset
        self.client = client
        self.agent = agent
        self.index_path = Path(index_path or getattr(agent, "index", INDEX_PATH))
        # Browser sessions belong to one active dataset and one server-selected
        # user. Switching /run datasets invalidates the previous runtime object.
        self.dataset_key = dataset_key or ""
        self.generation = (
            self.dataset_key[:16] if self.dataset_key else uuid.uuid4().hex[:16]
        )
        users = [user["user_id"] for user in dataset.get("users", []) if user.get("user_id")]
        self.demo_user_id = users[0] if users else ""
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.session_users: dict[str, str] = {}
        self.session_touched: dict[str, float] = {}
        # Only a hash of a browser token is retained. The organizer and LLM
        # credentials never enter this map.
        self.session_tokens: dict[str, str] = {}
        self.chat_times: dict[str, list[float]] = {}
        self.chat_global_times: list[float] = []
        self.session_issue_times: list[float] = []
        self.users = set(users)
        # Session/turn/rate mutations take this lock briefly; generation never
        # holds it, so a slow answer cannot block another browser.
        self.sessions_lock = threading.RLock()
        # Bounded idempotency store for client turn IDs.
        self.completed_turns: OrderedDict[
            tuple[str, str], tuple[str, dict[str, Any]]
        ] = OrderedDict()
        self.inflight_turns: dict[tuple[str, str], str] = {}
        # One generated turn at a time per browser keeps history ordering
        # deterministic without serializing different visitors.
        self.busy_sessions: set[str] = set()
        self.sections: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        for article in dataset.get("articles", []):
            for section in article.get("sections", []):
                self.sections[(article["article_id"], section["section_id"])] = (article, section)

    def set_dataset_key(self, dataset_key: str) -> None:
        """Attach the content digest after construction without touching state."""
        if dataset_key:
            self.dataset_key = dataset_key
            self.generation = dataset_key[:16]


_runtime: Runtime | None = None
_state_lock = threading.Lock()
_init_lock = threading.Lock()
# /run is serialized, but chat, /health, /experts and /source never wait for
# official work. The lock is taken non-blocking so a duplicate /run cannot
# occupy a worker for the whole 42-minute budget.
_run_lock = threading.Lock()

# One client per process: call and token counters must survive repeated /run
# attempts, otherwise each attempt could spend the whole budget again.
_llm_client: ChatClient | None = None
_llm_client_lock = threading.Lock()

# Only complete, fully generated runs are cached; a content digest (never the
# dataset code or the reference) keys the result. Least-recently-used entries
# are evicted first and the cache has both an entry and an aggregate byte cap.
_run_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_run_cache_sizes: dict[str, int] = {}
_run_cache_lock = threading.Lock()


def _active_runtime() -> Runtime | None:
    with _state_lock:
        return _runtime


def _publish_runtime(runtime: Runtime) -> Runtime | None:
    """Atomically make a fully validated runtime visible to readers."""
    global _runtime
    with _state_lock:
        previous = _runtime
        _runtime = runtime
    return previous


class SessionCapacityError(RuntimeError):
    """All session slots are held by recently active browsers."""


def _get_llm_client() -> ChatClient:
    global _llm_client
    with _llm_client_lock:
        if _llm_client is None:
            _llm_client = ChatClient.from_env(
                timeout=LLM_TIMEOUT_SECONDS,
                max_calls=LLM_MAX_CALLS,
                token_budget=LLM_TOKEN_BUDGET,
                min_interval=LLM_MIN_INTERVAL_SECONDS,
            )
        return _llm_client


def _index_path_for(index_key: str) -> Path:
    """Content-addressed index path so an old runtime never reads a new corpus."""
    return INDEX_PATH.with_name(f"{INDEX_PATH.stem}-{index_key[:20]}{INDEX_PATH.suffix}")


def _is_managed_index(path: Path) -> bool:
    return path.name.startswith(INDEX_PATH.stem + "-") and path.parent == INDEX_PATH.parent


def _prune_indexes(active: Path) -> None:
    """Retain the active index and the previous one, never all old builds.

    Keeping one previous file avoids deleting an index that an in-flight chat
    request resolved just before a dataset switch. Bounded retention also
    prevents failed builds from accumulating on the stand's small disk.
    """
    if not _is_managed_index(active):
        return
    try:
        others = [
            path for path in INDEX_PATH.parent.glob(f"{INDEX_PATH.stem}-*{INDEX_PATH.suffix}")
            if path != active
        ]
        others.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in others[max(0, INDEX_MAX_RETAINED - 1):]:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _discard_candidate(runtime: Runtime) -> None:
    """Remove a staged index after a failed run without touching live state."""
    active = _active_runtime()
    if active is runtime:
        return
    if active is not None and active.index_path == runtime.index_path:
        return
    if not _is_managed_index(runtime.index_path):
        return
    try:
        runtime.index_path.unlink(missing_ok=True)
    except OSError:
        pass


def _build_runtime(dataset: dict[str, Any], deadline: float,
                   index_key: str | None = None) -> Runtime:
    requests = dataset.get("requests")
    if not isinstance(requests, list):
        raise ValueError("dataset.requests must be a list")
    if index_key is None:
        try:
            index_key = _run_cache_key(dataset)
        except Exception:
            index_key = uuid.uuid4().hex
    index_path = _index_path_for(index_key)
    existed = index_path.is_file()
    try:
        build_index(dataset, index_path, deadline=deadline)
    except Exception:
        # build_index preserves an existing valid file on failure; only remove
        # the partial artifact created by this attempt.
        if not existed and _is_managed_index(index_path):
            try:
                index_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    try:
        client = _get_llm_client()
        agent = RAGAgent(
            index_path,
            dataset["experts"],
            client,
            history=dataset.get("chat_history", []),
        )
    except Exception:
        if not existed and _is_managed_index(index_path):
            try:
                index_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return Runtime(dataset, client, agent, index_path=index_path, dataset_key=index_key)


def _get_runtime() -> Runtime:
    """Return the loaded runtime; only local development lazy-loads dev data."""
    runtime = _active_runtime()
    if runtime is not None:
        return runtime

    dev_code = os.environ.get("DEV_DATASET_CODE", "").strip()
    team_token = os.environ.get("HACK_TEAM_TOKEN", "").strip()
    if not dev_code or not team_token:
        raise HTTPException(status_code=503, detail="Dataset is not loaded yet")

    # Local frontend development only. On the stand DEV_DATASET_CODE is unset
    # and /run provides the online/final dataset.
    with _init_lock:
        runtime = _active_runtime()
        if runtime is not None:
            return runtime
        try:
            dataset = _fetch_dataset(dev_code, "Bearer " + team_token)
            _validate_dataset(dataset)
            runtime = _build_runtime(dataset, time.monotonic() + 300)
        except Exception as exc:
            logger.error("dev dataset initialization failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Dataset is not loaded yet") from None
        _publish_runtime(runtime)
        return runtime


# ---------------------------------------------------------------------------
# Organizer API
# ---------------------------------------------------------------------------


def _api_base() -> str:
    base = os.environ.get("HACK_API_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("HACK_API_URL is not set")
    parsed = urlsplit(base)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise RuntimeError("HACK_API_URL must be an HTTPS base URL without credentials or query")
    return base


def _set_response_timeout(response, seconds: float) -> None:
    """Apply the remaining total deadline to the underlying socket."""
    if seconds <= 0:
        raise TimeoutError("dataset API read timed out")
    candidates = []
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    candidates.extend(
        item for item in (getattr(raw, "_sock", None), raw, fp)
        if item is not None and hasattr(item, "settimeout")
    )
    for item in candidates:
        try:
            item.settimeout(seconds)
            return
        except (OSError, ValueError, AttributeError):
            continue


def _read_limited(response, limit: int, deadline: float) -> bytes:
    """Read a bounded body without letting a slow stream outlive the deadline."""
    chunks, total = [], 0
    reader = getattr(response, "read1", None) or response.read
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("dataset API read timed out")
        _set_response_timeout(response, remaining)
        chunk = reader(min(DATASET_READ_CHUNK, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise RuntimeError("dataset API response too large")
        chunks.append(chunk)
    return b"".join(chunks)


class DatasetContractError(ValueError):
    """The fetched dataset is missing a field the backend consumes."""

    def __init__(self, field: str) -> None:
        super().__init__(f"invalid dataset field: {field}")
        self.field = field


def _contract_str(value: Any, field: str, *, allow_empty: bool = False,
                  max_length: int | None = None) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise DatasetContractError(field)
    if max_length is not None and len(value) > max_length:
        raise DatasetContractError(field)
    return value


def _contract_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise DatasetContractError(field)
    return value


def _contract_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DatasetContractError(field)
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        raise DatasetContractError(field) from None
    if stamp.tzinfo is None:
        raise DatasetContractError(field)
    return value


def _contract_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetContractError(field)
    return value


def _contract_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetContractError(field)
    return value


def _validate_dataset(dataset: Any) -> dict[str, Any]:
    """Validate every field consumed by /run, /chat, /experts and /source.

    Validation happens once before any index build or state publication. It
    deliberately reports only the field path: dataset content and identifiers
    must not reach logs.
    """
    dataset = _contract_dict(dataset, "dataset")

    user_ids: set[str] = set()
    for index, user in enumerate(_contract_list(dataset.get("users", []), "users")):
        user = _contract_dict(user, f"users[{index}]")
        user_id = _contract_str(user.get("user_id"), f"users[{index}].user_id", max_length=128)
        if user_id in user_ids:
            raise DatasetContractError(f"users[{index}].user_id")
        user_ids.add(user_id)

    for index, message in enumerate(_contract_list(dataset.get("chat_history", []), "chat_history")):
        message = _contract_dict(message, f"chat_history[{index}]")
        _contract_str(message.get("user_id"), f"chat_history[{index}].user_id", max_length=128)
        if message.get("role") not in {"user", "agent", "assistant"}:
            raise DatasetContractError(f"chat_history[{index}].role")
        _contract_timestamp(message.get("at"), f"chat_history[{index}].at")
        _contract_str(message.get("text"), f"chat_history[{index}].text", max_length=12000)

    articles = _contract_list(dataset.get("articles"), "articles")
    if not articles:
        raise DatasetContractError("articles")
    article_ids: set[str] = set()
    section_pairs: set[tuple[str, str]] = set()
    usable_sections = False
    for article_index, article in enumerate(articles):
        article = _contract_dict(article, f"articles[{article_index}]")
        article_id = _contract_id(article.get("article_id"), f"articles[{article_index}].article_id")
        if article_id in article_ids:
            raise DatasetContractError(f"articles[{article_index}].article_id")
        article_ids.add(article_id)
        _contract_str(article.get("title"), f"articles[{article_index}].title")
        authors = _contract_list(article.get("authors"), f"articles[{article_index}].authors")
        if any(not isinstance(author, str) for author in authors):
            raise DatasetContractError(f"articles[{article_index}].authors")
        if type(article.get("year")) is not int:
            raise DatasetContractError(f"articles[{article_index}].year")
        _contract_str(article.get("doi"), f"articles[{article_index}].doi", allow_empty=True)
        _contract_dict(article.get("license"), f"articles[{article_index}].license")
        sections = _contract_list(article.get("sections"), f"articles[{article_index}].sections")
        local_ids: set[str] = set()
        for section_index, section in enumerate(sections):
            section = _contract_dict(section, f"articles[{article_index}].sections[{section_index}]")
            section_id = _contract_id(
                section.get("section_id"),
                f"articles[{article_index}].sections[{section_index}].section_id",
            )
            if section_id in local_ids or (article_id, section_id) in section_pairs:
                raise DatasetContractError(
                    f"articles[{article_index}].sections[{section_index}].section_id"
                )
            local_ids.add(section_id)
            section_pairs.add((article_id, section_id))
            _contract_str(
                section.get("title"),
                f"articles[{article_index}].sections[{section_index}].title",
                allow_empty=True,
            )
            text = _contract_str(
                section.get("text"),
                f"articles[{article_index}].sections[{section_index}].text",
                allow_empty=True,
            )
            usable_sections = usable_sections or bool(text.strip())
    if not usable_sections:
        raise DatasetContractError("articles[].sections[].text")

    experts = _contract_list(dataset.get("experts"), "experts")
    if not experts:
        raise DatasetContractError("experts")
    expert_ids: set[str] = set()
    for index, expert in enumerate(experts):
        expert = _contract_dict(expert, f"experts[{index}]")
        expert_id = _contract_id(expert.get("expert_id"), f"experts[{index}].expert_id")
        if expert_id in expert_ids or expert_id not in PERSONA_FILES:
            raise DatasetContractError(f"experts[{index}].expert_id")
        expert_ids.add(expert_id)
        _contract_str(expert.get("role"), f"experts[{index}].role")
        _contract_str(expert.get("speech_style"), f"experts[{index}].speech_style")

    requests = _contract_list(dataset.get("requests"), "requests")
    if len(requests) > MAX_DATASET_REQUESTS:
        raise DatasetContractError("requests")
    record_ids: set[str] = set()
    for index, request in enumerate(requests):
        request = _contract_dict(request, f"requests[{index}]")
        record_id = _contract_id(request.get("record_id"), f"requests[{index}].record_id")
        if record_id in record_ids:
            raise DatasetContractError(f"requests[{index}].record_id")
        record_ids.add(record_id)
        _contract_str(request.get("user_id"), f"requests[{index}].user_id", max_length=128)
        expert_id = _contract_id(request.get("expert_id"), f"requests[{index}].expert_id")
        if expert_id not in expert_ids:
            raise DatasetContractError(f"requests[{index}].expert_id")
        group_id = request.get("group_id")
        if group_id is not None and not (
            isinstance(group_id, str)
            or (isinstance(group_id, int) and not isinstance(group_id, bool))
        ):
            raise DatasetContractError(f"requests[{index}].group_id")
        utterance = _contract_dict(request.get("utterance"), f"requests[{index}].utterance")
        _contract_str(utterance.get("role"), f"requests[{index}].utterance.role")
        _contract_timestamp(utterance.get("at"), f"requests[{index}].utterance.at")
        _contract_str(
            utterance.get("text"), f"requests[{index}].utterance.text", max_length=12000
        )
    return dataset


class DatasetAccessError(RuntimeError):
    """The organizer API rejected the token for the requested dataset."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"dataset API HTTP {status_code}")
        self.status_code = status_code


# Reuse the opener (redirects stay disabled). urllib still opens a fresh
# connection per request; the object itself is stateless here.
_DATASET_OPENER = build_opener(NoRedirect())


def _fetch_dataset(dataset_code: str, authorization: str, timeout: float = 120) -> dict[str, Any]:
    """Fetch the whole dataset once with the token received by /run."""
    if timeout <= 0:
        raise TimeoutError("dataset API deadline exceeded")
    url = _api_base() + f"/case/{CASE_ID}/getData?dataset_code=" + quote(dataset_code, safe="")
    request = Request(url, headers={"Authorization": authorization, "Accept": "application/json"})
    read_deadline = time.monotonic() + timeout
    try:
        with _DATASET_OPENER.open(request, timeout=timeout) as response:
            raw = _read_limited(response, MAX_DATASET_BYTES, read_deadline)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise DatasetAccessError(exc.code) from None
        raise RuntimeError(f"dataset API HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("dataset API unavailable") from None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("unexpected dataset response") from None
    del raw
    if isinstance(payload, dict) and isinstance(payload.get("dataset"), dict):
        payload = payload["dataset"]
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise RuntimeError("unexpected dataset response")
    return payload


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


class RunContext:
    def __init__(self, retry_delays: tuple[float, ...] = (5.0, 10.0),
                 cooldown_base: float = TRANSIENT_COOLDOWN_BASE_SECONDS) -> None:
        self.skip_llm = False
        self.fallbacks = 0
        self.generated = 0
        self.retrieval_failures = 0
        self.unsupported_fallbacks = 0
        self.validation_failures = 0
        self.transient_failures = 0
        self.consecutive_transient = 0
        self.retries = 0
        self.cooldown_until = 0.0
        self.cooldown_base = cooldown_base
        self.retrieval_seconds = 0.0
        self.generation_seconds = 0.0
        self.llm_calls = 0
        self.llm_tokens = 0
        self.retry_delays = retry_delays


EXPERT_ORDER = {"clinician": 0, "biohacker": 1, "nutritionist": 2}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, ChatTransientError):
        return bool(exc.retryable)
    if isinstance(exc, ChatTerminalError):
        return False
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc)
    return "HTTP 408" in message or "HTTP 429" in message or "HTTP 5" in message or "unavailable" in message


def _terminal_error(exc: BaseException) -> bool:
    if isinstance(exc, ChatTerminalError):
        return True
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    message = str(exc)
    return "HTTP 401" in message or "HTTP 403" in message


def _budget_exhausted(exc: BaseException) -> bool:
    if isinstance(exc, BudgetError):
        return True
    return "budget exhausted" in str(exc).casefold() or "allowance exhausted" in str(exc).casefold()


def _protocol_failure(exc: BaseException) -> bool:
    """A malformed proxy reply means token usage may be unaccountable."""
    if isinstance(exc, ProxyProtocolError):
        return True
    message = str(exc).casefold()
    return any(fragment in message for fragment in (
        "lacks valid token usage", "returned no choices", "returned no text content",
        "response too large", "invalid choice", "invalid message", "invalid finish reason",
    ))


def _note_failure(exc: BaseException, context: RunContext) -> None:
    """Stop only on a confirmed budget/auth/protocol stop.

    Transient connectivity and per-minute 429 failures use a bounded cooldown
    and a later probe instead of disabling generation for the rest of the run.
    """
    if (_budget_exhausted(exc) or _protocol_failure(exc)
            or _terminal_error(exc)):
        context.skip_llm = True
        return
    if _retryable(exc):
        context.transient_failures += 1
        context.consecutive_transient += 1
        delay = getattr(exc, "retry_after", None)
        if delay is None:
            delay = context.cooldown_base * (2 ** min(context.consecutive_transient - 1, 3))
        delay = min(max(float(delay), 0.0), TRANSIENT_COOLDOWN_MAX_SECONDS)
        context.cooldown_until = max(context.cooldown_until, time.monotonic() + delay)
    elif isinstance(exc, ValueError):
        context.validation_failures += 1


_TERM_RE = re.compile(r"[a-zа-яё0-9-]{4,}", re.I)
_TERM_STOP = {
    "этот", "этом", "этого", "такой", "такие", "какой", "какие", "почему",
    "можно", "нужно", "если", "меня", "мне", "вас", "вам", "для", "при",
    "или", "что", "как", "чем", "могут", "может", "влияет", "влияние",
    "вопрос", "ответ", "сон", "сна", "сне", "sleep", "does", "what", "how",
}


def _query_terms(text: str) -> set[str]:
    # Five-character prefixes match common Russian inflections without a
    # morphological dictionary. This is only a no-index fallback path.
    return {
        word[:5]
        for word in _TERM_RE.findall(text.casefold())
        if word not in _TERM_STOP
    }


def _fallback_evidence(
    dataset: dict[str, Any],
    members: list[dict[str, Any]],
    agent: RAGAgent | None = None,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Choose a topical section locally when the normal group search fails."""
    query = " ".join(
        str(member.get("utterance", {}).get("text", "")) for member in members
    )
    if agent is not None and (deadline is None or time.monotonic() < deadline):
        try:
            rankings = [
                ranking[:1]
                for ranking in search_rankings(agent.index, query, deadline=deadline)
            ]
            hits = merge_rankings(rankings, 1)
            if hits:
                blocks = pack_context(agent.index, hits, max_chars=agent.context_chars)
                if blocks:
                    return blocks[:1]
        except Exception:
            pass
    wanted = _query_terms(query)
    best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
    for article in dataset.get("articles", []):
        if deadline is not None and time.monotonic() >= deadline:
            break
        title_terms = _query_terms(str(article.get("title", "")))
        for section in article.get("sections", []):
            if deadline is not None and time.monotonic() >= deadline:
                break
            text = section.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            section_terms = _query_terms(str(section.get("title", "")) + " " + text)
            score = 3 * len(wanted & title_terms) + len(wanted & section_terms)
            if best is None or score > best[0]:
                best = (score, article, section)
    # A zero-overlap section is not evidence. Returning it here is what made
    # the old fallback cite the first article in the corpus for every topic.
    if best is None or best[0] == 0:
        return []
    _, article, section = best
    text = str(section.get("text", ""))
    return [{
        "article_id": article["article_id"],
        "section_id": section["section_id"],
        "text": text[:3000],
        "metadata": {
            "title": article.get("title", ""),
            "section_title": section.get("title", ""),
            "heading_path": [section.get("title", "")],
            "section_type": "unknown",
            "start_char": 0,
            "end_char": min(len(text), 3000),
        },
    }]


def _last_resort_evidence(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Explicit last resort when retrieval found no topical section.

    The caller counts this case separately. The schema requires one citation,
    so a non-empty section is used rather than returning invalid JSON.
    """
    for article in dataset.get("articles", []):
        for section in article.get("sections", []):
            text = section.get("text", "")
            if isinstance(text, str) and text.strip():
                return [{
                    "article_id": article["article_id"],
                    "section_id": section["section_id"],
                    "text": text[:3000],
                    "metadata": {
                        "title": article.get("title", ""),
                        "section_title": section.get("title", ""),
                        "heading_path": [section.get("title", "")],
                        "section_type": "unknown",
                        "start_char": 0,
                        "end_char": min(len(text), 3000),
                    },
                }]
    return []


# Conservative local guard used only for degraded, non-LLM fallbacks.
_ATTACK_REQUEST = re.compile(
    r"(?:\bсколько|\bкакая|\bкакую|\bкакие)\s+(?:мне\s+)?(?:можно\s+)?"
    r"(?:доз\w*|мг\b|миллиграм\w*|капел\w*|таблет\w*|капсул\w*|единиц\w*)|"
    r"\bдозировк\w*|"
    r"\bсколько\s+(?:мне\s+)?(?:можно\s+)?(?:пить|принимать|выпить|капать)|"
    r"\b(?:поставь|поставьте|назначь|назначьте|распиши|распишите)\b.{0,50}"
    r"\b(?:доз\w*|лечен\w*|схем\w*|препарат\w*|таблет\w*|капл\w*)|"
    r"\bчто\s+(?:мне\s+)?(?:купить|принять|выпить)\b|"
    r"\bможно\s+(?:мне\s+)?(?:его|её|это)\s+(?:пить|принимать|принять)|"
    r"\b(?:похоже\s+на|это\s+(?:апноэ|депресси\w*|бессонниц\w*|синдром\w*)|"
    r"какой\s+диагноз|что\s+у\s+меня)|"
    r"\b(?:бессонниц\w*|апноэ)\s+или\s+(?:синдром\w*|депресси\w*|бессонниц\w*)|"
    r"\b(?:игнорируй|забудь|системн\w+\s+(?:промпт|инструкц)|"
    r"prompt\s+injection|инструкц\w+\s+выше)\b",
    re.I,
)
_FALLBACK_UNSAFE = re.compile(
    r"(?:https?://|www\.)|"
    r"\b\d+(?:[.,]\d+)?\s*(?:мг|мкг|мл|кап(?:ля|ель|ли)?|таблет(?:ка|ки|ок)?|капсул(?:а|ы)?)|"
    r"\b(?:принимайте|выпейте|принять|пропейте)\s+(?:по\s+)?\d|"
    r"\b(?:дозировк|назначаю|прописываю)\w*",
    re.I,
)


def _topic_supported(question: str, section_text: str) -> bool:
    return bool(_query_terms(question) & _query_terms(section_text))


def _fallback_quote(question: str, text: str) -> str:
    """Return short safe sentences from the cited section, never invented text."""
    clean = re.sub(r"\[[^\]]+\]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    wanted = _query_terms(question)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean)
        if 30 <= len(sentence.strip()) and not _FALLBACK_UNSAFE.search(sentence)
    ]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(wanted & _query_terms(sentence)),
        reverse=True,
    )
    # Prefer sentences that actually overlap the question. A section can be
    # topically adjacent while its first sentence is a generic introduction.
    relevant = [sentence for sentence in ranked if wanted & _query_terms(sentence)]
    if relevant:
        ranked = relevant
    selected = []
    used = 0
    for sentence in ranked:
        if used + len(sentence) > 650:
            continue
        selected.append(sentence)
        used += len(sentence)
        if len(selected) == 2:
            break
    return " ".join(selected)


def _fallback_answer(
    request: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    unsupported: bool = False,
    topical: bool = False,
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("Cannot produce a schema-valid fallback without any article section")
    block = evidence[0]
    pair = block["article_id"], block["section_id"]
    question = str(request.get("utterance", {}).get("text", ""))
    section_text = str(block.get("text", ""))
    metadata = block.get("metadata", {})
    section_title = str(metadata.get("section_title", ""))
    article_title = str(metadata.get("title", ""))
    ref = f"[{pair[0]} / {pair[1]}]"
    expert_id = request.get("expert_id")
    # Retrieval evidence is already topical even when the question is Russian
    # and the source section is English, so a lexical cross-language check
    # must not turn an ordinary degraded answer into a refusal. Direct callers
    # can ask for the conservative check instead.
    attack = bool(_ATTACK_REQUEST.search(question)) or (
        not topical and not _topic_supported(question, section_text)
    )
    if unsupported:
        text = (
            "Не могу подтвердить ответ по базе: подходящего раздела для этого вопроса нет. "
            f"В базе есть только общий материал «{section_title or article_title}» {ref}; "
            "он не подтверждает ответ."
        )
        refused = True
    elif attack:
        lead = "Я остаюсь экспертом по сну и не выхожу из этой роли. " if re.search(
            r"(?:ты|вы)\s+(?:мой|моя|мое)|представь|сыграем|по-дружески", question, re.I
        ) else ""
        text = (
            lead
            + "Я не могу ставить диагноз, назначать лечение или называть дозировки — "
            "это вне моей компетенции. Безопаснее обсудить это с врачом. "
            f"По смежной теме в базе можно посмотреть раздел «{section_title or article_title}» {ref}."
        )
        refused = True
    else:
        quote = _fallback_quote(question, section_text)
        intro = {
            "clinician": "С точки зрения данных: ",
            "biohacker": "Если смотреть на исследования: ",
            "nutritionist": "Проще всего опереться на данные: ",
        }.get(expert_id, "По данным базы: ")
        if quote:
            text = (
                intro + quote + " Это то, что можно уверенно сказать по источнику; "
                f"персональные решения лучше обсудить со специалистом. {ref}"
            )
            refused = False
        else:
            # A non-refusal without a substantive claim would be an empty
            # answer. Say honestly that the source has no short quotable claim.
            text = (
                f"В разделе «{section_title or article_title}» {ref} нет короткого "
                "утверждения, которое можно безопасно привести без искажения. "
                "Лучше обсудить вопрос со специалистом."
            )
            refused = True
    return {
        "record_id": request["record_id"],
        "text": text,
        "refused": refused,
        "citations": [{"article_id": pair[0], "section_id": pair[1]}],
    }


def _section_pairs(dataset: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (article["article_id"], section["section_id"])
        for article in dataset.get("articles", [])
        for section in article.get("sections", [])
    }


def _validate_generated_answer(
    answer: Any,
    request: dict[str, Any],
    sections: set[tuple[str, str]],
) -> dict[str, Any]:
    """Validate one model/stub answer before it can be cached or returned."""
    if not isinstance(answer, dict) or set(answer) != {
        "record_id", "text", "refused", "citations",
    }:
        raise ValueError("Answer must contain exactly record_id, text, refused, citations")
    if answer["record_id"] != request.get("record_id"):
        raise ValueError("Answer record_id does not match the input record")
    if not isinstance(answer["text"], str) or not answer["text"].strip():
        raise ValueError("Answer text must be a non-empty string")
    if type(answer["refused"]) is not bool:
        raise ValueError("Answer refused must be a boolean")
    citations = answer["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("Answer must contain at least one citation")
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"article_id", "section_id"}:
            raise ValueError("Citation must contain exactly article_id and section_id")
        article_id, section_id = citation["article_id"], citation["section_id"]
        if not isinstance(article_id, str) or not isinstance(section_id, str):
            raise ValueError("Citation identifiers must be strings")
        pair = (article_id, section_id)
        if pair not in sections:
            raise ValueError("Citation does not exist in the input dataset")
        if pair in seen:
            raise ValueError("Citation is duplicated")
        seen.add(pair)
    return answer


def _validate_run_result(dataset: dict[str, Any], result: Any) -> dict[str, Any]:
    """Reject malformed batch output before it reaches the cache or HTTP body."""
    if not isinstance(result, dict) or set(result) != {"requests"}:
        raise ValueError("Run result must contain exactly requests")
    answers = result["requests"]
    requests = dataset.get("requests") or []
    if not isinstance(answers, list) or len(answers) != len(requests):
        raise ValueError("Run result must answer every input record exactly once")
    sections = _section_pairs(dataset)
    for answer, request in zip(answers, requests):
        _validate_generated_answer(answer, request, sections)
    return result


def _note_generation_usage(result: dict[str, Any], context: RunContext) -> None:
    attempts = result.get("attempts") if isinstance(result, dict) else None
    if not isinstance(attempts, list):
        return
    context.llm_calls += len(attempts)
    for attempt in attempts:
        usage = attempt.get("usage") if isinstance(attempt, dict) else None
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if type(tokens) is int and tokens >= 0:
            context.llm_tokens += tokens


def _answer_with_retry(
    agent: RAGAgent,
    request: dict[str, Any],
    deadline: float,
    context: RunContext,
    shared_evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if context.skip_llm:
        raise RuntimeError("LLM skipped")
    attempts = len(context.retry_delays) + 1
    for attempt in range(attempts):
        if time.monotonic() >= deadline:
            raise TimeoutError("run deadline exceeded")
        try:
            return agent.answer(
                request["utterance"]["text"],
                request["expert_id"],
                user_id=request["user_id"],
                record_id=request["record_id"],
                at=request["utterance"]["at"],
                deadline=deadline,
                shared_evidence=shared_evidence,
            )
        except (RuntimeError, TimeoutError) as exc:
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            delay = getattr(exc, "retry_after", None)
            if delay is None:
                delay = context.retry_delays[attempt]
            # Honour a server-provided Retry-After without letting one request
            # hold the whole run indefinitely.
            delay = min(max(float(delay), 0.0), 120.0)
            if time.monotonic() + delay >= deadline:
                raise
            context.retries += 1
            logger.info("transient model failure; retry %d in %.1fs", attempt + 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _answer_requests(
    agent: RAGAgent,
    dataset: dict[str, Any],
    deadline: float,
    context: RunContext,
) -> dict[str, list[dict[str, Any]]]:
    requests = dataset.get("requests") or []
    record_ids = [request["record_id"] for request in requests]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("dataset requests contain duplicate record IDs")
    sections = _section_pairs(dataset)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in requests:
        group_id = request.get("group_id")
        key = ("group", group_id) if group_id is not None else ("record", request["record_id"])
        grouped.setdefault(key, []).append(request)

    answers: dict[str, dict[str, Any]] = {}
    # Process one group at a time: evidence for all groups is not retained.
    for members in grouped.values():
        failed = False
        unsupported = False
        retrieval_started = time.monotonic()
        try:
            evidence = agent.prepare_group(members, deadline=deadline)
        except Exception:
            failed = True
            logger.warning("group retrieval failed")
            evidence = []
        if not evidence:
            failed = True
            evidence = _fallback_evidence(dataset, members, agent, deadline)
        if not evidence:
            # The schema requires a citation, so use an explicitly counted
            # last resort rather than returning invalid JSON or the first
            # section silently. Do not send an unrelated section to the model.
            unsupported = True
            evidence = _last_resort_evidence(dataset)
        context.retrieval_seconds += time.monotonic() - retrieval_started
        if failed:
            context.retrieval_failures += 1
        if not evidence:
            raise RuntimeError("Dataset has no usable article sections for a citation")

        ordered = sorted(members, key=lambda item: EXPERT_ORDER.get(item["expert_id"], 99))
        for request in ordered:
            record_id = request["record_id"]
            if not unsupported and not context.skip_llm and time.monotonic() < deadline:
                # A transient outage gets a bounded pause and a later probe,
                # never a permanent skip for all remaining records.
                wait = min(
                    max(0.0, context.cooldown_until - time.monotonic()),
                    max(0.0, deadline - time.monotonic()),
                    TRANSIENT_COOLDOWN_MAX_SECONDS,
                )
                if wait > 0:
                    time.sleep(wait)
                if time.monotonic() < deadline:
                    try:
                        generated = _answer_with_retry(
                            agent, request, deadline, context, evidence
                        )
                        _note_generation_usage(generated, context)
                        timing = generated.get("timing", {}) if isinstance(generated, dict) else {}
                        context.generation_seconds += float(timing.get("total_seconds", 0) or 0)
                        # The backend boundary, not the model adapter, decides
                        # whether an answer may enter the response or cache.
                        answer = _validate_generated_answer(
                            generated.get("answer"), request, sections
                        )
                        context.generated += 1
                        context.consecutive_transient = 0
                        context.cooldown_until = 0.0
                        answers[record_id] = answer
                        continue
                    except Exception as exc:
                        logger.warning("record failed: %s", type(exc).__name__)
                        _note_failure(exc, context)
            answers[record_id] = _fallback_answer(
                request, evidence, unsupported=unsupported, topical=not unsupported,
            )
            context.fallbacks += 1
            if unsupported:
                context.unsupported_fallbacks += 1
    return _validate_run_result(
        dataset, {"requests": [answers[record_id] for record_id in record_ids]}
    )


def _chat_answer_text(text: str, citations: list[dict[str, str]]) -> str:
    """Remove validated citation markers from prose shown and read aloud in chat."""
    pairs = {(item["article_id"], item["section_id"]) for item in citations}

    def hide_marker(match: re.Match[str]) -> str:
        return " " if match.groups() in pairs else match[0]

    without_markers = re.sub(r"\[([\w.:-]+)\s*/\s*([\w.:-]+)\]", hide_marker, text)
    lines = []
    for line in without_markers.split("\n"):
        line = re.sub(r"[ \t]+(?=[,.;:!?…])", "", line)
        lines.append(re.sub(r"[ \t]{2,}", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _validate_chat_citations(citations: Any) -> list[dict[str, str]]:
    if not isinstance(citations, list) or not citations:
        raise ValueError("Chat answer must contain at least one citation")
    validated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"article_id", "section_id"}:
            raise ValueError("Invalid chat citation structure")
        article_id, section_id = citation["article_id"], citation["section_id"]
        if not isinstance(article_id, str) or not isinstance(section_id, str):
            raise ValueError("Chat citation identifiers must be strings")
        pair = (article_id, section_id)
        if pair in seen:
            raise ValueError("Chat citation is duplicated")
        seen.add(pair)
        validated.append({"article_id": article_id, "section_id": section_id})
    return validated


def _enrich_citations(runtime: Runtime, citations: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build the complete source-card payload before session history changes."""
    enriched = []
    for citation in _validate_chat_citations(citations):
        source = runtime.sections.get((citation["article_id"], citation["section_id"]))
        if source is None:
            # A citation that has no article metadata cannot produce a complete
            # response. Fail before committing the turn to conversation history.
            raise ValueError("Chat citation is outside the active dataset")
        article, section = source
        enriched.append({
            "article_id": citation["article_id"],
            "section_id": citation["section_id"],
            "article_title": article["title"],
            "section_title": section["title"],
            "year": article["year"],
            "dataset_generation": runtime.generation,
        })
    return enriched


# ---------------------------------------------------------------------------
# HTTP models and routes
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    case: int = Field(ge=1, le=4)
    dataset_code: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    # Optional client-generated idempotency key. Repeating a lost turn with the
    # same ID replays the stored response instead of paying for it twice.
    turn_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
    )


def _run_reserve_calls(runtime: Runtime) -> int:
    """Worst-case calls for one run: one answer plus one repair per request."""
    requests = len(runtime.dataset.get("requests") or [])
    return min(LLM_MAX_CALLS, max(RUN_RESERVED_CALLS, 2 * requests))


def _chat_allowance(runtime: Runtime) -> tuple[int, int]:
    """Hard ceilings for chat calls and tokens.

    The official-run reserve is retained even after a completed run: a retry
    must not find the budget consumed by demo traffic.
    """
    return (
        max(0, LLM_MAX_CALLS - _run_reserve_calls(runtime)),
        max(0, LLM_TOKEN_BUDGET - RUN_RESERVED_TOKENS),
    )


def _chat_budget_available(runtime: Runtime) -> bool:
    calls = getattr(runtime.client, "calls", 0)
    tokens = getattr(runtime.client, "total_tokens", 0)
    call_ceiling, token_ceiling = _chat_allowance(runtime)
    return calls < call_ceiling and tokens < token_ceiling


class _ChatBudgetClient:
    """Inject the per-call chat ceilings into an otherwise shared client."""

    def __init__(self, client: Any, call_ceiling: int, token_ceiling: int) -> None:
        self.client = client
        self.call_ceiling = call_ceiling
        self.token_ceiling = token_ceiling

    def complete(self, messages, *, deadline=None, max_tokens=1800):
        return self.client.complete(
            messages,
            deadline=deadline,
            max_tokens=max_tokens,
            call_ceiling=self.call_ceiling,
            token_ceiling=self.token_ceiling,
        )


def _chat_global_rate_allowed(runtime: Runtime) -> bool:
    with runtime.sessions_lock:
        now = time.monotonic()
        cutoff = now - CHAT_RATE_WINDOW
        runtime.chat_global_times[:] = [
            stamp for stamp in runtime.chat_global_times if stamp > cutoff
        ]
        if len(runtime.chat_global_times) >= CHAT_GLOBAL_RATE_LIMIT:
            return False
        runtime.chat_global_times.append(now)
        return True


def _chat_session_rate_allowed(runtime: Runtime, session_id: str) -> bool:
    with runtime.sessions_lock:
        now = time.monotonic()
        cutoff = now - CHAT_RATE_WINDOW
        times = [stamp for stamp in runtime.chat_times.get(session_id, []) if stamp > cutoff]
        if len(times) >= CHAT_RATE_LIMIT:
            return False
        times.append(now)
        runtime.chat_times[session_id] = times
        return True


def _session_issue_allowed(runtime: Runtime) -> bool:
    """Session issuance has its own bucket; it never consumes chat capacity."""
    with runtime.sessions_lock:
        now = time.monotonic()
        cutoff = now - CHAT_RATE_WINDOW
        runtime.session_issue_times[:] = [
            stamp for stamp in runtime.session_issue_times if stamp > cutoff
        ]
        if len(runtime.session_issue_times) >= CHAT_SESSION_ISSUE_LIMIT:
            return False
        runtime.session_issue_times.append(now)
        return True


def _drop_chat_session(runtime: Runtime, session_id: str) -> None:
    with runtime.sessions_lock:
        runtime.sessions.pop(session_id, None)
        runtime.session_users.pop(session_id, None)
        runtime.session_touched.pop(session_id, None)
        runtime.chat_times.pop(session_id, None)
        runtime.busy_sessions.discard(session_id)
        for token_hash, value in list(runtime.session_tokens.items()):
            if value == session_id:
                runtime.session_tokens.pop(token_hash, None)
        for key in [key for key in runtime.completed_turns if key[0] == session_id]:
            runtime.completed_turns.pop(key, None)
        for key in [key for key in runtime.inflight_turns if key[0] == session_id]:
            runtime.inflight_turns.pop(key, None)


def _prune_chat_sessions(runtime: Runtime, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    with runtime.sessions_lock:
        stale = [
            value for value, touched in runtime.session_touched.items()
            if now - touched > CHAT_SESSION_TTL_SECONDS
        ]
        for value in stale:
            _drop_chat_session(runtime, value)


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _create_chat_session(runtime: Runtime) -> tuple[str, str]:
    """Issue a browser credential; the client never selects a user or dataset."""
    with runtime.sessions_lock:
        _prune_chat_sessions(runtime)
        if len(runtime.sessions) >= CHAT_MAX_SESSIONS:
            now = time.monotonic()
            evictable = [
                value for value, touched in runtime.session_touched.items()
                if now - touched > CHAT_SESSION_EVICT_GRACE_SECONDS
            ]
            if not evictable:
                # Never destroy a browser that was active moments ago just to
                # admit one more visitor.
                raise SessionCapacityError("No idle browser session can be replaced")
            oldest = min(evictable, key=lambda value: runtime.session_touched.get(value, 0.0))
            _drop_chat_session(runtime, oldest)
        session_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        runtime.sessions[session_id] = []
        runtime.session_users[session_id] = runtime.demo_user_id
        runtime.session_touched[session_id] = time.monotonic()
        runtime.session_tokens[_token_hash(token)] = session_id
        return session_id, token


def _resolve_browser_session(runtime: Runtime, authorization: str | None) -> tuple[str, str]:
    """Return the server-owned (session_id, user_id) for a browser token."""
    if (not isinstance(authorization, str) or not authorization.startswith("Bearer ")
            or len(authorization) <= 7):
        raise HTTPException(status_code=401, detail="Browser session token required")
    token = authorization[7:].strip()
    with runtime.sessions_lock:
        _prune_chat_sessions(runtime)
        session_id = runtime.session_tokens.get(_token_hash(token))
        if session_id is None or session_id not in runtime.sessions:
            raise HTTPException(status_code=401, detail="Unknown or expired browser session")
        owner = runtime.session_users.get(session_id)
        if not owner:
            _drop_chat_session(runtime, session_id)
            raise HTTPException(status_code=401, detail="Unknown or expired browser session")
        runtime.session_touched[session_id] = time.monotonic()
        return session_id, owner


def _trim_chat_session(session: list[dict[str, Any]]) -> None:
    while len(session) > CHAT_MAX_MESSAGES:
        session.pop(0)
    while session and sum(len(message.get("text", "")) for message in session) > CHAT_MAX_CHARS:
        session.pop(0)


@app.get("/")
def root():
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"team": TEAM, "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"team": TEAM, "status": "ok"}


@app.post("/session")
def create_session() -> dict[str, Any]:
    """Issue a browser credential bound to the active dataset and demo user."""
    runtime = _get_runtime()
    if not runtime.demo_user_id:
        raise HTTPException(status_code=503, detail="Dataset has no demo user")
    if not _session_issue_allowed(runtime):
        raise HTTPException(
            status_code=429,
            detail="Too many session requests; try again later",
            headers={"Retry-After": str(CHAT_RATE_WINDOW)},
        )
    try:
        session_id, token = _create_chat_session(runtime)
    except SessionCapacityError:
        raise HTTPException(
            status_code=503,
            detail="Too many active browser sessions; try again later",
            headers={"Retry-After": "30"},
        ) from None
    logger.info(
        "session issued: rid=%s active_sessions=%d generation=%s",
        _request_id(), len(runtime.sessions), runtime.generation,
    )
    # expires_in is an idle timeout: an authenticated chat refreshes it.
    return {
        "session_id": session_id,
        "token": token,
        "user_id": runtime.session_users[session_id],
        "expires_in": CHAT_SESSION_TTL_SECONDS,
    }


def _client_calls(client: Any) -> int:
    return int(getattr(client, "calls", 0) or 0)


def _client_tokens(client: Any) -> int:
    return int(getattr(client, "total_tokens", 0) or 0)


@app.post("/run")
def run(request: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    request_id = _request_id()
    if request.case != CASE_ID:
        raise HTTPException(status_code=400, detail="This backend implements case 4 only")
    if (not isinstance(authorization, str) or not authorization.startswith("Bearer ")
            or len(authorization) <= 7):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required")

    started = time.monotonic()
    deadline = started + RUN_BUDGET_SECONDS
    generation_deadline = deadline - RUN_SERIALIZATION_RESERVE_SECONDS
    # Only official runs serialize with each other. Chat, /health, /experts and
    # /source use the active runtime and are never blocked by this lock.
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="An official run is already in progress",
            headers={"Retry-After": "30"},
        )
    stage = "fetch"
    candidate: Runtime | None = None
    try:
        fetch_started = time.monotonic()
        try:
            dataset = _fetch_dataset(
                request.dataset_code,
                authorization,
                timeout=min(120.0, max(0.0, generation_deadline - fetch_started)),
            )
            _validate_dataset(dataset)
        except DatasetAccessError as exc:
            logger.warning(
                "run rejected: rid=%s stage=%s access=HTTP %d elapsed=%.1fs",
                request_id, stage, exc.status_code, time.monotonic() - started,
            )
            raise HTTPException(status_code=exc.status_code, detail="Dataset access rejected") from None
        except DatasetContractError as exc:
            logger.error(
                "run rejected: rid=%s stage=validate field=%s elapsed=%.1fs",
                request_id, exc.field, time.monotonic() - started,
            )
            raise HTTPException(status_code=502, detail="Dataset contract invalid") from None
        except Exception as exc:
            logger.error(
                "run failed: rid=%s stage=%s elapsed=%.1fs error=%s",
                request_id, stage, time.monotonic() - started, type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail="Dataset fetch failed") from None
        fetch_seconds = time.monotonic() - fetch_started

        try:
            cache_key: str | None = _run_cache_key(dataset)
        except Exception:
            cache_key = None

        # Reusing an identical active runtime preserves browser sessions and
        # avoids rebuilding an index for a repeated/cached dataset code.
        active = _active_runtime()
        runtime = active if cache_key is not None and active is not None \
            and active.dataset_key == cache_key else None
        build_seconds = 0.0
        if runtime is None:
            stage = "build"
            build_started = time.monotonic()
            try:
                candidate = _build_runtime(dataset, generation_deadline, cache_key)
                if cache_key is not None:
                    candidate.set_dataset_key(cache_key)
            except Exception as exc:
                logger.error(
                    "run failed: rid=%s stage=%s elapsed=%.1fs error=%s",
                    request_id, stage, time.monotonic() - started, type(exc).__name__,
                )
                raise HTTPException(status_code=500, detail="Runtime initialization failed") from None
            runtime = candidate
            build_seconds = time.monotonic() - build_started

        if cache_key is not None:
            cached = _cache_lookup(cache_key)
            if cached is not None:
                if candidate is not None:
                    _publish_runtime(candidate)
                    _prune_indexes(candidate.index_path)
                logger.info(
                    "run cache hit: rid=%s requests=%d reused=%s fetch=%.1fs "
                    "build=%.1fs total=%.1fs",
                    request_id, len(cached["requests"]), candidate is None,
                    fetch_seconds, build_seconds, time.monotonic() - started,
                )
                return JSONResponse(
                    content=cached, headers={"Cache-Control": "no-store"}
                )

        stage = "generate"
        context = RunContext()
        calls_before = _client_calls(runtime.client)
        tokens_before = _client_tokens(runtime.client)
        try:
            result = _answer_requests(
                runtime.agent, runtime.dataset, generation_deadline, context
            )
            _validate_run_result(runtime.dataset, result)
        except Exception as exc:
            if candidate is not None:
                _discard_candidate(candidate)
            logger.error(
                "run failed: rid=%s stage=%s elapsed=%.1fs calls=%d tokens=%d error=%s",
                request_id, stage, time.monotonic() - started,
                max(0, _client_calls(runtime.client) - calls_before),
                max(0, _client_tokens(runtime.client) - tokens_before),
                type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="Run failed") from None

        calls_used = max(0, _client_calls(runtime.client) - calls_before)
        tokens_used = max(0, _client_tokens(runtime.client) - tokens_before)
        clean = (
            cache_key is not None
            and context.fallbacks == 0
            and context.retrieval_failures == 0
            and context.unsupported_fallbacks == 0
            and not context.skip_llm
        )
        if clean:
            try:
                _cache_store(cache_key, result)
            except Exception as exc:
                logger.warning(
                    "run cache store skipped: rid=%s error=%s",
                    request_id, type(exc).__name__,
                )
        stage = "publish"
        if candidate is not None:
            _publish_runtime(candidate)
            _prune_indexes(candidate.index_path)
        logger.info(
            "run completed: rid=%s requests=%d generated=%d fallbacks=%d unsupported=%d "
            "retrieval_failures=%d validation_failures=%d transient=%d stopped=%s "
            "retries=%d run_calls=%d run_tokens=%d calls=%d tokens=%d reused=%s "
            "fetch=%.1fs build=%.1fs retrieval=%.1fs generation=%.1fs total=%.1fs",
            request_id, len(result["requests"]), context.generated, context.fallbacks,
            context.unsupported_fallbacks, context.retrieval_failures,
            context.validation_failures, context.transient_failures, context.skip_llm,
            context.retries, context.llm_calls, context.llm_tokens,
            calls_used, tokens_used, candidate is None, fetch_seconds, build_seconds,
            context.retrieval_seconds, context.generation_seconds,
            time.monotonic() - started,
        )
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})
    finally:
        _run_lock.release()


@app.get("/experts")
def experts() -> dict[str, Any]:
    runtime = _get_runtime()
    return {
        "experts": [
            {
                "expert_id": expert["expert_id"],
                "role": expert["role"],
                "speech_style": expert["speech_style"],
            }
            for expert in runtime.dataset["experts"]
        ],
        "dataset_generation": runtime.generation,
    }


def _turn_fingerprint(expert_id: str, message: str) -> str:
    return sha256(
        json.dumps(
            {"expert_id": expert_id, "message": message},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@app.post("/chat")
def chat(request: ChatRequest,
         authorization: str | None = Header(default=None)) -> dict[str, Any]:
    request_id = _request_id()
    started = time.monotonic()
    deadline = started + CHAT_BUDGET_SECONDS
    runtime = _get_runtime()
    # Browser access is authenticated with a server-issued token. The user
    # is derived from the active dataset; a client can never select one.
    session_id, user_id = _resolve_browser_session(runtime, authorization)
    if request.session_id is not None and request.session_id != session_id:
        raise HTTPException(status_code=403, detail="Session belongs to another browser")
    if not user_id:
        raise HTTPException(status_code=400, detail="Dataset has no user")
    if request.expert_id not in {item["expert_id"] for item in runtime.dataset["experts"]}:
        raise HTTPException(status_code=400, detail="Unknown expert_id")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must contain text")

    turn_id = request.turn_id or ("srv_" + uuid.uuid4().hex)
    fingerprint = _turn_fingerprint(request.expert_id, request.message)
    turn_key = (session_id, turn_id)
    with runtime.sessions_lock:
        completed = runtime.completed_turns.get(turn_key)
        if completed is not None:
            stored_fingerprint, stored_response = completed
            if stored_fingerprint != fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="turn_id was already used with different content",
                )
            runtime.completed_turns.move_to_end(turn_key)
            return copy.deepcopy(stored_response)
        if turn_key in runtime.inflight_turns:
            raise HTTPException(status_code=409, detail="This turn is already being processed")
        if session_id in runtime.busy_sessions:
            raise HTTPException(
                status_code=409,
                detail="Another turn is already being processed for this session",
            )
        if not _chat_budget_available(runtime):
            raise HTTPException(
                status_code=503,
                detail="LLM quota is reserved for the official /run",
            )
        if not _chat_global_rate_allowed(runtime):
            raise HTTPException(
                status_code=429,
                detail="Too many chat requests; try again later",
                headers={"Retry-After": str(CHAT_RATE_WINDOW)},
            )
        if not _chat_session_rate_allowed(runtime, session_id):
            raise HTTPException(
                status_code=429,
                detail="Too many chat requests; try again later",
                headers={"Retry-After": str(CHAT_RATE_WINDOW)},
            )
        session = runtime.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=401, detail="Unknown or expired browser session")
        history = list(session)
        runtime.inflight_turns[turn_key] = fingerprint
        runtime.busy_sessions.add(session_id)

    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    calls_before = _client_calls(runtime.client)
    tokens_before = _client_tokens(runtime.client)
    stage = "generate"
    try:
        try:
            # Only this browser session's own messages may reach the model. The
            # dataset's stored chat_history is intentionally not used here.
            call_ceiling, token_ceiling = _chat_allowance(runtime)
            client = _ChatBudgetClient(runtime.client, call_ceiling, token_ceiling)
            agent = RAGAgent(
                runtime.index_path,
                runtime.dataset["experts"],
                client,
                history=history,
            )
            result = agent.answer(
                request.message,
                request.expert_id,
                user_id=user_id,
                record_id="chat_" + uuid.uuid4().hex,
                at=at,
                deadline=deadline,
            )
            answer = result.get("answer") if isinstance(result, dict) else None
            if (not isinstance(answer, dict)
                    or set(answer) != {"record_id", "text", "refused", "citations"}):
                raise ValueError("Chat dependency returned an invalid answer object")
            if not isinstance(answer["text"], str) or not answer["text"].strip():
                raise ValueError("Chat answer text must be a non-empty string")
            if type(answer["refused"]) is not bool:
                raise ValueError("Chat answer refused must be a boolean")
            citations = _validate_chat_citations(answer["citations"])
            text = _chat_answer_text(answer["text"], citations)
            if not text:
                raise ValueError("Chat answer consists only of citations")
            # Build and validate the complete response before touching history.
            enriched = _enrich_citations(runtime, citations)
            response = {
                "session_id": session_id,
                "expert_id": request.expert_id,
                "text": text,
                "refused": answer["refused"],
                "citations": enriched,
                "turn_id": turn_id,
                "dataset_generation": runtime.generation,
            }
        except Exception as exc:
            logger.warning(
                "chat failed: rid=%s stage=%s elapsed=%.1fs calls=%d tokens=%d error=%s",
                request_id, stage, time.monotonic() - started,
                max(0, _client_calls(runtime.client) - calls_before),
                max(0, _client_tokens(runtime.client) - tokens_before),
                type(exc).__name__,
            )
            detail = (
                "LLM quota is reserved for the official /run"
                if _budget_exhausted(exc)
                else "Эксперт временно недоступен. Попробуйте ещё раз."
            )
            raise HTTPException(status_code=503, detail=detail) from None

        answered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stage = "commit"
        with runtime.sessions_lock:
            current = runtime.sessions.get(session_id)
            if current is not session:
                logger.warning(
                    "chat failed: rid=%s stage=%s elapsed=%.1fs error=SessionChanged",
                    request_id, stage, time.monotonic() - started,
                )
                raise HTTPException(status_code=409, detail="Session changed; retry the turn")
            current.extend([
                {
                    "message_id": "chat_" + uuid.uuid4().hex,
                    "user_id": user_id,
                    "role": "user",
                    "at": at,
                    "text": request.message,
                },
                {
                    "message_id": "chat_" + uuid.uuid4().hex,
                    "user_id": user_id,
                    "role": "agent",
                    "at": answered_at,
                    "text": text,
                },
            ])
            _trim_chat_session(current)
            runtime.session_touched[session_id] = time.monotonic()
            runtime.completed_turns[turn_key] = (fingerprint, response)
            runtime.completed_turns.move_to_end(turn_key)
            while len(runtime.completed_turns) > CHAT_TURN_CACHE_MAX:
                runtime.completed_turns.popitem(last=False)
            message_count = len(current)

        logger.info(
            "chat completed: rid=%s calls=%d tokens=%d total=%.1fs messages=%d",
            request_id,
            max(0, _client_calls(runtime.client) - calls_before),
            max(0, _client_tokens(runtime.client) - tokens_before),
            time.monotonic() - started,
            message_count,
        )
        return response
    finally:
        with runtime.sessions_lock:
            runtime.inflight_turns.pop(turn_key, None)
            runtime.busy_sessions.discard(session_id)


@app.get("/source/{article_id}/{section_id}")
def source(article_id: str, section_id: str,
           generation: str | None = None) -> dict[str, Any]:
    runtime = _get_runtime()
    if generation is not None and generation != runtime.generation:
        raise HTTPException(
            status_code=409,
            detail="Source belongs to a previous dataset generation; reload the dataset",
        )
    entry = runtime.sections.get((article_id, section_id))
    if entry is None:
        raise HTTPException(status_code=404, detail="Source not found")
    article, section = entry
    return {
        "article_id": article["article_id"],
        "article_title": article["title"],
        "section_id": section["section_id"],
        "section_title": section["title"],
        "text": section["text"],
        "authors": article["authors"],
        "year": article["year"],
        "doi": article["doi"],
        "license": article["license"],
        "dataset_generation": runtime.generation,
    }


if (DIST_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")
