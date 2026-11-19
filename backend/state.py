"""Runtime registry, index lifecycle and the shared LLM client."""

from __future__ import annotations

from collections import OrderedDict
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from rag_agent import ChatClient, RAGAgent
from rag_preprocessing import build_index

from . import organizer
from .cache import _run_cache_key
from .config import (
    INDEX_MAX_RETAINED, INDEX_PATH, LLM_MAX_CALLS, LLM_MIN_INTERVAL_SECONDS,
    LLM_TIMEOUT_SECONDS, LLM_TOKEN_BUDGET,
)
from .log import logger


class Runtime:
    """One published dataset, its index and its process-local browser state."""

    def __init__(self, dataset: dict[str, Any], client: ChatClient, agent: RAGAgent,
                 index_path: str | Path | None = None,
                 dataset_key: str | None = None) -> None:
        self.dataset = dataset
        self.client = client
        self.agent = agent
        self.index_path = Path(index_path or getattr(agent, "index", None) or INDEX_PATH)
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

    # Local API development only. On the stand DEV_DATASET_CODE is unset
    # and /run provides the online/final dataset.
    with _init_lock:
        runtime = _active_runtime()
        if runtime is not None:
            return runtime
        try:
            dataset = organizer._fetch_dataset(dev_code, "Bearer " + team_token)
            organizer._validate_dataset(dataset)
            runtime = _build_runtime(dataset, time.monotonic() + 300)
        except Exception as exc:
            logger.error("dev dataset initialization failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Dataset is not loaded yet") from None
        _publish_runtime(runtime)
        return runtime
