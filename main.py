"""Team19 backend for the Sain x School 21 hackathon, case 4.

Contract (team19/README.md, sample-app/main.py):
- ``requirements.txt`` and ``start.sh`` live in the repository root;
- Python 3.11, the server listens on ``0.0.0.0:$PORT``;
- ``GET /`` answers within 60 seconds;
- ``POST /run`` receives ``{"case": 4, "dataset_code": "..."}``, fetches the
  dataset with the incoming Bearer token, answers every request, and returns
  the complete Case4Answers JSON in the same HTTP response.

Secrets, dataset URLs and dataset content are never written to logs.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_agent import ChatClient, RAGAgent
from rag_preprocessing import build_index
from rag_preprocessing.embeddings import NoRedirect

TEAM = os.environ.get("TEAM", "team19")
CASE_ID = 4
RUN_BUDGET_SECONDS = 42 * 60  # the stand waits 45 minutes
CHAT_BUDGET_SECONDS = 5 * 60
INDEX_PATH = Path(os.environ.get("CASE4_INDEX_PATH", "/tmp/case4-index.sqlite3"))
DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CALLS = 150
LLM_TOKEN_BUDGET = 260_000
LLM_MIN_INTERVAL_SECONDS = 2.1
RUN_CACHE_VERSION = "1"
RUN_CACHE_MAX_ENTRIES = 4
MAX_DATASET_BYTES = 32 * 1024 * 1024

logger = logging.getLogger("team19.case4")

app = FastAPI(title=f"hackathon {TEAM} / case {CASE_ID}")


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class Runtime:
    """Dataset, LLM client and the single serialized RAG agent."""

    def __init__(self, dataset: dict[str, Any], client: ChatClient, agent: RAGAgent) -> None:
        self.dataset = dataset
        self.client = client
        self.agent = agent
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.users = {user["user_id"] for user in dataset.get("users", [])}
        self.sections: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        for article in dataset.get("articles", []):
            for section in article.get("sections", []):
                self.sections[(article["article_id"], section["section_id"])] = (article, section)


_runtime: Runtime | None = None
_state_lock = threading.Lock()
_init_lock = threading.Lock()
# The core agent is documented as single-run; this lock also serializes /run
# and /chat while keeping GET /health responsive.
_agent_lock = threading.RLock()

# One client per process: call and token counters must survive repeated /run
# attempts, otherwise each attempt could spend the whole budget again.
_llm_client: ChatClient | None = None
_llm_client_lock = threading.Lock()

# Only complete, fully generated runs are cached; a content digest (never the
# dataset code or the reference) keys the result.
_run_cache: dict[str, dict[str, Any]] = {}
_run_cache_lock = threading.Lock()


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


def _build_runtime(dataset: dict[str, Any], deadline: float) -> Runtime:
    requests = dataset.get("requests")
    if not isinstance(requests, list):
        raise ValueError("dataset.requests must be a list")
    build_index(dataset, INDEX_PATH, deadline=deadline)
    client = _get_llm_client()
    agent = RAGAgent(
        INDEX_PATH,
        dataset["experts"],
        client,
        history=dataset.get("chat_history", []),
    )
    return Runtime(dataset, client, agent)


def _get_runtime() -> Runtime:
    """Return the loaded runtime; only local development lazy-loads dev data."""
    global _runtime
    with _state_lock:
        runtime = _runtime
    if runtime is not None:
        return runtime

    dev_code = os.environ.get("DEV_DATASET_CODE", "").strip()
    team_token = os.environ.get("HACK_TEAM_TOKEN", "").strip()
    if not dev_code or not team_token:
        raise HTTPException(status_code=503, detail="Dataset is not loaded yet")

    # Local frontend development only. On the stand DEV_DATASET_CODE is unset
    # and /run provides the online/final dataset.
    with _agent_lock:
        with _state_lock:
            runtime = _runtime
        if runtime is not None:
            return runtime
        with _init_lock:
            with _state_lock:
                runtime = _runtime
            if runtime is not None:
                return runtime
            try:
                dataset = _fetch_dataset(dev_code, "Bearer " + team_token)
                runtime = _build_runtime(dataset, time.monotonic() + 300)
            except Exception as exc:
                logger.error("dev dataset initialization failed: %s", type(exc).__name__)
                raise HTTPException(status_code=503, detail="Dataset is not loaded yet") from None
            with _state_lock:
                _runtime = runtime
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


def _fetch_dataset(dataset_code: str, authorization: str, timeout: float = 120) -> dict[str, Any]:
    """Fetch the whole dataset once with the token received by /run."""
    url = _api_base() + f"/case/{CASE_ID}/getData?dataset_code=" + quote(dataset_code, safe="")
    request = Request(url, headers={"Authorization": authorization, "Accept": "application/json"})
    opener = build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_DATASET_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"dataset API HTTP {exc.code}") from None
    except URLError:
        raise RuntimeError("dataset API unavailable") from None
    if len(raw) > MAX_DATASET_BYTES:
        raise RuntimeError("dataset API response too large")
    payload = json.loads(raw)
    if isinstance(payload, dict) and isinstance(payload.get("dataset"), dict):
        payload = payload["dataset"]
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise RuntimeError("unexpected dataset response")
    return payload


def _run_cache_key(dataset: dict[str, Any]) -> str:
    """Content digest of the fields that affect answers; never the reference."""
    content = {
        name: dataset.get(name)
        for name in ("requests", "experts", "chat_history", "articles", "users")
    }
    canonical = json.dumps(
        {
            "version": RUN_CACHE_VERSION,
            "model": os.environ.get("LLM_MODEL", ""),
            "dataset": content,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _cache_lookup(cache_key: str) -> dict[str, Any] | None:
    with _run_cache_lock:
        return _run_cache.get(cache_key)


def _cache_store(cache_key: str, result: dict[str, Any]) -> None:
    with _run_cache_lock:
        if cache_key not in _run_cache and len(_run_cache) >= RUN_CACHE_MAX_ENTRIES:
            _run_cache.pop(next(iter(_run_cache)))
        _run_cache[cache_key] = result


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


class RunContext:
    def __init__(self, retry_delays: tuple[float, ...] = (5.0, 10.0)) -> None:
        self.skip_llm = False
        self.fallbacks = 0
        self.retry_delays = retry_delays


EXPERT_ORDER = {"clinician": 0, "biohacker": 1, "nutritionist": 2}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc)
    return "HTTP 429" in message or "HTTP 5" in message or "unavailable" in message


def _should_skip_remaining(exc: BaseException) -> bool:
    message = str(exc)
    return ("budget exhausted" in message or "HTTP 429" in message or "unavailable" in message)


def _first_section(dataset: dict[str, Any]) -> list[dict[str, str]]:
    for article in dataset.get("articles", []):
        for section in article.get("sections", []):
            return [{"article_id": article["article_id"], "section_id": section["section_id"]}]
    raise ValueError("Dataset has no article sections")


def _fallback_answer(
    dataset: dict[str, Any],
    request: dict[str, Any],
    preferred: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    citations = preferred or _first_section(dataset)
    refs = " ".join(f"[{item['article_id']} / {item['section_id']}]" for item in citations)
    return {
        "record_id": request["record_id"],
        "text": (
            "По этому вопросу у меня нет подтверждённого вывода в базе. "
            "Не стоит действовать наугад; обсудите это со специалистом. " + refs
        ),
        "refused": True,
        "citations": [dict(item) for item in citations],
    }


def _answer_with_retry(
    agent: RAGAgent,
    request: dict[str, Any],
    deadline: float,
    context: RunContext,
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
            )
        except (RuntimeError, TimeoutError) as exc:
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            delay = context.retry_delays[attempt]
            if time.monotonic() + delay >= deadline:
                raise
            logger.info("transient model failure; retry %d in %ss", attempt + 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _restrict_evidence_to_answer(
    agent: Any,
    request: dict[str, Any],
    answer: dict[str, Any],
    deadline: float,
) -> None:
    """Keep only the fixed citation sections for the next experts.

    Best-effort by design: any mismatch leaves the full evidence in place, so
    this optimization cannot break a run. A refusal keeps full evidence so the
    other experts can independently confirm the boundary.
    """
    if answer.get("refused"):
        return
    prepare = getattr(agent, "prepare", None)
    citation_cache = getattr(agent, "citation_cache", None)
    evidence_cache = getattr(agent, "evidence_cache", None)
    if (not callable(prepare) or not isinstance(citation_cache, dict)
            or not isinstance(evidence_cache, dict)):
        return
    try:
        prepared = prepare(
            request["utterance"]["text"],
            request["expert_id"],
            user_id=request["user_id"],
            record_id=request["record_id"],
            at=request["utterance"]["at"],
            deadline=deadline,
        )
        key = prepared["cache_key"]
        required = citation_cache.get(key)
        blocks = evidence_cache.get(key)
        if not required or not blocks:
            return
        filtered = [
            block for block in blocks
            if (block["article_id"], block["section_id"]) in required
        ]
        if filtered:
            evidence_cache[key] = filtered
    except Exception:
        return


def _answer_group(
    agent: RAGAgent,
    dataset: dict[str, Any],
    members: list[dict[str, Any]],
    deadline: float,
    context: RunContext,
) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    successful: list[dict[str, Any]] = []
    ordered = sorted(members, key=lambda item: EXPERT_ORDER.get(item["expert_id"], 99))
    for request in ordered:
        if context.skip_llm or time.monotonic() >= deadline:
            continue
        try:
            result = _answer_with_retry(agent, request, deadline, context)
            answers[request["record_id"]] = result["answer"]
            if not successful:
                _restrict_evidence_to_answer(agent, request, result["answer"], deadline)
            successful.append(result["answer"])
        except Exception as exc:
            logger.warning("record failed: %s", type(exc).__name__)
            if _should_skip_remaining(exc):
                context.skip_llm = True

    # The case requires identical citations for all three experts of a triple.
    preferred = successful[0]["citations"] if successful else None
    for request in members:
        if request["record_id"] not in answers:
            answers[request["record_id"]] = _fallback_answer(dataset, request, preferred)
            context.fallbacks += 1
    return answers


def _answer_requests(
    agent: RAGAgent,
    dataset: dict[str, Any],
    deadline: float,
    context: RunContext,
) -> dict[str, list[dict[str, Any]]]:
    requests = dataset.get("requests") or []
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    singles: list[dict[str, Any]] = []
    for request in requests:
        group_id = request.get("group_id")
        if group_id is None:
            singles.append(request)
        else:
            if group_id not in grouped:
                grouped[group_id] = []
                group_order.append(group_id)
            grouped[group_id].append(request)

    answers: dict[str, dict[str, Any]] = {}
    for group_id in group_order:
        answers.update(_answer_group(agent, dataset, grouped[group_id], deadline, context))
    for request in singles:
        if context.skip_llm or time.monotonic() >= deadline:
            answers[request["record_id"]] = _fallback_answer(dataset, request)
            context.fallbacks += 1
            continue
        try:
            answers[request["record_id"]] = _answer_with_retry(agent, request, deadline, context)["answer"]
        except Exception as exc:
            logger.warning("record failed: %s", type(exc).__name__)
            if _should_skip_remaining(exc):
                context.skip_llm = True
            answers[request["record_id"]] = _fallback_answer(dataset, request)
            context.fallbacks += 1
    return {"requests": [answers[request["record_id"]] for request in requests]}


def _enrich_citations(runtime: Runtime, citations: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched = []
    for citation in citations:
        item: dict[str, Any] = {
            "article_id": citation["article_id"],
            "section_id": citation["section_id"],
        }
        source = runtime.sections.get((citation["article_id"], citation["section_id"]))
        if source is not None:
            article, section = source
            item["article_title"] = article["title"]
            item["section_title"] = section["title"]
            item["year"] = article["year"]
        enriched.append(item)
    return enriched


# ---------------------------------------------------------------------------
# HTTP models and routes
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    case: int = Field(ge=1, le=4)
    dataset_code: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    expert_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)


@app.get("/")
def root():
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"team": TEAM, "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"team": TEAM, "status": "ok"}


@app.post("/run")
def run(request: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    global _runtime
    if request.case != CASE_ID:
        raise HTTPException(status_code=400, detail="This backend implements case 4 only")
    if not authorization or not authorization.startswith("Bearer ") or len(authorization) <= 7:
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required")

    started = time.monotonic()
    deadline = started + RUN_BUDGET_SECONDS
    with _agent_lock:
        try:
            dataset = _fetch_dataset(request.dataset_code, authorization)
        except Exception as exc:
            logger.error("dataset fetch failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Dataset fetch failed") from None
        try:
            cache_key: str | None = _run_cache_key(dataset)
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = _cache_lookup(cache_key)
            if cached is not None:
                logger.info(
                    "run cache hit: %d requests in %.1fs",
                    len(cached["requests"]),
                    time.monotonic() - started,
                )
                return JSONResponse(content=cached)
        try:
            runtime = _build_runtime(dataset, deadline)
        except Exception as exc:
            logger.error("runtime initialization failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Runtime initialization failed") from None
        with _state_lock:
            _runtime = runtime
        context = RunContext()
        try:
            result = _answer_requests(runtime.agent, dataset, deadline, context)
        except Exception as exc:
            logger.error("run failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Run failed") from None
        if cache_key is not None and context.fallbacks == 0 and not context.skip_llm:
            _cache_store(cache_key, result)
        logger.info(
            "run completed: %d requests in %.1fs",
            len(result["requests"]),
            time.monotonic() - started,
        )
        return JSONResponse(content=result)


@app.get("/experts")
def experts() -> dict[str, list[dict[str, str]]]:
    runtime = _get_runtime()
    return {
        "experts": [
            {
                "expert_id": expert["expert_id"],
                "role": expert["role"],
                "speech_style": expert["speech_style"],
            }
            for expert in runtime.dataset["experts"]
        ]
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    with _agent_lock:
        runtime = _get_runtime()
        if request.user_id is not None:
            user_id = request.user_id
        elif runtime.users:
            user_id = sorted(runtime.users)[0]
        else:
            raise HTTPException(status_code=400, detail="Dataset has no user")
        if user_id not in runtime.users:
            raise HTTPException(status_code=400, detail="Unknown user_id")
        if request.expert_id not in {item["expert_id"] for item in runtime.dataset["experts"]}:
            raise HTTPException(status_code=400, detail="Unknown expert_id")

        session_id = request.session_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id")
        session = runtime.sessions.setdefault(session_id, [])

        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        history = list(runtime.dataset.get("chat_history", [])) + session
        agent = RAGAgent(
            INDEX_PATH,
            runtime.dataset["experts"],
            runtime.client,
            history=history,
        )
        try:
            result = agent.answer(
                request.message,
                request.expert_id,
                user_id=user_id,
                record_id="chat_" + uuid.uuid4().hex,
                at=at,
                deadline=time.monotonic() + CHAT_BUDGET_SECONDS,
            )
        except Exception as exc:
            logger.warning("chat failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Эксперт временно недоступен. Попробуйте ещё раз.") from None

        answer = result["answer"]
        answered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session.extend([
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
                "text": answer["text"],
            },
        ])
        return {
            "session_id": session_id,
            "expert_id": request.expert_id,
            "text": answer["text"],
            "refused": answer["refused"],
            "citations": _enrich_citations(runtime, answer["citations"]),
        }


@app.get("/source/{article_id}/{section_id}")
def source(article_id: str, section_id: str) -> dict[str, Any]:
    runtime = _get_runtime()
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
    }


if (DIST_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")
