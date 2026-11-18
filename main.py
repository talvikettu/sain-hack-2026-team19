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

from collections import OrderedDict
from contextlib import contextmanager
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

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rag_agent import ChatClient, RAGAgent
from rag_agent.client import BudgetError, ProxyProtocolError
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
CHAT_LOCK_WAIT_SECONDS = 10
# Reserve enough generation capacity for the single official /run before chat
# may use the shared proxy budget. One answer plus one repair per request is
# the documented worst case; the floor keeps small datasets from freeing too
# much of the process budget for pre-run chat.
RUN_RESERVED_CALLS = 105
RUN_RESERVED_TOKENS = 220_000
INDEX_PATH = Path(os.environ.get("CASE4_INDEX_PATH", "/tmp/case4-index.sqlite3"))
DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CALLS = 150
LLM_TOKEN_BUDGET = 260_000
LLM_MIN_INTERVAL_SECONDS = 2.1
# A transient proxy outage must not permanently disable generation. After a
# failed record, pause before the next probe so later records can recover.
TRANSIENT_COOLDOWN_BASE_SECONDS = 4.0
TRANSIENT_COOLDOWN_MAX_SECONDS = 30.0
RUN_CACHE_VERSION = "9"
RUN_CACHE_MAX_ENTRIES = 4
MAX_DATASET_BYTES = 32 * 1024 * 1024
DATASET_READ_CHUNK = 64 * 1024

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

app = FastAPI(title=f"hackathon {TEAM} / case {CASE_ID}")


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class Runtime:
    """Dataset, LLM client and the single serialized RAG agent."""

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
        users = [user["user_id"] for user in dataset.get("users", []) if user.get("user_id")]
        self.demo_user_id = users[0] if users else ""
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.session_users: dict[str, str] = {}
        self.session_touched: dict[str, float] = {}
        # Only a hash of a browser token is retained. The organizer and LLM
        # credentials never enter this map.
        self.session_tokens: dict[str, str] = {}
        self.issued_sessions: set[str] = set()
        self.chat_times: dict[str, list[float]] = {}
        self.chat_global_times: list[float] = []
        self.run_completed = False
        self.users = set(users)
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
# dataset code or the reference) keys the result. Least-recently-used entries
# are evicted first.
_run_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_run_cache_lock = threading.Lock()


@contextmanager
def _agent_lock_until(deadline: float, *, wait: float | None = None):
    """Bound lock waiting by the same deadline as the request."""
    timeout = max(0.0, deadline - time.monotonic())
    if wait is not None:
        timeout = min(timeout, max(0.0, wait))
    if not _agent_lock.acquire(timeout=timeout):
        raise HTTPException(status_code=503, detail="Server is busy; retry shortly")
    try:
        yield
    finally:
        _agent_lock.release()


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


def _prune_indexes(active: Path) -> None:
    """Keep only the index used by the active runtime; failures are harmless."""
    if not active.name.startswith(INDEX_PATH.stem + "-"):
        return
    try:
        for path in INDEX_PATH.parent.glob(f"{INDEX_PATH.stem}-*{INDEX_PATH.suffix}"):
            if path != active:
                path.unlink(missing_ok=True)
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
    build_index(dataset, index_path, deadline=deadline)
    client = _get_llm_client()
    agent = RAGAgent(
        index_path,
        dataset["experts"],
        client,
        history=dataset.get("chat_history", []),
    )
    return Runtime(dataset, client, agent, index_path=index_path, dataset_key=index_key)


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


def _read_limited(response, limit: int, deadline: float) -> bytes:
    """Read a bounded body without letting a slow server hold the request."""
    chunks, total = [], 0
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError("dataset API read timed out")
        chunk = response.read(min(DATASET_READ_CHUNK, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise RuntimeError("dataset API response too large")
        chunks.append(chunk)
    return b"".join(chunks)


class DatasetAccessError(RuntimeError):
    """The organizer API rejected the token for the requested dataset."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"dataset API HTTP {status_code}")
        self.status_code = status_code


def _fetch_dataset(dataset_code: str, authorization: str, timeout: float = 120) -> dict[str, Any]:
    """Fetch the whole dataset once with the token received by /run."""
    if timeout <= 0:
        raise TimeoutError("dataset API deadline exceeded")
    url = _api_base() + f"/case/{CASE_ID}/getData?dataset_code=" + quote(dataset_code, safe="")
    request = Request(url, headers={"Authorization": authorization, "Accept": "application/json"})
    opener = build_opener(NoRedirect())
    read_deadline = time.monotonic() + timeout
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = _read_limited(response, MAX_DATASET_BYTES, read_deadline)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise DatasetAccessError(exc.code) from None
        raise RuntimeError(f"dataset API HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("dataset API unavailable") from None
    payload = json.loads(raw)
    del raw
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
        result = _run_cache.get(cache_key)
        if result is not None:
            _run_cache.move_to_end(cache_key)
        return result


def _cache_store(cache_key: str, result: dict[str, Any]) -> None:
    with _run_cache_lock:
        if cache_key in _run_cache:
            _run_cache.move_to_end(cache_key)
        elif len(_run_cache) >= RUN_CACHE_MAX_ENTRIES:
            _run_cache.popitem(last=False)
        _run_cache[cache_key] = result


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
        self.retry_delays = retry_delays


EXPERT_ORDER = {"clinician": 0, "biohacker": 1, "nutritionist": 2}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc)
    return "HTTP 408" in message or "HTTP 429" in message or "HTTP 5" in message or "unavailable" in message


def _budget_exhausted(exc: BaseException) -> bool:
    if isinstance(exc, BudgetError):
        return True
    return "budget exhausted" in str(exc).casefold()


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
    message = str(exc)
    if (_budget_exhausted(exc) or _protocol_failure(exc)
            or "HTTP 401" in message or "HTTP 403" in message):
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
        title_terms = _query_terms(str(article.get("title", "")))
        for section in article.get("sections", []):
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in requests:
        group_id = request.get("group_id")
        key = ("group", group_id) if group_id is not None else ("record", request["record_id"])
        grouped.setdefault(key, []).append(request)

    # Keep retrieval available even if generation uses the remaining deadline.
    # A failure in one group must not abort the whole run: fall back to a local
    # keyword-selected section and keep answering every record.
    prepared = []
    for members in grouped.values():
        failed = False
        unsupported = False
        retrieval_started = time.monotonic()
        try:
            evidence = agent.prepare_group(members, deadline=deadline)
        except Exception as exc:
            failed = True
            logger.warning("group retrieval failed: %s", type(exc).__name__)
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
        prepared.append((members, evidence, unsupported))

    answers: dict[str, dict[str, Any]] = {}
    for members, evidence, unsupported in prepared:
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
                        result = _answer_with_retry(agent, request, deadline, context, evidence)
                        timing = result.get("timing", {}) if isinstance(result, dict) else {}
                        context.generation_seconds += float(timing.get("total_seconds", 0) or 0)
                        context.generated += 1
                        context.consecutive_transient = 0
                        context.cooldown_until = 0.0
                        answers[record_id] = result["answer"]
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
    return {"requests": [answers[request["record_id"]] for request in requests]}


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
    model_config = ConfigDict(extra="forbid")

    expert_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)


def _run_reserve_calls(runtime: Runtime) -> int:
    """Worst-case calls for one run: one answer plus one repair per request."""
    requests = len(runtime.dataset.get("requests") or [])
    return min(LLM_MAX_CALLS, max(RUN_RESERVED_CALLS, 2 * requests))


def _chat_budget_available(runtime: Runtime) -> bool:
    """Reserve most of the shared proxy budget for the official /run."""
    calls = getattr(runtime.client, "calls", 0)
    tokens = getattr(runtime.client, "total_tokens", 0)
    if calls >= LLM_MAX_CALLS or tokens >= LLM_TOKEN_BUDGET:
        return False
    if runtime.run_completed:
        return True
    return (
        calls < max(0, LLM_MAX_CALLS - _run_reserve_calls(runtime))
        and tokens < max(0, LLM_TOKEN_BUDGET - RUN_RESERVED_TOKENS)
    )


def _chat_global_rate_allowed(runtime: Runtime) -> bool:
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
    now = time.monotonic()
    cutoff = now - CHAT_RATE_WINDOW
    times = [stamp for stamp in runtime.chat_times.get(session_id, []) if stamp > cutoff]
    if len(times) >= CHAT_RATE_LIMIT:
        return False
    times.append(now)
    runtime.chat_times[session_id] = times
    return True


def _drop_chat_session(runtime: Runtime, session_id: str) -> None:
    runtime.sessions.pop(session_id, None)
    runtime.session_users.pop(session_id, None)
    runtime.session_touched.pop(session_id, None)
    runtime.chat_times.pop(session_id, None)
    runtime.issued_sessions.discard(session_id)
    for token_hash, value in list(runtime.session_tokens.items()):
        if value == session_id:
            runtime.session_tokens.pop(token_hash, None)


def _prune_chat_sessions(runtime: Runtime, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
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
    _prune_chat_sessions(runtime)
    if len(runtime.sessions) >= CHAT_MAX_SESSIONS:
        oldest = min(
            runtime.sessions,
            key=lambda value: runtime.session_touched.get(value, 0.0),
        )
        _drop_chat_session(runtime, oldest)
    session_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(32)
    runtime.sessions[session_id] = []
    runtime.session_users[session_id] = runtime.demo_user_id
    runtime.session_touched[session_id] = time.monotonic()
    runtime.session_tokens[_token_hash(token)] = session_id
    runtime.issued_sessions.add(session_id)
    return session_id, token


def _resolve_browser_session(runtime: Runtime, authorization: str | None) -> tuple[str, str]:
    """Return the server-owned (session_id, user_id) for a browser token."""
    if (not isinstance(authorization, str) or not authorization.startswith("Bearer ")
            or len(authorization) <= 7):
        raise HTTPException(status_code=401, detail="Browser session token required")
    token = authorization[7:].strip()
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
    deadline = time.monotonic() + CHAT_LOCK_WAIT_SECONDS
    with _agent_lock_until(deadline, wait=CHAT_LOCK_WAIT_SECONDS):
        runtime = _get_runtime()
        if not runtime.demo_user_id:
            raise HTTPException(status_code=503, detail="Dataset has no demo user")
        if not _chat_global_rate_allowed(runtime):
            raise HTTPException(
                status_code=429,
                detail="Too many session requests; try again later",
                headers={"Retry-After": str(CHAT_RATE_WINDOW)},
            )
        session_id, token = _create_chat_session(runtime)
        return {
            "session_id": session_id,
            "token": token,
            "user_id": runtime.session_users[session_id],
            "expires_in": CHAT_SESSION_TTL_SECONDS,
        }


@app.post("/run")
def run(request: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    global _runtime
    if request.case != CASE_ID:
        raise HTTPException(status_code=400, detail="This backend implements case 4 only")
    if (not isinstance(authorization, str) or not authorization.startswith("Bearer ")
            or len(authorization) <= 7):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required")

    started = time.monotonic()
    deadline = started + RUN_BUDGET_SECONDS
    # The agent lock, dataset download, index build and generation all share
    # the same 42-minute deadline.
    with _agent_lock_until(deadline):
        fetch_started = time.monotonic()
        try:
            dataset = _fetch_dataset(
                request.dataset_code,
                authorization,
                timeout=min(120.0, max(0.0, deadline - fetch_started)),
            )
        except DatasetAccessError as exc:
            logger.warning("dataset access rejected: HTTP %d", exc.status_code)
            raise HTTPException(status_code=exc.status_code, detail="Dataset access rejected") from None
        except Exception as exc:
            logger.error("dataset fetch failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Dataset fetch failed") from None
        fetch_seconds = time.monotonic() - fetch_started
        try:
            cache_key: str | None = _run_cache_key(dataset)
        except Exception:
            cache_key = None
        build_started = time.monotonic()
        try:
            runtime = _build_runtime(dataset, deadline, cache_key)
        except Exception as exc:
            logger.error("runtime initialization failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Runtime initialization failed") from None
        build_seconds = time.monotonic() - build_started
        # Activate the requested dataset/index before serving a cached answer so
        # /experts, /source and /chat always match the dataset just submitted.
        with _state_lock:
            _runtime = runtime
        _prune_indexes(runtime.index_path)
        if cache_key is not None:
            cached = _cache_lookup(cache_key)
            if cached is not None:
                runtime.run_completed = True
                logger.info(
                    "run cache hit: requests=%d fetch=%.1fs build=%.1fs total=%.1fs",
                    len(cached["requests"]), fetch_seconds, build_seconds,
                    time.monotonic() - started,
                )
                return JSONResponse(content=cached)
        context = RunContext()
        calls_before = getattr(runtime.client, "calls", 0)
        tokens_before = getattr(runtime.client, "total_tokens", 0)
        try:
            result = _answer_requests(runtime.agent, dataset, deadline, context)
        except Exception as exc:
            logger.error("run failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Run failed") from None
        runtime.run_completed = True
        calls_used = max(0, getattr(runtime.client, "calls", 0) - calls_before)
        tokens_used = max(0, getattr(runtime.client, "total_tokens", 0) - tokens_before)
        if (cache_key is not None and context.fallbacks == 0
                and context.retrieval_failures == 0
                and context.unsupported_fallbacks == 0
                and not context.skip_llm):
            _cache_store(cache_key, result)
        logger.info(
            "run completed: requests=%d generated=%d fallbacks=%d unsupported=%d "
            "retrieval_failures=%d validation_failures=%d transient=%d stopped=%s "
            "retries=%d calls=%d tokens=%d fetch=%.1fs build=%.1fs retrieval=%.1fs "
            "generation=%.1fs total=%.1fs",
            len(result["requests"]), context.generated, context.fallbacks,
            context.unsupported_fallbacks, context.retrieval_failures,
            context.validation_failures, context.transient_failures, context.skip_llm,
            context.retries, calls_used, tokens_used, fetch_seconds, build_seconds,
            context.retrieval_seconds, context.generation_seconds,
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
def chat(request: ChatRequest,
         authorization: str | None = Header(default=None)) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + CHAT_BUDGET_SECONDS
    with _agent_lock_until(deadline, wait=CHAT_LOCK_WAIT_SECONDS):
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
        session = runtime.sessions[session_id]
        if not _chat_session_rate_allowed(runtime, session_id):
            raise HTTPException(
                status_code=429,
                detail="Too many chat requests; try again later",
                headers={"Retry-After": str(CHAT_RATE_WINDOW)},
            )

        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Only this browser session's own messages may reach the model. The
        # dataset's stored chat_history is intentionally not used here.
        history = list(session)
        agent = RAGAgent(
            runtime.index_path,
            runtime.dataset["experts"],
            runtime.client,
            history=history,
        )
        calls_before = getattr(runtime.client, "calls", 0)
        tokens_before = getattr(runtime.client, "total_tokens", 0)
        try:
            result = agent.answer(
                request.message,
                request.expert_id,
                user_id=user_id,
                record_id="chat_" + uuid.uuid4().hex,
                at=at,
                deadline=deadline,
            )
            answer = result["answer"]
            text = _chat_answer_text(answer["text"], answer["citations"])
            if not text:
                raise ValueError("Chat answer consists only of citations")
        except Exception as exc:
            logger.warning("chat failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Эксперт временно недоступен. Попробуйте ещё раз.") from None

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
                "text": text,
            },
        ])
        _trim_chat_session(session)
        runtime.session_touched[session_id] = time.monotonic()
        logger.info(
            "chat completed: calls=%d tokens=%d total=%.1fs messages=%d",
            max(0, getattr(runtime.client, "calls", 0) - calls_before),
            max(0, getattr(runtime.client, "total_tokens", 0) - tokens_before),
            time.monotonic() - started,
            len(session),
        )
        return {
            "session_id": session_id,
            "expert_id": request.expert_id,
            "text": text,
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
