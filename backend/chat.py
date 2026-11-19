"""Browser sessions, chat quota, idempotent turns and chat routes."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import secrets
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from rag_agent import RAGAgent

from .answering import _budget_exhausted
from .config import (
    CHAT_BUDGET_SECONDS, CHAT_GLOBAL_RATE_LIMIT, CHAT_MAX_CHARS,
    CHAT_MAX_MESSAGES, CHAT_MAX_SESSIONS, CHAT_RATE_LIMIT, CHAT_RATE_WINDOW,
    CHAT_SESSION_EVICT_GRACE_SECONDS, CHAT_SESSION_ISSUE_LIMIT,
    CHAT_SESSION_TTL_SECONDS, CHAT_TURN_CACHE_MAX, LLM_MAX_CALLS,
    LLM_TOKEN_BUDGET, RUN_RESERVED_CALLS, RUN_RESERVED_TOKENS,
)
from .context import request_id
from .errors import SessionCapacityError
from .log import logger
from .models import ChatRequest
from .state import Runtime, _get_runtime
from .usage import client_calls, client_tokens

router = APIRouter()


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


@router.post("/session")
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
        request_id(), len(runtime.sessions), runtime.generation,
    )
    # expires_in is an idle timeout: an authenticated chat refreshes it.
    return {
        "session_id": session_id,
        "token": token,
        "user_id": runtime.session_users[session_id],
        "expires_in": CHAT_SESSION_TTL_SECONDS,
    }


def _turn_fingerprint(expert_id: str, message: str) -> str:
    return sha256(
        json.dumps(
            {"expert_id": expert_id, "message": message},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@router.post("/chat")
def chat(request: ChatRequest,
         authorization: str | None = Header(default=None)) -> dict[str, Any]:
    rid = request_id()
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
    calls_before = client_calls(runtime.client)
    tokens_before = client_tokens(runtime.client)
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
                rid, stage, time.monotonic() - started,
                max(0, client_calls(runtime.client) - calls_before),
                max(0, client_tokens(runtime.client) - tokens_before),
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
                    rid, stage, time.monotonic() - started,
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
            rid,
            max(0, client_calls(runtime.client) - calls_before),
            max(0, client_tokens(runtime.client) - tokens_before),
            time.monotonic() - started,
            message_count,
        )
        return response
    finally:
        with runtime.sessions_lock:
            runtime.inflight_turns.pop(turn_key, None)
            runtime.busy_sessions.discard(session_id)
