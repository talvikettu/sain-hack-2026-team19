"""Official POST /run handler and transactional runtime publication."""

from __future__ import annotations

import time

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from . import answering, cache, organizer, state
from .config import (
    CASE_ID, RUN_BUDGET_SECONDS, RUN_SERIALIZATION_RESERVE_SECONDS,
)
from .context import request_id
from .log import logger
from .models import RunRequest
from .usage import client_calls, client_tokens

router = APIRouter()

@router.post("/run")
def run(request: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    rid = request_id()
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
    if not state._run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="An official run is already in progress",
            headers={"Retry-After": "30"},
        )
    stage = "fetch"
    candidate: state.Runtime | None = None
    try:
        fetch_started = time.monotonic()
        try:
            dataset = organizer._fetch_dataset(
                request.dataset_code,
                authorization,
                timeout=min(120.0, max(0.0, generation_deadline - fetch_started)),
            )
            organizer._validate_dataset(dataset)
        except organizer.DatasetAccessError as exc:
            logger.warning(
                "run rejected: rid=%s stage=%s access=HTTP %d elapsed=%.1fs",
                rid, stage, exc.status_code, time.monotonic() - started,
            )
            raise HTTPException(status_code=exc.status_code, detail="Dataset access rejected") from None
        except organizer.DatasetContractError as exc:
            logger.error(
                "run rejected: rid=%s stage=validate field=%s elapsed=%.1fs",
                rid, exc.field, time.monotonic() - started,
            )
            raise HTTPException(status_code=502, detail="Dataset contract invalid") from None
        except Exception as exc:
            logger.error(
                "run failed: rid=%s stage=%s elapsed=%.1fs error=%s",
                rid, stage, time.monotonic() - started, type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail="Dataset fetch failed") from None
        fetch_seconds = time.monotonic() - fetch_started

        try:
            cache_key: str | None = cache._run_cache_key(dataset)
        except Exception:
            cache_key = None

        # Reusing an identical active runtime preserves browser sessions and
        # avoids rebuilding an index for a repeated/cached dataset code.
        active = state._active_runtime()
        runtime = active if cache_key is not None and active is not None \
            and active.dataset_key == cache_key else None
        build_seconds = 0.0
        if runtime is None:
            stage = "build"
            build_started = time.monotonic()
            try:
                candidate = state._build_runtime(dataset, generation_deadline, cache_key)
                if cache_key is not None:
                    candidate.set_dataset_key(cache_key)
            except Exception as exc:
                logger.error(
                    "run failed: rid=%s stage=%s elapsed=%.1fs error=%s",
                    rid, stage, time.monotonic() - started, type(exc).__name__,
                )
                raise HTTPException(status_code=500, detail="Runtime initialization failed") from None
            runtime = candidate
            build_seconds = time.monotonic() - build_started

        if cache_key is not None:
            cached = cache._cache_lookup(cache_key)
            if cached is not None:
                if candidate is not None:
                    state._publish_runtime(candidate)
                    state._prune_indexes(candidate.index_path)
                logger.info(
                    "run cache hit: rid=%s requests=%d reused=%s fetch=%.1fs "
                    "build=%.1fs total=%.1fs",
                    rid, len(cached["requests"]), candidate is None,
                    fetch_seconds, build_seconds, time.monotonic() - started,
                )
                return JSONResponse(
                    content=cached, headers={"Cache-Control": "no-store"}
                )

        stage = "generate"
        context = answering.RunContext()
        calls_before = client_calls(runtime.client)
        tokens_before = client_tokens(runtime.client)
        try:
            result = answering._answer_requests(
                runtime.agent, runtime.dataset, generation_deadline, context
            )
            answering._validate_run_result(runtime.dataset, result)
        except Exception as exc:
            if candidate is not None:
                state._discard_candidate(candidate)
            logger.error(
                "run failed: rid=%s stage=%s elapsed=%.1fs calls=%d tokens=%d error=%s",
                rid, stage, time.monotonic() - started,
                max(0, client_calls(runtime.client) - calls_before),
                max(0, client_tokens(runtime.client) - tokens_before),
                type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="Run failed") from None

        calls_used = max(0, client_calls(runtime.client) - calls_before)
        tokens_used = max(0, client_tokens(runtime.client) - tokens_before)
        clean = (
            context.fallbacks == 0
            and context.retrieval_failures == 0
            and context.group_citation_failures == 0
            and context.unsupported_fallbacks == 0
            and not context.skip_llm
        )
        if cache_key is not None and clean:
            try:
                cache._cache_store(cache_key, result)
            except Exception as exc:
                logger.warning(
                    "run cache store skipped: rid=%s error=%s",
                    rid, type(exc).__name__,
                )
        stage = "publish"
        if candidate is not None:
            state._publish_runtime(candidate)
            state._prune_indexes(candidate.index_path)
        logger.info(
            "run completed: rid=%s requests=%d generated=%d fallbacks=%d unsupported=%d "
            "retrieval_failures=%d validation_failures=%d group_citations=%d "
            "transient=%d stopped=%s "
            "retries=%d run_calls=%d run_tokens=%d calls=%d tokens=%d reused=%s "
            "fetch=%.1fs build=%.1fs retrieval=%.1fs generation=%.1fs total=%.1fs",
            rid, len(result["requests"]), context.generated, context.fallbacks,
            context.unsupported_fallbacks, context.retrieval_failures,
            context.validation_failures, context.group_citation_failures,
            context.transient_failures, context.skip_llm,
            context.retries, context.llm_calls, context.llm_tokens,
            calls_used, tokens_used, candidate is None, fetch_seconds, build_seconds,
            context.retrieval_seconds, context.generation_seconds,
            time.monotonic() - started,
        )
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})
    finally:
        state._run_lock.release()


