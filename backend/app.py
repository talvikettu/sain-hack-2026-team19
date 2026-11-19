"""FastAPI assembly: middleware, health, routers and the static client."""

from __future__ import annotations

import threading
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, chat, routes_run
from .config import (
    CASE_ID, MAX_INFLIGHT_REQUESTS, TEAM, WEB_DIR, _REQUEST_ID_HEADER,
    _REQUEST_ID_PATTERN,
)
from .context import bind_request_id, reset_request_id

app = FastAPI(title=f"hackathon {TEAM} / case {CASE_ID}")

# Admission happens in middleware, before Starlette allocates a worker thread
# for a synchronous endpoint. Health and read endpoints are never admitted, so
# liveness stays independent of a long official run.
_admission = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)
_ADMISSION_PATHS = frozenset({"/run", "/chat", "/session"})


@app.middleware("http")
async def _request_context(request: Request, call_next):
    incoming = request.headers.get(_REQUEST_ID_HEADER, "")
    request_id = (
        incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex[:16]
    )
    token = bind_request_id(request_id)
    admitted = request.url.path in _ADMISSION_PATHS
    if admitted and not _admission.acquire(blocking=False):
        reset_request_id(token)
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
        reset_request_id(token)
    response.headers[_REQUEST_ID_HEADER] = request_id
    return response


@app.get("/")
def root():
    index = WEB_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"team": TEAM, "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"team": TEAM, "status": "ok"}


app.include_router(catalog.router)
app.include_router(chat.router)
app.include_router(routes_run.router)

if (WEB_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
