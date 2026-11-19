"""Organizer API access and dataset contract validation."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from rag_agent.prompts import PERSONA_FILES
from rag_preprocessing.embeddings import NoRedirect

from .config import (
    CASE_ID, DATASET_READ_CHUNK, MAX_DATASET_BYTES, MAX_DATASET_REQUESTS,
    _ID_PATTERN,
)
from .errors import DatasetAccessError, DatasetContractError


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
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    for item in (getattr(raw, "_sock", None), raw, fp):
        if item is None or not hasattr(item, "settimeout"):
            continue
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
