"""Retrieval orchestration, bounded retry, fallbacks and answer validation."""

from __future__ import annotations

import math
import re
import time
from typing import Any

from rag_agent import RAGAgent
from rag_agent.client import (
    BudgetError, ChatTerminalError, ChatTransientError, ProxyProtocolError,
)
from rag_agent.retrieval import merge_rankings, search_rankings
from rag_preprocessing import pack_context

from .config import (
    TRANSIENT_COOLDOWN_BASE_SECONDS, TRANSIENT_COOLDOWN_MAX_SECONDS,
)
from .log import logger

class RunContext:
    def __init__(self, retry_delays: tuple[float, ...] = (5.0, 10.0),
                 cooldown_base: float = TRANSIENT_COOLDOWN_BASE_SECONDS) -> None:
        self.skip_llm = False
        self.fallbacks = 0
        self.generated = 0
        self.retrieval_failures = 0
        self.unsupported_fallbacks = 0
        self.validation_failures = 0
        self.group_citation_failures = 0
        self.transient_failures = 0
        self.consecutive_transient = 0
        self.retries = 0
        self.cooldown_until = 0.0
        self.retry_not_before = 0.0
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


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Return a sane server Retry-After hint, or None when it is unusable."""
    value = getattr(exc, "retry_after", None)
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _note_failure(
    exc: BaseException, context: RunContext, deadline: float | None = None
) -> None:
    """Stop only on a confirmed budget/auth/protocol stop.

    Transient connectivity and per-minute 429 failures use a bounded cooldown
    and a later probe instead of disabling generation for the rest of the run.
    A provider Retry-After is never shortened: if it reaches past the run
    deadline, dispatch stops and the remaining records use local fallbacks.
    """
    if (_budget_exhausted(exc) or _protocol_failure(exc)
            or _terminal_error(exc)):
        context.skip_llm = True
        return
    if _retryable(exc):
        context.transient_failures += 1
        context.consecutive_transient += 1
        hint = _retry_after_seconds(exc)
        if hint is not None:
            earliest = time.monotonic() + hint
            context.retry_not_before = max(context.retry_not_before, earliest)
            if deadline is not None and earliest >= deadline:
                logger.warning(
                    "provider retry hint exceeds the run deadline; "
                    "using fallbacks for the remaining records"
                )
                context.skip_llm = True
                return
            context.cooldown_until = max(context.cooldown_until, earliest)
            return
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


def _validate_evidence(
    evidence: Any, sections: set[tuple[str, str]]
) -> tuple[list[dict[str, Any]], bool]:
    """Reject a malformed dependency evidence envelope at the boundary.

    Returns ``(blocks, well_formed)``. An empty list is well formed: the
    existing group-local recovery path handles it. A nonempty list is accepted
    only when every block names an existing source pair, so generation and
    fallback construction cannot fail on a bad dependency response.
    """
    if not isinstance(evidence, list):
        return [], False
    if not evidence:
        return [], True
    blocks: list[dict[str, Any]] = []
    for block in evidence:
        if not isinstance(block, dict):
            return [], False
        article_id = block.get("article_id")
        section_id = block.get("section_id")
        if not isinstance(article_id, str) or not article_id:
            return [], False
        if not isinstance(section_id, str) or not section_id:
            return [], False
        if (article_id, section_id) not in sections:
            return [], False
        if "text" in block and not isinstance(block["text"], str):
            return [], False
        if "metadata" in block and not isinstance(block["metadata"], dict):
            return [], False
        blocks.append(block)
    return blocks, True


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
            delay = _retry_after_seconds(exc)
            if delay is None:
                # Local backoff stays bounded; a provider hint is honoured as
                # given so a later call is never sent before recovery time.
                delay = min(max(float(context.retry_delays[attempt]), 0.0), 120.0)
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
    deadline_evidence: list[dict[str, Any]] | None = None
    # Process one group at a time: evidence for all groups is not retained.
    for members in grouped.values():
        if time.monotonic() >= deadline:
            # The run deadline is exhausted. Do not call retrieval or the
            # model again: answer every remaining record locally from one
            # reusable section so the batch stays bounded and schema-valid.
            if deadline_evidence is None:
                deadline_evidence = _last_resort_evidence(dataset)
            if not deadline_evidence:
                raise RuntimeError("Dataset has no usable article sections for a citation")
            for request in members:
                answers[request["record_id"]] = _fallback_answer(
                    request, deadline_evidence, unsupported=True,
                )
                context.fallbacks += 1
                context.unsupported_fallbacks += 1
            continue
        failed = False
        unsupported = False
        retrieval_started = time.monotonic()
        try:
            evidence = agent.prepare_group(members, deadline=deadline)
        except Exception:
            failed = True
            logger.warning("group retrieval failed")
            evidence = []
        else:
            evidence, well_formed = _validate_evidence(evidence, sections)
            if not well_formed:
                failed = True
                logger.warning("group retrieval returned malformed evidence")
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
        group_generated: set[str] = set()
        for request in ordered:
            record_id = request["record_id"]
            if not unsupported and not context.skip_llm and time.monotonic() < deadline:
                # A transient outage gets a bounded pause and a later probe,
                # never a permanent skip for all remaining records.
                wait = min(
                    max(0.0, context.cooldown_until - time.monotonic()),
                    max(0.0, deadline - time.monotonic()),
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
                        group_generated.add(record_id)
                        continue
                    except Exception as exc:
                        logger.warning("record failed: %s", type(exc).__name__)
                        _note_failure(exc, context, deadline)
            answers[record_id] = _fallback_answer(
                request, evidence, unsupported=unsupported, topical=not unsupported,
            )
            context.fallbacks += 1
            if unsupported:
                context.unsupported_fallbacks += 1

        # Records sharing an evidence group must cite one shared source set.
        # A dependency regression that breaks this invariant is repaired inside
        # the group instead of failing the whole batch at final validation.
        if len(members) > 1:
            signatures = {
                frozenset(
                    (citation["article_id"], citation["section_id"])
                    for citation in answers[request["record_id"]]["citations"]
                )
                for request in members
            }
            if len(signatures) > 1:
                context.group_citation_failures += 1
                logger.warning("group citations disagreed; using the shared group fallback")
                for request in members:
                    record_id = request["record_id"]
                    if record_id in group_generated:
                        context.generated -= 1
                        context.fallbacks += 1
                        if unsupported:
                            context.unsupported_fallbacks += 1
                    answers[record_id] = _fallback_answer(
                        request, evidence,
                        unsupported=unsupported, topical=not unsupported,
                    )
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
