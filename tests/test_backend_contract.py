"""Contract-level tests for the /run shape. No network and no LLM calls."""

from __future__ import annotations

import json
import os
import re
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from backend import (
    answering, cache, catalog, chat, config, models, organizer, routes_run, state,
)
from backend.app import app as backend_app
from backend.errors import SessionCapacityError
from rag_agent import RAGAgent
from rag_agent.client import (
    BudgetError, ChatClient, ChatTerminalError, ChatTransientError,
)
from rag_agent.retrieval import search_rankings
from rag_preprocessing import build_index

ROOT = Path(__file__).resolve().parents[1]
DEV_DATASET = ROOT / "data" / "dataset.json"


def _dataset() -> dict:
    return {
        "meta": {"dataset_id": "test", "case_id": 4, "phase": "dev"},
        "users": [{"user_id": "u1"}],
        "chat_history": [],
        "articles": [
            {
                "article_id": "art_01",
                "title": "Caffeine and sleep",
                "authors": ["Test Author"],
                "year": 2026,
                "doi": "10.0000/test",
                "license": {"name": "CC BY", "version": "4.0", "url": "https://example.test"},
                "sections": [
                    {"section_id": "s001", "title": "Results", "text": "Caffeine can delay sleep."}
                ],
            }
        ],
        "experts": [
            {"expert_id": expert_id, "role": "role", "speech_style": "style"}
            for expert_id in ("clinician", "biohacker", "nutritionist")
        ],
        "requests": [
            {
                "record_id": f"r_{expert_id}",
                "user_id": "u1",
                "expert_id": expert_id,
                "group_id": "g1",
                "utterance": {"role": "user", "at": "2026-11-17T12:00:00+03:00", "text": "coffee?"},
            }
            for expert_id in ("clinician", "biohacker", "nutritionist")
        ],
    }


def _answer(record_id: str, refused: bool = False) -> dict:
    return {
        "record_id": record_id,
        "text": f"Answer [art_01 / s001]",
        "refused": refused,
        "citations": [{"article_id": "art_01", "section_id": "s001"}],
    }


def _chat_credential(runtime) -> tuple[str, str]:
    session_id, token = chat._create_chat_session(runtime)
    return session_id, "Bearer " + token


class _FakeAgent:
    def __init__(self, fail_for: set[str] | None = None, error: str = "boom",
                 prepare_error: str | None = None, prepare_evidence: list | None = None) -> None:
        self.fail_for = fail_for or set()
        self.error = error
        self.prepare_error = prepare_error
        self.prepare_evidence = prepare_evidence
        self.calls: list[str] = []

    def prepare_group(self, members, *, deadline=None):
        if self.prepare_error:
            raise RuntimeError(self.prepare_error)
        if self.prepare_evidence is not None:
            return self.prepare_evidence
        return [{"article_id": "art_01", "section_id": "s001"}]

    def answer(self, question, expert_id, **kwargs):
        self.calls.append(expert_id)
        if expert_id in self.fail_for:
            raise RuntimeError(self.error)
        return {"answer": _answer(kwargs["record_id"])}


class _FakeChat:
    """Returns a valid compact answer with adjacent citation markers."""

    def complete(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        if not payload.get("evidence"):
            raise AssertionError("fake chat needs evidence")
        refs = [block["ref"] for block in payload["evidence"]]
        return {
            "content": json.dumps({"text": "Ответ" + "".join(f"[{ref}]" for ref in refs) + ".\n\nВторой абзац.", "refused": False, "refs": refs}),
            "finish_reason": "stop",
            "usage": {"total_tokens": 100},
            "seconds": 0.0,
        }


class RunContractTests(unittest.TestCase):
    def setUp(self):
        with cache._run_cache_lock:
            cache._run_cache.clear()

    @unittest.skipUnless(DEV_DATASET.is_file(), "dev dataset not present")
    def test_real_dev_dataset_groups_share_citations(self):
        dataset = json.loads(DEV_DATASET.read_text(encoding="utf-8"))
        index = ROOT / "tmp" / "backend-test-index.sqlite3"
        build_index(dataset, index)
        agent = RAGAgent(index, dataset["experts"], _FakeChat(), history=dataset["chat_history"])
        comparison = [request for request in dataset["requests"] if request["group_id"] == "g_se9fna"]
        self.assertEqual(len(comparison), 3)
        evidence = agent.prepare_group(comparison)
        self.assertEqual([(item["article_id"], item["section_id"]) for item in evidence], [("art_12", "s015")])
        paraphrase = agent.prepare(
            "Если в офисе днём светло, меняет ли это разницу между экраном и книжкой перед сном?",
            "clinician", user_id="synthetic", record_id="paraphrase",
        )
        first = paraphrase["evidence"][0]
        self.assertEqual((first["article_id"], first["section_id"]), ("art_12", "s015"))
        routine = [request for request in dataset["requests"] if request["group_id"] == "g_ueuvez"]
        rankings = search_rankings(index, agent.group_query(routine))
        self.assertIn("art_07", {hit["article_id"] for ranking in rankings for hit in ranking})
        shared = agent.prepare_group(routine)
        self.assertEqual([(item["article_id"], item["section_id"]) for item in shared], [("art_04", "s006")])
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 300, answering.RunContext(retry_delays=())
        )
        self.assertEqual(len(result["requests"]), len(dataset["requests"]))
        grouped: dict[str, set[str]] = {}
        for request, answer in zip(dataset["requests"], result["requests"]):
            self.assertEqual(answer["record_id"], request["record_id"])
            self.assertEqual(set(answer), {"record_id", "text", "refused", "citations"})
            self.assertEqual(set(answer["citations"][0]), {"article_id", "section_id"})
            self.assertTrue(answer["text"])
            grouped.setdefault(request["group_id"], set()).add(
                json.dumps(answer["citations"], sort_keys=True)
            )
        for group_id, citation_sets in grouped.items():
            self.assertEqual(len(citation_sets), 1, group_id)

    def test_exact_schema_and_group_citations(self):
        dataset = _dataset()
        result = answering._answer_requests(
            _FakeAgent(), dataset, time.monotonic() + 60, answering.RunContext(retry_delays=())
        )
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(
            [item["record_id"] for item in result["requests"]],
            [request["record_id"] for request in dataset["requests"]],
        )
        first_citations = result["requests"][0]["citations"]
        for item in result["requests"]:
            self.assertEqual(set(item), {"record_id", "text", "refused", "citations"})
            self.assertEqual(set(item["citations"][0]), {"article_id", "section_id"})
            self.assertIsInstance(item["refused"], bool)
            self.assertEqual(item["citations"], first_citations)

    def test_one_failed_expert_uses_the_group_citation_set(self):
        dataset = _dataset()
        agent = _FakeAgent(fail_for={"biohacker", "nutritionist"})
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 60, answering.RunContext(retry_delays=())
        )
        self.assertEqual(len(result["requests"]), 3)
        self.assertFalse(result["requests"][0]["refused"])
        self.assertTrue(result["requests"][1]["refused"])
        self.assertTrue(result["requests"][2]["refused"])
        self.assertEqual(
            [item["citations"] for item in result["requests"]],
            [[{"article_id": "art_01", "section_id": "s001"}]] * 3,
        )

    def test_transient_rate_limit_does_not_disable_remaining_records(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Chat proxy HTTP 429; no automatic retry",
        )
        context = answering.RunContext(retry_delays=(), cooldown_base=0)
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(len(result["requests"]), 3)
        # A per-request transient failure no longer sets skip_llm for the run.
        self.assertEqual(agent.calls, ["clinician", "biohacker", "nutritionist"])
        self.assertTrue(all(item["refused"] for item in result["requests"]))
        self.assertFalse(context.skip_llm)
        self.assertEqual(context.transient_failures, 3)

    def test_transient_failure_recovers_for_later_records(self):
        dataset = _dataset()
        agent = _FakeAgent(fail_for={"clinician"},
                           error="Chat proxy HTTP 429; no automatic retry")
        context = answering.RunContext(retry_delays=(0,), cooldown_base=0)
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(agent.calls, ["clinician", "clinician", "biohacker", "nutritionist"])
        self.assertEqual([item["refused"] for item in result["requests"]], [True, False, False])
        self.assertEqual(context.transient_failures, 1)
        self.assertEqual(context.consecutive_transient, 0)
        self.assertEqual(context.generated, 2)

    def test_retry_after_beyond_deadline_stops_remaining_dispatch(self):
        dataset = _dataset()

        class LongThrottleAgent(_FakeAgent):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def answer(self, question, expert_id, **kwargs):
                self.attempts += 1
                raise ChatTransientError(
                    "Chat proxy HTTP 429", retry_after=600.0, status_code=429,
                )

        agent = LongThrottleAgent()
        context = answering.RunContext(retry_delays=(0,))
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 30, context
        )
        self.assertEqual(agent.attempts, 1)
        self.assertTrue(context.skip_llm)
        self.assertEqual(context.retries, 0)
        self.assertEqual(context.fallbacks, 3)
        self.assertTrue(all(answer["citations"] for answer in result["requests"]))

    def test_provider_retry_after_is_not_shortened(self):
        dataset = _dataset()

        class OnceThrottledAgent(_FakeAgent):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def answer(self, question, expert_id, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise ChatTransientError(
                        "Chat proxy HTTP 429", retry_after=0.05, status_code=429,
                    )
                return super().answer(question, expert_id, **kwargs)

        agent = OnceThrottledAgent()
        context = answering.RunContext(retry_delays=(0,))
        started = time.monotonic()
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 30, context
        )
        self.assertGreaterEqual(time.monotonic() - started, 0.05)
        self.assertEqual(context.retries, 1)
        self.assertEqual(context.generated, 3)
        self.assertEqual(agent.attempts, 4)
        self.assertTrue(all(answer["citations"] for answer in result["requests"]))

    def test_local_budget_exhaustion_is_not_retried(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Local LLM call/token budget exhausted",
        )
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 60, answering.RunContext(retry_delays=(0, 0))
        )
        self.assertEqual(len(agent.calls), 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))

    def test_auth_failure_stops_remaining_calls(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Chat proxy HTTP 401; no automatic retry",
        )
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 60, answering.RunContext(retry_delays=(0, 0))
        )
        self.assertEqual(len(agent.calls), 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))

    def test_protocol_failure_stops_remaining_calls(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Proxy response lacks valid token usage; stop to protect budget",
        )
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 60, answering.RunContext(retry_delays=(0, 0))
        )
        self.assertEqual(len(agent.calls), 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))

    def test_empty_sections_are_skipped_by_the_keyword_fallback(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine"
        dataset["articles"][0]["sections"].insert(
            0, {"section_id": "empty", "title": "Empty", "text": "   "}
        )
        agent = _FakeAgent(prepare_error="index unavailable")
        evidence = answering._fallback_evidence(dataset, dataset["requests"], agent,
                                           time.monotonic() + 60)
        self.assertEqual(evidence[0]["section_id"], "s001")

    def test_keyword_fallback_never_cites_an_unrelated_section(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "stock market volatility"
        agent = _FakeAgent(prepare_error="index unavailable")
        evidence = answering._fallback_evidence(dataset, dataset["requests"], agent,
                                           time.monotonic() + 60)
        self.assertEqual(evidence, [])

    def test_read_limited_rejects_an_oversized_dataset_body(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def read(self, size=-1):
                chunk, self.payload = self.payload[:size], self.payload[size:]
                return chunk

        with self.assertRaises(RuntimeError):
            organizer._read_limited(Response(b"x" * 10), 8, time.monotonic() + 60)
        self.assertEqual(
            organizer._read_limited(Response(b"x" * 8), 8, time.monotonic() + 60), b"x" * 8
        )

    def test_retrieval_failure_does_not_abort_the_run(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine"
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            prepare_error="index unavailable",
        )
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(context.retrieval_failures, 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))
        self.assertTrue(all(item["citations"] for item in result["requests"]))
        self.assertEqual(context.unsupported_fallbacks, 0)

    def test_unrelated_retrieval_failure_skips_generation(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "stock market volatility"
        agent = _FakeAgent(prepare_error="index unavailable")
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(agent.calls, [])
        self.assertEqual(context.unsupported_fallbacks, 3)
        self.assertEqual(context.fallbacks, 3)
        self.assertTrue(all(item["refused"] for item in result["requests"]))
        self.assertTrue(all(item["citations"] for item in result["requests"]))

    def test_malformed_middle_group_does_not_abort_the_batch(self):
        dataset = _dataset()
        template = dataset["requests"][0]
        dataset["requests"] = [
            {**json.loads(json.dumps(template)), "record_id": f"r{i}", "group_id": f"g{i}"}
            for i in range(3)
        ]

        class MiddleGroupAgent(_FakeAgent):
            def __init__(self, bad_evidence):
                super().__init__()
                self.bad_evidence = bad_evidence
                self.prepared = 0

            def prepare_group(self, members, *, deadline=None):
                self.prepared += 1
                if self.prepared == 2:
                    return self.bad_evidence
                return [{"article_id": "art_01", "section_id": "s001"}]

        for bad_evidence in (
            [{"article_id": "art_01"}],
            [{"article_id": "art_99", "section_id": "s999"}],
            {"article_id": "art_01", "section_id": "s001"},
        ):
            with self.subTest(bad_evidence=bad_evidence):
                agent = MiddleGroupAgent(bad_evidence)
                context = answering.RunContext(retry_delays=())
                result = answering._answer_requests(
                    agent, dataset, time.monotonic() + 60, context
                )
                self.assertEqual(len(result["requests"]), 3)
                self.assertEqual(context.retrieval_failures, 1)
                self.assertEqual(context.generated, 2)
                self.assertEqual(context.fallbacks, 1)
                self.assertTrue(all(answer["citations"] for answer in result["requests"]))

    def test_group_citation_disagreement_is_repaired_locally(self):
        dataset = _dataset()
        dataset["articles"].append({
            "article_id": "art_02",
            "title": "Evening light",
            "authors": ["Test Author"],
            "year": 2026,
            "doi": "10.0000/light",
            "license": {"name": "CC BY", "version": "4.0", "url": "https://example.test"},
            "sections": [
                {"section_id": "s002", "title": "Results", "text": "Evening light can delay sleep."}
            ],
        })

        class DisagreeingAgent(_FakeAgent):
            def answer(self, question, expert_id, **kwargs):
                self.calls.append(expert_id)
                article_id, section_id = (
                    ("art_02", "s002") if expert_id == "biohacker" else ("art_01", "s001")
                )
                return {"answer": {
                    "record_id": kwargs["record_id"],
                    "text": f"Answer [{article_id} / {section_id}]",
                    "refused": False,
                    "citations": [{"article_id": article_id, "section_id": section_id}],
                }}

        agent = DisagreeingAgent()
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(
            agent, dataset, time.monotonic() + 60, context
        )
        self.assertEqual(context.group_citation_failures, 1)
        self.assertEqual(context.generated, 0)
        self.assertEqual(context.fallbacks, 3)
        self.assertEqual(
            [answer["citations"] for answer in result["requests"]],
            [[{"article_id": "art_01", "section_id": "s001"}]] * 3,
        )

    def test_expired_deadline_skips_retrieval_and_generation(self):
        dataset = _dataset()

        class TrackingAgent(_FakeAgent):
            def __init__(self):
                super().__init__()
                self.prepared = 0

            def prepare_group(self, members, *, deadline=None):
                self.prepared += 1
                return super().prepare_group(members, deadline=deadline)

        agent = TrackingAgent()
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(
            agent, dataset, time.monotonic() - 1, context
        )
        self.assertEqual(agent.prepared, 0)
        self.assertEqual(agent.calls, [])
        self.assertEqual(context.fallbacks, 3)
        self.assertEqual(context.unsupported_fallbacks, 3)
        self.assertTrue(all(answer["refused"] for answer in result["requests"]))
        self.assertTrue(all(answer["citations"] for answer in result["requests"]))

    def test_empty_group_evidence_uses_topical_fallback(self):
        dataset = _dataset()
        dataset["articles"].append({
            "article_id": "art_02",
            "title": "Evening light",
            "authors": ["Test Author"],
            "year": 2026,
            "doi": "10.0000/light",
            "license": {"name": "CC BY", "version": "4.0", "url": "https://example.test"},
            "sections": [{"section_id": "s001", "title": "Results", "text": "Evening light can delay sleep."}],
        })
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            prepare_evidence=[],
        )
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine and sleep"
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(context.retrieval_failures, 1)
        self.assertEqual(result["requests"][0]["citations"],
                         [{"article_id": "art_01", "section_id": "s001"}])

    def test_fallback_citation_exists_in_dataset(self):
        evidence = [{"article_id": "art_02", "section_id": "s003"}]
        answer = answering._fallback_answer(_dataset()["requests"][0], evidence)
        self.assertTrue(answer["refused"])
        self.assertEqual(answer["citations"], evidence)
        self.assertIn("[art_02 / s003]", answer["text"])
        self.assertIn("не могу", answer["text"])

    def test_topical_fallback_returns_a_grounded_ordinary_answer(self):
        dataset = _dataset()
        section = dataset["articles"][0]["sections"][0]
        section["text"] = (
            "Caffeine can delay sleep onset and reduce total sleep time in "
            "healthy adults, especially when it is consumed late in the day."
        )
        request = dataset["requests"][0]
        request["utterance"]["text"] = "How does caffeine affect sleep?"
        evidence = [{
            "article_id": "art_01",
            "section_id": "s001",
            "text": section["text"],
            "metadata": {
                "title": dataset["articles"][0]["title"],
                "section_title": section["title"],
            },
        }]
        answer = answering._fallback_answer(request, evidence, topical=True)
        self.assertFalse(answer["refused"])
        self.assertEqual(answer["citations"], [{"article_id": "art_01", "section_id": "s001"}])
        self.assertIn("Caffeine can delay sleep onset", answer["text"])
        self.assertIn("[art_01 / s001]", answer["text"])

    @unittest.skipUnless(DEV_DATASET.is_file(), "dev dataset not present")
    def test_chat_handler_with_fake_client(self):
        dataset = json.loads(DEV_DATASET.read_text(encoding="utf-8"))
        index = ROOT / "tmp" / "backend-test-chat-index.sqlite3"
        build_index(dataset, index)
        client = _FakeChat()
        agent = RAGAgent(index, dataset["experts"], client, history=dataset["chat_history"])
        runtime = state.Runtime(dataset, client, agent)
        session_id, authorization = _chat_credential(runtime)
        original_index, original_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = index, runtime
        try:
            response = chat.chat(
                models.ChatRequest(
                    expert_id="clinician",
                    message="Как кофеин влияет на сон?",
                    session_id=session_id,
                ),
                authorization=authorization,
            )
        finally:
            state.INDEX_PATH, state._runtime = original_index, original_runtime
        self.assertEqual(response["session_id"], session_id)
        self.assertFalse(response["refused"])
        self.assertEqual(response["text"], "Ответ.\n\nВторой абзац.")
        self.assertEqual(set(response["citations"][0]), {
            "article_id", "section_id", "article_title", "section_title", "year",
            "dataset_generation",
        })
        self.assertEqual(len(runtime.sessions[session_id]), 2)
        self.assertEqual(runtime.sessions[session_id][0]["role"], "user")
        self.assertEqual(runtime.sessions[session_id][1]["role"], "agent")
        self.assertEqual(runtime.sessions[session_id][1]["text"], response["text"])

    def test_chat_does_not_accept_a_client_selected_user(self):
        with self.assertRaises(PydanticValidationError):
            models.ChatRequest(
                expert_id="clinician", message="hello", user_id="u2",
            )

    def test_chat_does_not_send_stored_dataset_history(self):
        dataset = _dataset()
        dataset["chat_history"] = [{
            "message_id": "m1", "user_id": "u1", "role": "user",
            "at": "2026-11-01T10:00:00+03:00", "text": "PRIVATE-DATASET-HISTORY",
        }]
        index = ROOT / "tmp" / "backend-test-chat-history.sqlite3"
        build_index(dataset, index)

        class RecordingChat(_FakeChat):
            def __init__(self):
                self.payloads = []

            def complete(self, messages, **kwargs):
                self.payloads.append(messages)
                return super().complete(messages, **kwargs)

        client = RecordingChat()
        agent = RAGAgent(index, dataset["experts"], client, history=dataset["chat_history"])
        runtime = state.Runtime(dataset, client, agent)
        session_id, authorization = _chat_credential(runtime)
        original_index, original_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = index, runtime
        try:
            chat.chat(models.ChatRequest(
                expert_id="clinician",
                message="Как раньше, кофе влияет на сон?",
                session_id=session_id,
            ), authorization=authorization)
        finally:
            state.INDEX_PATH, state._runtime = original_index, original_runtime
        self.assertTrue(client.payloads)
        self.assertNotIn("PRIVATE-DATASET-HISTORY", json.dumps(client.payloads, ensure_ascii=False))

    def test_chat_session_store_is_bounded(self):
        session = [
            {"role": "user" if index % 2 == 0 else "agent", "text": "x" * 100}
            for index in range(chat.CHAT_MAX_MESSAGES + 10)
        ]
        chat._trim_chat_session(session)
        self.assertEqual(len(session), chat.CHAT_MAX_MESSAGES)
        session = [{"role": "user", "text": "x" * 1000} for _ in range(30)]
        chat._trim_chat_session(session)
        self.assertLessEqual(sum(len(message["text"]) for message in session), chat.CHAT_MAX_CHARS)

    def test_chat_rate_limit_is_bounded(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        for _ in range(chat.CHAT_RATE_LIMIT):
            self.assertTrue(chat._chat_session_rate_allowed(runtime, "session"))
        self.assertFalse(chat._chat_session_rate_allowed(runtime, "session"))

    def test_chat_global_rate_limit_is_bounded_across_sessions(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        for _ in range(chat.CHAT_GLOBAL_RATE_LIMIT):
            self.assertTrue(chat._chat_global_rate_allowed(runtime))
        self.assertFalse(chat._chat_global_rate_allowed(runtime))

    def test_chat_budget_reserves_two_calls_per_request(self):
        dataset = _dataset()
        dataset["requests"] = dataset["requests"] * 20  # 60 requests
        runtime = state.Runtime(dataset, object(), _FakeAgent())

        class Client:
            calls = chat.LLM_MAX_CALLS - chat.RUN_RESERVED_CALLS
            total_tokens = 0

        runtime.client = Client()
        self.assertFalse(chat._chat_budget_available(runtime))
        # The official-run reserve stays in place even after a completed run:
        # a retry must not find the budget consumed by demo traffic.
        runtime.client.calls = chat.LLM_MAX_CALLS - chat._run_reserve_calls(runtime) - 1
        self.assertTrue(chat._chat_budget_available(runtime))

    def test_chat_session_owner_comes_from_the_server_session(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        runtime.session_users[session_id] = "u2"
        resolved_session, owner = chat._resolve_browser_session(runtime, authorization)
        self.assertEqual(resolved_session, session_id)
        self.assertEqual(owner, "u2")

    def test_chat_requires_a_browser_session_token(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        shared_index, shared_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(HTTPException) as raised:
                chat.chat(models.ChatRequest(expert_id="clinician", message="hello"))
        finally:
            state.INDEX_PATH, state._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_rejects_a_session_id_from_another_browser(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        _, authorization = _chat_credential(runtime)
        shared_index, shared_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(HTTPException) as raised:
                chat.chat(
                    models.ChatRequest(
                        expert_id="clinician", message="hello", session_id="guessed",
                    ),
                    authorization=authorization,
                )
        finally:
            state.INDEX_PATH, state._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 403)

    def test_chat_rejects_a_guessable_browser_token(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        session_id, _ = _chat_credential(runtime)
        shared_index, shared_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(HTTPException) as raised:
                chat.chat(
                    models.ChatRequest(
                        expert_id="clinician", message="hello", session_id=session_id,
                    ),
                    authorization="Bearer guessed-token",
                )
        finally:
            state.INDEX_PATH, state._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_stores_only_a_hash_of_the_browser_token(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        token = authorization.removeprefix("Bearer ")
        self.assertNotIn(token, runtime.session_tokens)
        self.assertEqual(runtime.session_tokens[chat._token_hash(token)], session_id)

    def test_new_browser_session_uses_the_server_selected_user(self):
        dataset = _dataset()
        dataset["users"].append({"user_id": "u2"})
        runtime = state.Runtime(dataset, object(), _FakeAgent())
        session_id, _ = _chat_credential(runtime)
        self.assertEqual(runtime.session_users[session_id], "u1")
        self.assertEqual(runtime.sessions[session_id], [])

    def test_browser_tokens_do_not_survive_a_dataset_switch(self):
        runtime_a = state.Runtime(_dataset(), object(), _FakeAgent())
        dataset_b = _dataset()
        dataset_b["articles"][0]["article_id"] = "art_09"
        runtime_b = state.Runtime(dataset_b, object(), _FakeAgent())
        _, authorization = _chat_credential(runtime_a)
        shared_index, shared_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime_b
        try:
            with self.assertRaises(HTTPException) as raised:
                chat.chat(
                    models.ChatRequest(expert_id="clinician", message="hello"),
                    authorization=authorization,
                )
        finally:
            state.INDEX_PATH, state._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_rejects_a_blank_message(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        _, authorization = _chat_credential(runtime)
        shared_index, shared_runtime = state.INDEX_PATH, state._runtime
        state.INDEX_PATH, state._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(HTTPException) as raised:
                chat.chat(models.ChatRequest(expert_id="clinician", message="   "),
                          authorization=authorization)
        finally:
            state.INDEX_PATH, state._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 400)

    def test_expired_chat_sessions_are_dropped(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        runtime.session_touched[session_id] = (
            time.monotonic() - chat.CHAT_SESSION_TTL_SECONDS - 1
        )
        runtime.chat_times[session_id] = [time.monotonic()]
        with self.assertRaises(HTTPException):
            chat._resolve_browser_session(runtime, authorization)
        self.assertNotIn(session_id, runtime.sessions)
        self.assertNotIn(session_id, runtime.session_users)
        self.assertNotIn(session_id, runtime.chat_times)
        self.assertEqual(runtime.session_tokens, {})

    def test_run_route_returns_exact_body(self):
        dataset = _dataset()
        agent = _FakeAgent()
        runtime = state.Runtime(dataset, object(), agent)
        original_runtime = state._runtime
        try:
            with patch.object(organizer, "_fetch_dataset", return_value=dataset), \
                    patch.object(state, "_build_runtime", return_value=runtime):
                response = routes_run.run(
                    models.RunRequest(case=4, dataset_code="test-code"),
                    authorization="Bearer test-token",
                )
        finally:
            state._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(set(body), {"requests"})
        self.assertEqual(len(body["requests"]), 3)
        for answer in body["requests"]:
            self.assertEqual(set(answer), {"record_id", "text", "refused", "citations"})
            self.assertIn("[art_01 / s001]", answer["text"])

    def test_run_route_survives_retrieval_and_generation_failure(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            prepare_error="index unavailable",
        )
        runtime = state.Runtime(dataset, object(), agent)
        original_runtime = state._runtime
        try:
            with patch.object(organizer, "_fetch_dataset", return_value=dataset), \
                    patch.object(state, "_build_runtime", return_value=runtime):
                response = routes_run.run(
                    models.RunRequest(case=4, dataset_code="test-code"),
                    authorization="Bearer test-token",
                )
        finally:
            state._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(len(body["requests"]), 3)
        self.assertTrue(all(answer["refused"] for answer in body["requests"]))
        self.assertTrue(all(answer["citations"] for answer in body["requests"]))


class HttpHandlerTests(unittest.TestCase):
    """Exercise the actual FastAPI handlers, not only the route functions."""

    def setUp(self):
        self.client = TestClient(backend_app)
        with cache._run_cache_lock:
            cache._run_cache.clear()

    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"team": config.TEAM, "status": "ok"})

    def test_run_requires_a_bearer_token(self):
        response = self.client.post("/run", json={"case": 4, "dataset_code": "code"})
        self.assertEqual(response.status_code, 401)

    def test_run_rejects_other_cases(self):
        response = self.client.post(
            "/run",
            json={"case": 3, "dataset_code": "code"},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 400)

    def test_run_returns_the_exact_case4_schema(self):
        dataset = _dataset()
        runtime = state.Runtime(dataset, object(), _FakeAgent())
        original_runtime = state._runtime
        try:
            with patch.object(organizer, "_fetch_dataset", return_value=dataset), \
                    patch.object(state, "_build_runtime", return_value=runtime):
                response = self.client.post(
                    "/run",
                    json={"case": 4, "dataset_code": "test-code"},
                    headers={"Authorization": "Bearer test-token"},
                )
        finally:
            state._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"requests"})
        self.assertEqual(len(body["requests"]), 3)
        for answer in body["requests"]:
            self.assertEqual(set(answer), {"record_id", "text", "refused", "citations"})

    def test_chat_requires_a_browser_token_over_http(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        with patch.object(chat, "_get_runtime", return_value=runtime):
            response = self.client.post(
                "/chat", json={"expert_id": "clinician", "message": "hello"},
            )
        self.assertEqual(response.status_code, 401)

    def test_session_issues_an_opaque_token(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        original_runtime = state._runtime
        state._runtime = runtime
        try:
            response = self.client.post("/session")
        finally:
            state._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"session_id", "token", "user_id", "expires_in"})
        self.assertEqual(body["user_id"], "u1")
        self.assertGreaterEqual(len(body["token"]), 32)
        self.assertNotIn(body["token"], runtime.session_tokens)
        self.assertEqual(
            runtime.session_tokens[chat._token_hash(body["token"])],
            body["session_id"],
        )

    def test_chat_accepts_the_issued_browser_token_over_http(self):
        dataset = _dataset()
        index = ROOT / "tmp" / "backend-test-http-chat-index.sqlite3"
        build_index(dataset, index)
        client = _FakeChat()
        agent = RAGAgent(index, dataset["experts"], client)
        runtime = state.Runtime(dataset, client, agent, index_path=index)
        session_id, token = chat._create_chat_session(runtime)
        with patch.object(chat, "_get_runtime", return_value=runtime):
            response = self.client.post(
                "/chat",
                json={
                    "expert_id": "clinician",
                    "message": "Как кофеин влияет на сон?",
                    "session_id": session_id,
                },
                headers={"Authorization": "Bearer " + token},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["session_id"], session_id)
        self.assertEqual(body["text"], "Ответ.\n\nВторой абзац.")
        self.assertFalse(body["refused"])
        self.assertEqual(len(runtime.sessions[session_id]), 2)


class QuotaAndCacheTests(unittest.TestCase):
    def setUp(self):
        with cache._run_cache_lock:
            cache._run_cache.clear()
            cache._run_cache_sizes.clear()

    def _run_twice(self, agent, dataset, first_code="code-1", second_code="code-2"):
        runtime = state.Runtime(dataset, object(), agent)
        original_runtime = state._runtime
        builds = []

        def build(loaded, deadline, index_key=None):
            builds.append(loaded)
            return runtime

        try:
            with patch.object(organizer, "_fetch_dataset", return_value=dataset) as fetch, \
                    patch.object(state, "_build_runtime", side_effect=build):
                first = routes_run.run(
                    models.RunRequest(case=4, dataset_code=first_code),
                    authorization="Bearer test-token",
                )
                second = routes_run.run(
                    models.RunRequest(case=4, dataset_code=second_code),
                    authorization="Bearer test-token",
                )
        finally:
            state._runtime = original_runtime
        return first, second, fetch, builds

    def test_successful_run_is_cached_across_dataset_codes(self):
        dataset = _dataset()
        agent = _FakeAgent()
        first, second, fetch, builds = self._run_twice(agent, dataset)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        # Identical content reuses the active runtime: no index rebuild and no
        # browser-session reset on a same-content cache hit.
        self.assertEqual(len(builds), 1)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(json.loads(first.body), json.loads(second.body))
        self.assertEqual(agent.calls, ["clinician", "biohacker", "nutritionist"])
        with cache._run_cache_lock:
            self.assertEqual(len(cache._run_cache), 1)

    def test_cache_hit_activates_the_requested_dataset(self):
        dataset_a = _dataset()
        dataset_b = json.loads(json.dumps(dataset_a))
        dataset_b["articles"][0]["title"] = "Evening light"
        for request in dataset_b["requests"]:
            request["utterance"]["text"] = "evening light"
        agent_a, agent_b = _FakeAgent(), _FakeAgent()
        runtime_a = state.Runtime(dataset_a, object(), agent_a)
        runtime_b = state.Runtime(dataset_b, object(), agent_b)
        original_runtime = state._runtime

        def build(loaded, deadline, index_key=None):
            return runtime_a if loaded is dataset_a else runtime_b

        try:
            with patch.object(organizer, "_fetch_dataset",
                              side_effect=[dataset_a, dataset_b, dataset_a]), \
                    patch.object(state, "_build_runtime", side_effect=build):
                first = routes_run.run(models.RunRequest(case=4, dataset_code="a"),
                                 authorization="Bearer test-token")
                second = routes_run.run(models.RunRequest(case=4, dataset_code="b"),
                                  authorization="Bearer test-token")
                third = routes_run.run(models.RunRequest(case=4, dataset_code="a"),
                                 authorization="Bearer test-token")
                active = state._runtime
        finally:
            state._runtime = original_runtime
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(json.loads(third.body), json.loads(first.body))
        self.assertIs(active, runtime_a)
        self.assertEqual(agent_a.calls, ["clinician", "biohacker", "nutritionist"])
        self.assertEqual(agent_b.calls, ["clinician", "biohacker", "nutritionist"])

    def test_partial_run_is_not_cached(self):
        dataset = _dataset()
        agent = _FakeAgent(fail_for={"biohacker", "nutritionist"})
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls.count("clinician"), 2)
        self.assertEqual(agent.calls.count("biohacker"), 2)
        self.assertEqual(agent.calls.count("nutritionist"), 2)
        with cache._run_cache_lock:
            self.assertEqual(cache._run_cache, {})

    def test_retrieval_fallback_run_is_not_cached(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine"
        agent = _FakeAgent(prepare_error="index unavailable")
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls.count("clinician"), 2)
        self.assertEqual(agent.calls.count("biohacker"), 2)
        self.assertEqual(agent.calls.count("nutritionist"), 2)
        with cache._run_cache_lock:
            self.assertEqual(cache._run_cache, {})

    def test_unsupported_retrieval_run_is_not_cached(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "stock market volatility"
        agent = _FakeAgent(prepare_error="index unavailable")
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls, [])
        with cache._run_cache_lock:
            self.assertEqual(cache._run_cache, {})

    def test_cache_evicts_least_recently_used(self):
        with cache._run_cache_lock:
            cache._run_cache.clear()
        for key in ("a", "b", "c", "d"):
            cache._cache_store(key, {"requests": [key]})
        self.assertEqual(cache._cache_lookup("a"), {"requests": ["a"]})
        cache._cache_store("e", {"requests": ["e"]})
        self.assertIsNone(cache._cache_lookup("b"))
        self.assertIsNotNone(cache._cache_lookup("a"))
        self.assertIsNotNone(cache._cache_lookup("e"))

    def test_global_llm_client_is_created_once(self):
        original = state._llm_client
        state._llm_client = None
        client = object()
        try:
            with patch.object(ChatClient, "from_env", return_value=client) as factory:
                first = state._get_llm_client()
                second = state._get_llm_client()
        finally:
            state._llm_client = original
        self.assertIs(first, client)
        self.assertIs(second, client)
        self.assertEqual(factory.call_count, 1)

    def test_cache_key_ignores_meta_and_reference(self):
        dataset = _dataset()
        baseline = cache._run_cache_key(dataset)
        dataset["reference"] = {"requests": [{"record_id": "r_clinician"}]}
        dataset["meta"]["phase"] = "online"
        self.assertEqual(cache._run_cache_key(dataset), baseline)
        dataset["chat_history"].append({"role": "user", "text": "extra"})
        self.assertNotEqual(cache._run_cache_key(dataset), baseline)

    def test_cache_key_changes_with_the_model(self):
        dataset = _dataset()
        with patch.dict(os.environ, {"LLM_MODEL": "model-a"}):
            baseline = cache._run_cache_key(dataset)
        with patch.dict(os.environ, {"LLM_MODEL": "model-b"}):
            self.assertNotEqual(cache._run_cache_key(dataset), baseline)


class SharedEvidenceTests(unittest.TestCase):
    def test_evidence_is_selected_once_before_any_expert_answers(self):
        class TrackingAgent(_FakeAgent):
            def prepare_group(self, members, *, deadline=None):
                self.calls.append("retrieve")
                return super().prepare_group(members, deadline=deadline)

            def answer(self, question, expert_id, *, shared_evidence=None, **kwargs):
                if shared_evidence != [{"article_id": "art_01", "section_id": "s001"}]:
                    raise AssertionError("Shared evidence not passed")
                return super().answer(question, expert_id, **kwargs)

        agent = TrackingAgent()
        dataset = _dataset()
        dataset["requests"][1]["utterance"]["text"] = "A different wording"
        result = answering._answer_requests(agent, dataset, time.monotonic() + 60, answering.RunContext(retry_delays=()))
        self.assertEqual(agent.calls, ["retrieve", "clinician", "biohacker", "nutritionist"])
        self.assertTrue(all(answer["text"].startswith("Answer") for answer in result["requests"]))


class AuditFixTests(unittest.TestCase):
    """Regression coverage for the backend audit findings H2-H4 and M1-M10."""

    def setUp(self):
        with cache._run_cache_lock:
            cache._run_cache.clear()
            cache._run_cache_sizes.clear()
        self.original_runtime = state._runtime

    def tearDown(self):
        state._runtime = self.original_runtime

    def test_same_content_run_reuses_runtime_and_preserves_browser_session(self):
        dataset = _dataset()
        agent = _FakeAgent()
        runtime = state.Runtime(dataset, object(), agent)
        builds = []

        def build(loaded, deadline, index_key=None):
            builds.append(loaded)
            return runtime

        with patch.object(organizer, "_fetch_dataset", return_value=dataset), \
                patch.object(state, "_build_runtime", side_effect=build):
            first = routes_run.run(
                models.RunRequest(case=4, dataset_code="code-1"),
                authorization="Bearer test-token",
            )
            session_id, authorization = _chat_credential(runtime)
            runtime.sessions[session_id].append({
                "message_id": "m1", "user_id": "u1", "role": "user",
                "at": "2026-11-17T12:00:00+03:00", "text": "keep this history",
            })
            second = routes_run.run(
                models.RunRequest(case=4, dataset_code="code-2"),
                authorization="Bearer test-token",
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(builds), 1)
        self.assertIs(state._runtime, runtime)
        resolved_session, owner = chat._resolve_browser_session(runtime, authorization)
        self.assertEqual((resolved_session, owner), (session_id, "u1"))
        self.assertEqual(len(runtime.sessions[session_id]), 1)
        self.assertEqual(runtime.sessions[session_id][0]["text"], "keep this history")

    def test_failed_candidate_does_not_replace_active_runtime(self):
        dataset_a = _dataset()
        dataset_b = json.loads(json.dumps(dataset_a))
        dataset_b["articles"][0]["title"] = "Evening light"
        runtime_a = state.Runtime(dataset_a, object(), _FakeAgent())
        candidate_index = state._index_path_for(cache._run_cache_key(dataset_b))
        candidate_index.parent.mkdir(parents=True, exist_ok=True)
        candidate_index.write_bytes(b"candidate")
        runtime_b = state.Runtime(
            dataset_b, object(), _FakeAgent(), index_path=candidate_index
        )
        original_answer_requests = answering._answer_requests

        def answer_requests(agent, dataset, deadline, context):
            if dataset is dataset_b:
                raise RuntimeError("fatal candidate failure")
            return original_answer_requests(agent, dataset, deadline, context)

        try:
            with patch.object(organizer, "_fetch_dataset", side_effect=[dataset_a, dataset_b]), \
                    patch.object(state, "_build_runtime", side_effect=[runtime_a, runtime_b]), \
                    patch.object(answering, "_answer_requests", side_effect=answer_requests):
                first = routes_run.run(
                    models.RunRequest(case=4, dataset_code="a"),
                    authorization="Bearer test-token",
                )
                with self.assertRaises(HTTPException) as raised:
                    routes_run.run(
                        models.RunRequest(case=4, dataset_code="b"),
                        authorization="Bearer test-token",
                    )
        finally:
            candidate_index.unlink(missing_ok=True)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIs(state._runtime, runtime_a)
        self.assertFalse(candidate_index.exists())

    def test_chat_allowance_is_rechecked_at_each_proxy_call(self):
        response = MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 1},
        }).encode()
        client = ChatClient(
            "https://proxy.test/v1", "SECRET", "test-model",
            max_calls=10, min_interval=0,
        )
        client.opener = MagicMock()
        client.opener.open.return_value.__enter__.return_value = response
        client.complete([{"role": "user", "content": "x"}], call_ceiling=1)
        with self.assertRaises(BudgetError):
            client.complete([{"role": "user", "content": "x"}], call_ceiling=1)

        token_limited = ChatClient(
            "https://proxy.test/v1", "SECRET", "test-model",
            max_calls=10, min_interval=0,
        )
        token_limited.opener = MagicMock()
        token_limited.opener.open.return_value.__enter__.return_value = response
        with self.assertRaises(BudgetError):
            token_limited.complete([{"role": "user", "content": "x"}], token_ceiling=50)

    def test_wrong_record_id_is_rejected_before_cache_or_response(self):
        class WrongRecordAgent(_FakeAgent):
            def answer(self, question, expert_id, **kwargs):
                result = super().answer(question, expert_id, **kwargs)
                result["answer"]["record_id"] = "wrong"
                return result

        dataset = _dataset()
        context = answering.RunContext(retry_delays=())
        result = answering._answer_requests(
            WrongRecordAgent(), dataset, time.monotonic() + 60, context
        )
        self.assertEqual(
            [answer["record_id"] for answer in result["requests"]],
            [request["record_id"] for request in dataset["requests"]],
        )
        self.assertEqual(context.generated, 0)
        self.assertEqual(context.validation_failures, 3)
        self.assertEqual(context.fallbacks, 3)

    def test_duplicate_input_record_ids_are_rejected(self):
        dataset = _dataset()
        dataset["requests"].append(json.loads(json.dumps(dataset["requests"][0])))
        with self.assertRaises(ValueError):
            answering._answer_requests(
                _FakeAgent(), dataset, time.monotonic() + 60,
                answering.RunContext(retry_delays=()),
            )

    def test_chat_failure_before_commit_leaves_history_unchanged(self):
        runtime = state.Runtime(_dataset(), _FakeChat(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        state._runtime = runtime

        class MissingRefusalAgent:
            def __init__(self, *args, **kwargs):
                pass

            def answer(self, *args, **kwargs):
                return {"answer": {
                    "record_id": "x",
                    "text": "hello [art_01 / s001]",
                    "citations": [{"article_id": "art_01", "section_id": "s001"}],
                }}

        with patch.object(chat, "RAGAgent", MissingRefusalAgent):
            with self.assertRaises(HTTPException) as raised:
                chat.chat(
                    models.ChatRequest(expert_id="clinician", message="hello"),
                    authorization=authorization,
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(runtime.sessions[session_id], [])
        self.assertEqual(runtime.inflight_turns, {})
        self.assertEqual(runtime.busy_sessions, set())

        class MissingSourceAgent(MissingRefusalAgent):
            def answer(self, *args, **kwargs):
                return {"answer": {
                    "record_id": "x",
                    "text": "hello [art_99 / s999]",
                    "refused": False,
                    "citations": [{"article_id": "art_99", "section_id": "s999"}],
                }}

        with patch.object(chat, "RAGAgent", MissingSourceAgent):
            with self.assertRaises(HTTPException):
                chat.chat(
                    models.ChatRequest(expert_id="clinician", message="hello"),
                    authorization=authorization,
                )
        self.assertEqual(runtime.sessions[session_id], [])

    def test_second_turn_for_a_busy_session_is_rejected(self):
        runtime = state.Runtime(_dataset(), _FakeChat(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        state._runtime = runtime
        runtime.busy_sessions.add(session_id)
        with self.assertRaises(HTTPException) as raised:
            chat.chat(
                models.ChatRequest(expert_id="clinician", message="hello"),
                authorization=authorization,
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(runtime.inflight_turns, {})

    def test_completed_turn_id_replays_without_second_generation(self):
        dataset = _dataset()
        index = ROOT / "tmp" / "backend-test-idempotency-index.sqlite3"
        build_index(dataset, index)

        class CountingChat(_FakeChat):
            def __init__(self):
                self.calls = 0
                self.total_tokens = 0

            def complete(self, messages, **kwargs):
                self.calls += 1
                return super().complete(messages, **kwargs)

        client = CountingChat()
        agent = RAGAgent(index, dataset["experts"], client)
        runtime = state.Runtime(dataset, client, agent, index_path=index)
        session_id, authorization = _chat_credential(runtime)
        state._runtime = runtime
        request = models.ChatRequest(
            expert_id="clinician",
            message="Как кофеин влияет на сон?",
            session_id=session_id,
            turn_id="turn-1",
        )
        first = chat.chat(request, authorization=authorization)
        second = chat.chat(request, authorization=authorization)
        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(runtime.sessions[session_id]), 2)
        self.assertIn((session_id, "turn-1"), runtime.completed_turns)
        with self.assertRaises(HTTPException) as raised:
            chat.chat(
                models.ChatRequest(
                    expert_id="clinician",
                    message="A different question",
                    session_id=session_id,
                    turn_id="turn-1",
                ),
                authorization=authorization,
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(len(runtime.sessions[session_id]), 2)

    def test_citation_generation_and_stale_source(self):
        runtime = state.Runtime(_dataset(), _FakeChat(), _FakeAgent())
        runtime.set_dataset_key("a" * 64)
        enriched = chat._enrich_citations(
            runtime, [{"article_id": "art_01", "section_id": "s001"}]
        )
        self.assertEqual(enriched[0]["dataset_generation"], runtime.generation)
        state._runtime = runtime
        with self.assertRaises(HTTPException) as raised:
            catalog.source("art_01", "s001", generation="old-generation")
        self.assertEqual(raised.exception.status_code, 409)
        body = catalog.source("art_01", "s001", generation=runtime.generation)
        self.assertEqual(body["dataset_generation"], runtime.generation)
        self.assertEqual(body["text"], "Caffeine can delay sleep.")

    def test_invalid_dataset_does_not_replace_active_runtime(self):
        active = state.Runtime(_dataset(), object(), _FakeAgent())
        state._runtime = active
        invalid = _dataset()
        del invalid["articles"][0]["title"]
        with patch.object(organizer, "_fetch_dataset", return_value=invalid):
            with self.assertRaises(HTTPException) as raised:
                routes_run.run(
                    models.RunRequest(case=4, dataset_code="invalid"),
                    authorization="Bearer test-token",
                )
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIs(state._runtime, active)
        with cache._run_cache_lock:
            self.assertEqual(cache._run_cache, {})

    def test_cache_entry_size_is_bounded(self):
        with patch.object(cache, "RUN_CACHE_MAX_ENTRY_BYTES", 16), \
                patch.object(cache, "RUN_CACHE_MAX_BYTES", 64):
            cache._cache_store("big", {"requests": [{"text": "x" * 100}]})
            self.assertIsNone(cache._cache_lookup("big"))
            cache._cache_store("small", {"requests": []})
            self.assertIsNotNone(cache._cache_lookup("small"))

    def test_session_issue_budget_is_separate_from_chat_budget(self):
        runtime = state.Runtime(_dataset(), object(), _FakeAgent())
        for _ in range(chat.CHAT_SESSION_ISSUE_LIMIT):
            self.assertTrue(chat._session_issue_allowed(runtime))
        self.assertFalse(chat._session_issue_allowed(runtime))
        # Issuance did not consume the chat bucket.
        self.assertTrue(chat._chat_global_rate_allowed(runtime))

    def test_session_capacity_keeps_fresh_browsers(self):
        with patch.object(chat, "CHAT_MAX_SESSIONS", 1), \
                patch.object(chat, "CHAT_SESSION_EVICT_GRACE_SECONDS", 300):
            runtime = state.Runtime(_dataset(), object(), _FakeAgent())
            first, _ = chat._create_chat_session(runtime)
            with self.assertRaises(SessionCapacityError):
                chat._create_chat_session(runtime)
            self.assertIn(first, runtime.sessions)
            runtime.session_touched[first] = time.monotonic() - 301
            second, _ = chat._create_chat_session(runtime)
            self.assertNotIn(first, runtime.sessions)
            self.assertIn(second, runtime.sessions)

    def test_request_id_header_is_returned(self):
        client = TestClient(backend_app)
        response = client.get("/health", headers={"X-Request-ID": "audit-123"})
        self.assertEqual(response.headers["X-Request-ID"], "audit-123")
        generated = client.get("/health", headers={"X-Request-ID": "not valid"})
        self.assertRegex(generated.headers["X-Request-ID"], r"^[0-9a-f]{16}$")

    def test_terminal_http_error_stops_generation(self):
        terminal = ChatTerminalError("Chat proxy HTTP 401", status_code=401)
        transient = ChatTransientError(
            "Chat proxy HTTP 429", status_code=429, retryable=True,
        )
        self.assertFalse(answering._retryable(terminal))
        self.assertTrue(answering._retryable(transient))
        context = answering.RunContext(retry_delays=(0, 0))
        answering._note_failure(terminal, context)
        self.assertTrue(context.skip_llm)
        self.assertEqual(context.transient_failures, 0)


if __name__ == "__main__":
    unittest.main()
