"""Contract-level tests for the /run shape. No network and no LLM calls."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

import main
from rag_agent import RAGAgent
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
    session_id, token = main._create_chat_session(runtime)
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
        with main._run_cache_lock:
            main._run_cache.clear()

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
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 300, main.RunContext(retry_delays=())
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
        result = main._answer_requests(
            _FakeAgent(), dataset, time.monotonic() + 60, main.RunContext(retry_delays=())
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
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=())
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
        context = main.RunContext(retry_delays=(), cooldown_base=0)
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, context)
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
        context = main.RunContext(retry_delays=(0,), cooldown_base=0)
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(agent.calls, ["clinician", "clinician", "biohacker", "nutritionist"])
        self.assertEqual([item["refused"] for item in result["requests"]], [True, False, False])
        self.assertEqual(context.transient_failures, 1)
        self.assertEqual(context.consecutive_transient, 0)
        self.assertEqual(context.generated, 2)

    def test_local_budget_exhaustion_is_not_retried(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Local LLM call/token budget exhausted",
        )
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=(0, 0))
        )
        self.assertEqual(len(agent.calls), 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))

    def test_auth_failure_stops_remaining_calls(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Chat proxy HTTP 401; no automatic retry",
        )
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=(0, 0))
        )
        self.assertEqual(len(agent.calls), 1)
        self.assertTrue(all(item["refused"] for item in result["requests"]))

    def test_protocol_failure_stops_remaining_calls(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Proxy response lacks valid token usage; stop to protect budget",
        )
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=(0, 0))
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
        evidence = main._fallback_evidence(dataset, dataset["requests"], agent,
                                           time.monotonic() + 60)
        self.assertEqual(evidence[0]["section_id"], "s001")

    def test_keyword_fallback_never_cites_an_unrelated_section(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "stock market volatility"
        agent = _FakeAgent(prepare_error="index unavailable")
        evidence = main._fallback_evidence(dataset, dataset["requests"], agent,
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
            main._read_limited(Response(b"x" * 10), 8, time.monotonic() + 60)
        self.assertEqual(
            main._read_limited(Response(b"x" * 8), 8, time.monotonic() + 60), b"x" * 8
        )

    def test_retrieval_failure_does_not_abort_the_run(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine"
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            prepare_error="index unavailable",
        )
        context = main.RunContext(retry_delays=())
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, context)
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
        context = main.RunContext(retry_delays=())
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(agent.calls, [])
        self.assertEqual(context.unsupported_fallbacks, 3)
        self.assertEqual(context.fallbacks, 3)
        self.assertTrue(all(item["refused"] for item in result["requests"]))
        self.assertTrue(all(item["citations"] for item in result["requests"]))

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
        context = main.RunContext(retry_delays=())
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, context)
        self.assertEqual(context.retrieval_failures, 1)
        self.assertEqual(result["requests"][0]["citations"],
                         [{"article_id": "art_01", "section_id": "s001"}])

    def test_fallback_citation_exists_in_dataset(self):
        evidence = [{"article_id": "art_02", "section_id": "s003"}]
        answer = main._fallback_answer(_dataset()["requests"][0], evidence)
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
        answer = main._fallback_answer(request, evidence, topical=True)
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
        runtime = main.Runtime(dataset, client, agent)
        session_id, authorization = _chat_credential(runtime)
        original_index, original_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = index, runtime
        try:
            response = main.chat(
                main.ChatRequest(
                    expert_id="clinician",
                    message="Как кофеин влияет на сон?",
                    session_id=session_id,
                ),
                authorization=authorization,
            )
        finally:
            main.INDEX_PATH, main._runtime = original_index, original_runtime
        self.assertEqual(response["session_id"], session_id)
        self.assertFalse(response["refused"])
        self.assertEqual(response["text"], "Ответ.\n\nВторой абзац.")
        self.assertEqual(set(response["citations"][0]), {
            "article_id", "section_id", "article_title", "section_title", "year",
        })
        self.assertEqual(len(runtime.sessions[session_id]), 2)
        self.assertEqual(runtime.sessions[session_id][0]["role"], "user")
        self.assertEqual(runtime.sessions[session_id][1]["role"], "agent")
        self.assertEqual(runtime.sessions[session_id][1]["text"], response["text"])

    def test_chat_does_not_accept_a_client_selected_user(self):
        with self.assertRaises(PydanticValidationError):
            main.ChatRequest(
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
        runtime = main.Runtime(dataset, client, agent)
        session_id, authorization = _chat_credential(runtime)
        original_index, original_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = index, runtime
        try:
            main.chat(main.ChatRequest(
                expert_id="clinician",
                message="Как раньше, кофе влияет на сон?",
                session_id=session_id,
            ), authorization=authorization)
        finally:
            main.INDEX_PATH, main._runtime = original_index, original_runtime
        self.assertTrue(client.payloads)
        self.assertNotIn("PRIVATE-DATASET-HISTORY", json.dumps(client.payloads, ensure_ascii=False))

    def test_chat_session_store_is_bounded(self):
        session = [
            {"role": "user" if index % 2 == 0 else "agent", "text": "x" * 100}
            for index in range(main.CHAT_MAX_MESSAGES + 10)
        ]
        main._trim_chat_session(session)
        self.assertEqual(len(session), main.CHAT_MAX_MESSAGES)
        session = [{"role": "user", "text": "x" * 1000} for _ in range(30)]
        main._trim_chat_session(session)
        self.assertLessEqual(sum(len(message["text"]) for message in session), main.CHAT_MAX_CHARS)

    def test_chat_rate_limit_is_bounded(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        for _ in range(main.CHAT_RATE_LIMIT):
            self.assertTrue(main._chat_session_rate_allowed(runtime, "session"))
        self.assertFalse(main._chat_session_rate_allowed(runtime, "session"))

    def test_chat_global_rate_limit_is_bounded_across_sessions(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        for _ in range(main.CHAT_GLOBAL_RATE_LIMIT):
            self.assertTrue(main._chat_global_rate_allowed(runtime))
        self.assertFalse(main._chat_global_rate_allowed(runtime))

    def test_chat_budget_reserves_two_calls_per_request(self):
        dataset = _dataset()
        dataset["requests"] = dataset["requests"] * 20  # 60 requests
        runtime = main.Runtime(dataset, object(), _FakeAgent())

        class Client:
            calls = main.LLM_MAX_CALLS - main.RUN_RESERVED_CALLS
            total_tokens = 0

        runtime.client = Client()
        self.assertFalse(main._chat_budget_available(runtime))
        runtime.run_completed = True
        self.assertTrue(main._chat_budget_available(runtime))

    def test_chat_session_owner_comes_from_the_server_session(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        runtime.session_users[session_id] = "u2"
        resolved_session, owner = main._resolve_browser_session(runtime, authorization)
        self.assertEqual(resolved_session, session_id)
        self.assertEqual(owner, "u2")

    def test_chat_requires_a_browser_session_token(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        shared_index, shared_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(main.HTTPException) as raised:
                main.chat(main.ChatRequest(expert_id="clinician", message="hello"))
        finally:
            main.INDEX_PATH, main._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_rejects_a_session_id_from_another_browser(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        _, authorization = _chat_credential(runtime)
        shared_index, shared_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(main.HTTPException) as raised:
                main.chat(
                    main.ChatRequest(
                        expert_id="clinician", message="hello", session_id="guessed",
                    ),
                    authorization=authorization,
                )
        finally:
            main.INDEX_PATH, main._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 403)

    def test_chat_rejects_a_guessable_browser_token(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        session_id, _ = _chat_credential(runtime)
        shared_index, shared_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(main.HTTPException) as raised:
                main.chat(
                    main.ChatRequest(
                        expert_id="clinician", message="hello", session_id=session_id,
                    ),
                    authorization="Bearer guessed-token",
                )
        finally:
            main.INDEX_PATH, main._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_stores_only_a_hash_of_the_browser_token(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        token = authorization.removeprefix("Bearer ")
        self.assertNotIn(token, runtime.session_tokens)
        self.assertEqual(runtime.session_tokens[main._token_hash(token)], session_id)

    def test_new_browser_session_uses_the_server_selected_user(self):
        dataset = _dataset()
        dataset["users"].append({"user_id": "u2"})
        runtime = main.Runtime(dataset, object(), _FakeAgent())
        session_id, _ = _chat_credential(runtime)
        self.assertEqual(runtime.session_users[session_id], "u1")
        self.assertEqual(runtime.sessions[session_id], [])

    def test_browser_tokens_do_not_survive_a_dataset_switch(self):
        runtime_a = main.Runtime(_dataset(), object(), _FakeAgent())
        dataset_b = _dataset()
        dataset_b["articles"][0]["article_id"] = "art_09"
        runtime_b = main.Runtime(dataset_b, object(), _FakeAgent())
        _, authorization = _chat_credential(runtime_a)
        shared_index, shared_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime_b
        try:
            with self.assertRaises(main.HTTPException) as raised:
                main.chat(
                    main.ChatRequest(expert_id="clinician", message="hello"),
                    authorization=authorization,
                )
        finally:
            main.INDEX_PATH, main._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_rejects_a_blank_message(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        _, authorization = _chat_credential(runtime)
        shared_index, shared_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = Path("/tmp/does-not-exist.sqlite3"), runtime
        try:
            with self.assertRaises(main.HTTPException) as raised:
                main.chat(main.ChatRequest(expert_id="clinician", message="   "),
                          authorization=authorization)
        finally:
            main.INDEX_PATH, main._runtime = shared_index, shared_runtime
        self.assertEqual(raised.exception.status_code, 400)

    def test_expired_chat_sessions_are_dropped(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        session_id, authorization = _chat_credential(runtime)
        runtime.session_touched[session_id] = (
            time.monotonic() - main.CHAT_SESSION_TTL_SECONDS - 1
        )
        runtime.chat_times[session_id] = [time.monotonic()]
        with self.assertRaises(main.HTTPException):
            main._resolve_browser_session(runtime, authorization)
        self.assertNotIn(session_id, runtime.sessions)
        self.assertNotIn(session_id, runtime.issued_sessions)
        self.assertNotIn(session_id, runtime.chat_times)
        self.assertEqual(runtime.session_tokens, {})

    def test_run_route_returns_exact_body(self):
        dataset = _dataset()
        agent = _FakeAgent()
        runtime = main.Runtime(dataset, object(), agent)
        original_runtime = main._runtime
        try:
            with patch.object(main, "_fetch_dataset", return_value=dataset), \
                    patch.object(main, "_build_runtime", return_value=runtime):
                response = main.run(
                    main.RunRequest(case=4, dataset_code="test-code"),
                    authorization="Bearer test-token",
                )
        finally:
            main._runtime = original_runtime
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
        runtime = main.Runtime(dataset, object(), agent)
        original_runtime = main._runtime
        try:
            with patch.object(main, "_fetch_dataset", return_value=dataset), \
                    patch.object(main, "_build_runtime", return_value=runtime):
                response = main.run(
                    main.RunRequest(case=4, dataset_code="test-code"),
                    authorization="Bearer test-token",
                )
        finally:
            main._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(len(body["requests"]), 3)
        self.assertTrue(all(answer["refused"] for answer in body["requests"]))
        self.assertTrue(all(answer["citations"] for answer in body["requests"]))


class HttpHandlerTests(unittest.TestCase):
    """Exercise the actual FastAPI handlers, not only the route functions."""

    def setUp(self):
        self.client = TestClient(main.app)
        with main._run_cache_lock:
            main._run_cache.clear()

    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"team": main.TEAM, "status": "ok"})

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
        runtime = main.Runtime(dataset, object(), _FakeAgent())
        original_runtime = main._runtime
        try:
            with patch.object(main, "_fetch_dataset", return_value=dataset), \
                    patch.object(main, "_build_runtime", return_value=runtime):
                response = self.client.post(
                    "/run",
                    json={"case": 4, "dataset_code": "test-code"},
                    headers={"Authorization": "Bearer test-token"},
                )
        finally:
            main._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"requests"})
        self.assertEqual(len(body["requests"]), 3)
        for answer in body["requests"]:
            self.assertEqual(set(answer), {"record_id", "text", "refused", "citations"})

    def test_chat_requires_a_browser_token_over_http(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        with patch.object(main, "_get_runtime", return_value=runtime):
            response = self.client.post(
                "/chat", json={"expert_id": "clinician", "message": "hello"},
            )
        self.assertEqual(response.status_code, 401)

    def test_session_issues_an_opaque_token(self):
        runtime = main.Runtime(_dataset(), object(), _FakeAgent())
        original_runtime = main._runtime
        main._runtime = runtime
        try:
            response = self.client.post("/session")
        finally:
            main._runtime = original_runtime
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"session_id", "token", "user_id", "expires_in"})
        self.assertEqual(body["user_id"], "u1")
        self.assertGreaterEqual(len(body["token"]), 32)
        self.assertNotIn(body["token"], runtime.session_tokens)
        self.assertEqual(
            runtime.session_tokens[main._token_hash(body["token"])],
            body["session_id"],
        )

    def test_chat_accepts_the_issued_browser_token_over_http(self):
        dataset = _dataset()
        index = ROOT / "tmp" / "backend-test-http-chat-index.sqlite3"
        build_index(dataset, index)
        client = _FakeChat()
        agent = RAGAgent(index, dataset["experts"], client)
        runtime = main.Runtime(dataset, client, agent, index_path=index)
        session_id, token = main._create_chat_session(runtime)
        with patch.object(main, "_get_runtime", return_value=runtime):
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
        with main._run_cache_lock:
            main._run_cache.clear()

    def _run_twice(self, agent, dataset, first_code="code-1", second_code="code-2"):
        runtime = main.Runtime(dataset, object(), agent)
        original_runtime = main._runtime
        builds = []

        def build(loaded, deadline, index_key=None):
            builds.append(loaded)
            return runtime

        try:
            with patch.object(main, "_fetch_dataset", return_value=dataset) as fetch, \
                    patch.object(main, "_build_runtime", side_effect=build):
                first = main.run(
                    main.RunRequest(case=4, dataset_code=first_code),
                    authorization="Bearer test-token",
                )
                second = main.run(
                    main.RunRequest(case=4, dataset_code=second_code),
                    authorization="Bearer test-token",
                )
        finally:
            main._runtime = original_runtime
        return first, second, fetch, builds

    def test_successful_run_is_cached_across_dataset_codes(self):
        dataset = _dataset()
        agent = _FakeAgent()
        first, second, fetch, builds = self._run_twice(agent, dataset)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        # The runtime is rebuilt/activated even on a cache hit so the demo
        # endpoints never point at a previously submitted dataset.
        self.assertEqual(len(builds), 2)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(json.loads(first.body), json.loads(second.body))
        self.assertEqual(agent.calls, ["clinician", "biohacker", "nutritionist"])
        with main._run_cache_lock:
            self.assertEqual(len(main._run_cache), 1)

    def test_cache_hit_activates_the_requested_dataset(self):
        dataset_a = _dataset()
        dataset_b = json.loads(json.dumps(dataset_a))
        dataset_b["articles"][0]["article_id"] = "art_09"
        for request in dataset_b["requests"]:
            request["utterance"]["text"] = "evening light"
        agent_a, agent_b = _FakeAgent(), _FakeAgent()
        runtime_a = main.Runtime(dataset_a, object(), agent_a)
        runtime_b = main.Runtime(dataset_b, object(), agent_b)
        original_runtime = main._runtime

        def build(loaded, deadline, index_key=None):
            return runtime_a if loaded is dataset_a else runtime_b

        try:
            with patch.object(main, "_fetch_dataset",
                              side_effect=[dataset_a, dataset_b, dataset_a]), \
                    patch.object(main, "_build_runtime", side_effect=build):
                first = main.run(main.RunRequest(case=4, dataset_code="a"),
                                 authorization="Bearer test-token")
                second = main.run(main.RunRequest(case=4, dataset_code="b"),
                                  authorization="Bearer test-token")
                third = main.run(main.RunRequest(case=4, dataset_code="a"),
                                 authorization="Bearer test-token")
                active = main._runtime
        finally:
            main._runtime = original_runtime
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
        with main._run_cache_lock:
            self.assertEqual(main._run_cache, {})

    def test_retrieval_fallback_run_is_not_cached(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "caffeine"
        agent = _FakeAgent(prepare_error="index unavailable")
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls.count("clinician"), 2)
        self.assertEqual(agent.calls.count("biohacker"), 2)
        self.assertEqual(agent.calls.count("nutritionist"), 2)
        with main._run_cache_lock:
            self.assertEqual(main._run_cache, {})

    def test_unsupported_retrieval_run_is_not_cached(self):
        dataset = _dataset()
        for request in dataset["requests"]:
            request["utterance"]["text"] = "stock market volatility"
        agent = _FakeAgent(prepare_error="index unavailable")
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls, [])
        with main._run_cache_lock:
            self.assertEqual(main._run_cache, {})

    def test_cache_evicts_least_recently_used(self):
        with main._run_cache_lock:
            main._run_cache.clear()
        for key in ("a", "b", "c", "d"):
            main._cache_store(key, {"requests": [key]})
        self.assertEqual(main._cache_lookup("a"), {"requests": ["a"]})
        main._cache_store("e", {"requests": ["e"]})
        self.assertIsNone(main._cache_lookup("b"))
        self.assertIsNotNone(main._cache_lookup("a"))
        self.assertIsNotNone(main._cache_lookup("e"))

    def test_global_llm_client_is_created_once(self):
        original = main._llm_client
        main._llm_client = None
        client = object()
        try:
            with patch.object(main.ChatClient, "from_env", return_value=client) as factory:
                first = main._get_llm_client()
                second = main._get_llm_client()
        finally:
            main._llm_client = original
        self.assertIs(first, client)
        self.assertIs(second, client)
        self.assertEqual(factory.call_count, 1)

    def test_cache_key_ignores_meta_and_reference(self):
        dataset = _dataset()
        baseline = main._run_cache_key(dataset)
        dataset["reference"] = {"requests": [{"record_id": "r_clinician"}]}
        dataset["meta"]["phase"] = "online"
        self.assertEqual(main._run_cache_key(dataset), baseline)
        dataset["chat_history"].append({"role": "user", "text": "extra"})
        self.assertNotEqual(main._run_cache_key(dataset), baseline)

    def test_cache_key_changes_with_the_model(self):
        dataset = _dataset()
        with patch.dict(os.environ, {"LLM_MODEL": "model-a"}):
            baseline = main._run_cache_key(dataset)
        with patch.dict(os.environ, {"LLM_MODEL": "model-b"}):
            self.assertNotEqual(main._run_cache_key(dataset), baseline)


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
        result = main._answer_requests(agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=()))
        self.assertEqual(agent.calls, ["retrieve", "clinician", "biohacker", "nutritionist"])
        self.assertTrue(all(answer["text"].startswith("Answer") for answer in result["requests"]))


if __name__ == "__main__":
    unittest.main()
