"""Contract-level tests for the /run shape. No network and no LLM calls."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _FakeAgent:
    def __init__(self, fail_for: set[str] | None = None, error: str = "boom") -> None:
        self.fail_for = fail_for or set()
        self.error = error
        self.calls: list[str] = []

    def prepare_group(self, members, *, deadline=None):
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
            "content": json.dumps({"decision": "answer", "text": "Ответ" + "".join(f"[{ref}]" for ref in refs) + ".\n\nВторой абзац.", "refs": refs}),
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
        self.assertFalse(result["requests"][1]["refused"])
        self.assertFalse(result["requests"][2]["refused"])
        self.assertEqual(
            [item["citations"] for item in result["requests"]],
            [[{"article_id": "art_01", "section_id": "s001"}]] * 3,
        )

    def test_rate_limit_stops_the_remaining_calls(self):
        dataset = _dataset()
        agent = _FakeAgent(
            fail_for={"clinician", "biohacker", "nutritionist"},
            error="Chat proxy HTTP 429; no automatic retry",
        )
        result = main._answer_requests(
            agent, dataset, time.monotonic() + 60, main.RunContext(retry_delays=())
        )
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(agent.calls, ["clinician"])
        self.assertTrue(all(not item["refused"] for item in result["requests"]))

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
        self.assertTrue(all(not item["refused"] for item in result["requests"]))

    def test_fallback_citation_exists_in_dataset(self):
        evidence = [{"article_id": "art_02", "section_id": "s003"}]
        answer = main._fallback_answer(_dataset()["requests"][0], evidence)
        self.assertFalse(answer["refused"])
        self.assertEqual(answer["citations"], evidence)
        self.assertIn("[art_02 / s003]", answer["text"])

    @unittest.skipUnless(DEV_DATASET.is_file(), "dev dataset not present")
    def test_chat_handler_with_fake_client(self):
        dataset = json.loads(DEV_DATASET.read_text(encoding="utf-8"))
        index = ROOT / "tmp" / "backend-test-chat-index.sqlite3"
        build_index(dataset, index)
        client = _FakeChat()
        agent = RAGAgent(index, dataset["experts"], client, history=dataset["chat_history"])
        runtime = main.Runtime(dataset, client, agent)
        original_index, original_runtime = main.INDEX_PATH, main._runtime
        main.INDEX_PATH, main._runtime = index, runtime
        try:
            response = main.chat(
                main.ChatRequest(
                    expert_id="clinician",
                    message="Как кофеин влияет на сон?",
                    session_id="test-session",
                )
            )
        finally:
            main.INDEX_PATH, main._runtime = original_index, original_runtime
        self.assertEqual(response["session_id"], "test-session")
        self.assertFalse(response["refused"])
        self.assertEqual(response["text"], "Ответ.\n\nВторой абзац.")
        self.assertEqual(set(response["citations"][0]), {
            "article_id", "section_id", "article_title", "section_title", "year",
        })
        self.assertEqual(len(runtime.sessions["test-session"]), 2)
        self.assertEqual(runtime.sessions["test-session"][0]["role"], "user")
        self.assertEqual(runtime.sessions["test-session"][1]["role"], "agent")
        self.assertEqual(runtime.sessions["test-session"][1]["text"], response["text"])

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


class QuotaAndCacheTests(unittest.TestCase):
    def setUp(self):
        with main._run_cache_lock:
            main._run_cache.clear()

    def _run_twice(self, agent, dataset, first_code="code-1", second_code="code-2"):
        runtime = main.Runtime(dataset, object(), agent)
        original_runtime = main._runtime
        builds = []

        def build(loaded, deadline):
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
        self.assertEqual(len(builds), 1)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(json.loads(first.body), json.loads(second.body))
        self.assertEqual(agent.calls, ["clinician", "biohacker", "nutritionist"])
        with main._run_cache_lock:
            self.assertEqual(len(main._run_cache), 1)

    def test_partial_run_is_not_cached(self):
        dataset = _dataset()
        agent = _FakeAgent(fail_for={"biohacker", "nutritionist"})
        self._run_twice(agent, dataset)
        self.assertEqual(agent.calls.count("clinician"), 2)
        self.assertEqual(agent.calls.count("biohacker"), 2)
        self.assertEqual(agent.calls.count("nutritionist"), 2)
        with main._run_cache_lock:
            self.assertEqual(main._run_cache, {})

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
