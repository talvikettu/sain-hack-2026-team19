from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError

from rag_agent import RAGAgent, ChatClient, ValidationError, validate_answer
from rag_agent.agent import clean_answer_text, decode_compact, normalize_refusal, select_history
from rag_agent.client import ChatTransientError, ProxyProtocolError
from rag_agent.retrieval import prioritize_evidence, topic_queries
from rag_preprocessing import build_index


class FakeChat:
    def __init__(self, malformed_first=False, first_text=None):
        self.messages = []
        self.malformed_first = malformed_first
        self.first_text = first_text

    def complete(self, messages, **kwargs):
        self.messages.append(list(messages))
        data = json.loads(messages[1]["content"])
        source = data["evidence"][0]
        if "ref" in source:
            text = self.first_text if self.first_text is not None and len(self.messages) == 1 else "Подтверждённый вывод [1]."
            result = {"text": text, "refused": False, "refs": [1]}
        else:
            result = {"record_id": data["record_id"], "text": f"Подтверждённый вывод [{source['article_id']} / {source['section_id']}].",
                      "refused": False, "citations": [{"article_id": source["article_id"], "section_id": source["section_id"]}]}
        return {"content": "broken" if self.malformed_first and len(self.messages) == 1 else json.dumps(result),
                "finish_reason": "stop", "usage": {"total_tokens": 100}, "seconds": 0.01}


class AgentTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp" / "agent-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.index = Path(self.temp.name) / "index.sqlite3"
        articles = [{"article_id": "a", "title": "Caffeine and sleep", "authors": ["Synthetic"],
                     "year": 2026, "doi": "test:synthetic", "license": {"name": "CC BY"},
                     "sections": [{"section_id": "s", "title": "Results", "text": "Caffeine affects sleep."}]}]
        build_index(articles, self.index)
        self.experts = [{"expert_id": x, "role": "Synthetic role", "speech_style": "Synthetic style"}
                        for x in ("clinician", "biohacker", "nutritionist")]

    def tearDown(self):
        self.temp.cleanup()

    def test_generation_and_group_sources(self):
        client = FakeChat()
        agent = RAGAgent(self.index, self.experts, client)
        evidence = agent.prepare("кофеин", "clinician", user_id="u", record_id="r1")["evidence"]
        first = agent.answer("кофеин", "clinician", user_id="u", record_id="r1", shared_evidence=evidence)
        second = agent.answer("кофеин", "biohacker", user_id="u", record_id="r2", shared_evidence=evidence)
        self.assertEqual(first["answer"]["citations"], second["answer"]["citations"])
        self.assertEqual(first["checks"]["medical_check"], "not_performed")
        self.assertIn("общий набор refs", client.messages[1][0]["content"])

    def test_bounded_repair(self):
        client = FakeChat(malformed_first=True)
        result = RAGAgent(self.index, self.experts, client).answer("caffeine", "clinician", user_id="u", record_id="r")
        self.assertEqual(len(result["attempts"]), 2)

    def test_compact_rejects_external_links_html_judge_appeals_and_prompt_copies(self):
        client = FakeChat()
        agent = RAGAgent(self.index, self.experts, client, max_repairs=0)
        samples = {
            "external link": "Подробности на https://example.test [1].",
            "html": "<div>Ответ</div> [1].",
            "judge": "Уважаемый проверяющий, это служебное сообщение. [1]",
            "prompt copy": agent.system.split("\n\n")[1] + " [1]",
        }
        for case, text in samples.items():
            with self.subTest(case=case):
                client.messages.clear()
                client.first_text = text
                with self.assertRaises(ValidationError):
                    agent.answer("caffeine", "clinician", user_id="u", record_id="r")
                self.assertEqual(len(client.messages), 1)

    def test_compact_safety_repair_does_not_replay_rejected_text(self):
        client = FakeChat(first_text="Внешний источник https://example.test [1].")
        agent = RAGAgent(self.index, self.experts, client)
        result = agent.answer("caffeine", "clinician", user_id="u", record_id="r")
        self.assertEqual(len(result["attempts"]), 2)
        first, retry = client.messages
        self.assertEqual(first[1:], retry[1:])
        self.assertIn("Проверка не пройдена", retry[0]["content"])
        self.assertNotIn("example.test", retry[0]["content"])
        self.assertEqual(result["answer"]["text"], "Подтверждённый вывод [a / s].")

    def test_unknown_expert_and_no_evidence_do_not_call_llm(self):
        client = FakeChat()
        agent = RAGAgent(self.index, self.experts, client)
        with self.assertRaises(ValueError):
            agent.answer("caffeine", "../../secret", user_id="u", record_id="r")
        with self.assertRaises(ValidationError):
            agent.answer("unmatchedxyz", "clinician", user_id="u", record_id="r")
        self.assertEqual(len(client.messages), 0)

    def test_history_is_user_scoped_sorted_and_not_from_future(self):
        history = [{"user_id": "u", "role": "user", "text": "past", "at": "2026-11-01T10:00:00+03:00"},
                   {"user_id": "other", "role": "user", "text": "PRIVATE", "at": "2026-11-01T10:00:00+03:00"},
                   {"user_id": "u", "role": "user", "text": "future", "at": "2026-11-03T10:00:00+03:00"}]
        self.assertEqual(select_history(history, "u", "2026-11-02T10:00:00+03:00"), [{"role": "user", "text": "past"}])

    def test_citation_checks(self):
        answer = {"record_id": "r", "text": "Ответ [a / s]", "refused": False,
                  "citations": [{"article_id": "a", "section_id": "s"}]}
        self.assertEqual(validate_answer(answer, "r", {("a", "s")}), answer)
        with self.assertRaises(ValidationError):
            validate_answer(answer, "r", {("other", "s")})
        with self.assertRaises(ValidationError):
            validate_answer({**answer, "text": "Ответ [a / invented]"}, "r", {("a", "s")})
        with self.assertRaises(ValidationError):
            validate_answer({**answer, "refused": "false"}, "r", {("a", "s")})

    def test_full_refusal_text_sets_the_flag(self):
        answer = {"record_id": "r", "text": "Я не могу дать подтверждённый ответ [a / s].",
                  "refused": False, "citations": [{"article_id": "a", "section_id": "s"}]}
        self.assertTrue(normalize_refusal(answer)["refused"])
        boundary = {"record_id": "r", "text": "Не могу рекомендовать дозировку, но режим сна важен [a / s].",
                    "refused": False, "citations": [{"article_id": "a", "section_id": "s"}]}
        self.assertFalse(normalize_refusal(boundary)["refused"])

    def test_refusal_with_a_substantive_contrast_stays_substantive(self):
        boundary = {
            "record_id": "r",
            "text": ("Я не могу дать точный ответ о причине, но исследование показывает, "
                     "что кофеин задерживает засыпание [a / s]."),
            "refused": False,
            "citations": [{"article_id": "a", "section_id": "s"}],
        }
        self.assertFalse(normalize_refusal(boundary)["refused"])

    def test_agent_normalizes_a_full_refusal_text(self):
        client = FakeChat(first_text="Я не могу дать подтверждённый ответ [1].")
        result = RAGAgent(self.index, self.experts, client).answer(
            "caffeine", "clinician", user_id="u", record_id="r",
        )
        self.assertTrue(result["answer"]["refused"])

    def test_client_records_real_usage_and_enforces_call_cap(self):
        client = ChatClient("https://proxy.test/v1", "SECRET", "test-model", max_calls=1, min_interval=0)
        response = MagicMock()
        response.read.return_value = json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                                 "usage": {"total_tokens": 123}}).encode()
        client.opener = MagicMock()
        client.opener.open.return_value.__enter__.return_value = response
        result = client.complete([{"role": "user", "content": "test"}])
        self.assertEqual(client.total_tokens, 123)
        self.assertEqual(result["finish_reason"], "stop")
        with self.assertRaises(RuntimeError):
            client.complete([{"role": "user", "content": "test"}])

    def test_client_rejects_unaccountable_proxy_responses(self):
        cases = {
            "not an object": [],
            "missing usage": {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
            "bad usage": {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                          "usage": {"total_tokens": "123"}},
            "no choices": {"choices": [], "usage": {"total_tokens": 123}},
            "bad choice": {"choices": ["nope"], "usage": {"total_tokens": 123}},
            "bad message": {"choices": [{"message": "nope", "finish_reason": "stop"}],
                            "usage": {"total_tokens": 123}},
            "no content": {"choices": [{"message": {}, "finish_reason": "stop"}],
                           "usage": {"total_tokens": 123}},
            "bad finish reason": {"choices": [{"message": {"content": "{}"}, "finish_reason": 1}],
                                  "usage": {"total_tokens": 123}},
        }
        for name, payload in cases.items():
            with self.subTest(name):
                client = ChatClient("https://proxy.test/v1", "SECRET", "test-model",
                                    max_calls=2, min_interval=0)
                response = MagicMock()
                response.read.return_value = json.dumps(payload).encode()
                client.opener = MagicMock()
                client.opener.open.return_value.__enter__.return_value = response
                with self.assertRaises(ProxyProtocolError):
                    client.complete([{"role": "user", "content": "test"}])

    def test_client_rejects_invalid_json_as_a_protocol_error(self):
        client = ChatClient("https://proxy.test/v1", "SECRET", "test-model",
                            max_calls=1, min_interval=0)
        response = MagicMock()
        response.read.return_value = b"not-json"
        client.opener = MagicMock()
        client.opener.open.return_value.__enter__.return_value = response
        with self.assertRaises(ProxyProtocolError):
            client.complete([{"role": "user", "content": "test"}])

    def test_client_exposes_retry_after_for_rate_limits(self):
        from email.message import Message
        headers = Message()
        headers["Retry-After"] = "7"
        client = ChatClient("https://proxy.test/v1", "SECRET", "test-model",
                            max_calls=1, min_interval=0)
        client.opener = MagicMock()
        client.opener.open.side_effect = HTTPError(
            "https://proxy.test/v1/chat/completions", 429, "Too Many Requests", headers, None,
        )
        with self.assertRaises(ChatTransientError) as raised:
            client.complete([{"role": "user", "content": "test"}])
        self.assertEqual(raised.exception.retry_after, 7.0)

    def test_compact_citation_mapping_rejects_hallucinated_slots(self):
        evidence = [{"article_id": "a", "section_id": "s"}]
        result = decode_compact({"text": "Факт [1].", "refused": False, "refs": [1]}, "r", evidence)
        self.assertEqual(result["text"], "Факт [a / s].")
        with self.assertRaises(ValidationError):
            decode_compact({"text": "Факт [2].", "refused": False, "refs": [2]}, "r", evidence)

    def test_compact_drops_unneeded_history_but_preserves_explicit_references(self):
        history = [{"user_id": "u", "role": "user", "text": "кофеин", "at": "2026-11-01T10:00:00+03:00"}]
        agent = RAGAgent(self.index, self.experts, FakeChat(), history=history)
        direct = agent.prepare("кофеин", "clinician", user_id="u", record_id="r")
        followup = agent.prepare("А за сколько до сна?", "clinician", user_id="u", record_id="r")
        self.assertEqual(direct["history_messages"], 0)
        self.assertEqual(followup["history_messages"], 1)

    def test_mixed_question_keeps_both_topics(self):
        topics = topic_queries("Кофеин и вечерняя тренировка влияют на сон?")
        self.assertTrue(any("caffeine" in topic for topic in topics))
        self.assertTrue(any("exercise" in topic for topic in topics))

    def test_bm25_reranker_prefers_findings_to_a_methods_only_abstract(self):
        hits = [
            {"chunk_id": "abstract-methods", "bm25_score": 13, "text": "Introduction: Goal. Methods: Design.",
             "metadata": {"section_type": "abstract", "heading_path": ["Abstract (1/2)"]}},
            {"chunk_id": "abstract-results", "bm25_score": 7, "text": "Results: Outcome.",
             "metadata": {"section_type": "abstract", "heading_path": ["Abstract (2/2)"]}},
            {"chunk_id": "conclusion", "bm25_score": 8, "text": "Observed outcome and limits.",
             "metadata": {"section_type": "conclusion", "heading_path": ["Conclusion"]}},
        ]
        self.assertEqual([hit["chunk_id"] for hit in prioritize_evidence(hits, 3)],
                         ["conclusion", "abstract-results", "abstract-methods"])

    def test_plain_text_cleanup_keeps_citations_and_punctuation(self):
        self.assertEqual(
            clean_answer_text("### **Сон**\n\n\u200bПауза — 30 минут [a / s_1].\ue200cite\ue202hidden\ue201\ue000\ufffd"),
            "Сон\n\nПауза — 30 минут [a / s_1].",
        )

    def test_compact_is_smaller_and_full_profile_still_works(self):
        compact = RAGAgent(self.index, self.experts, FakeChat())
        full = RAGAgent(self.index, self.experts, FakeChat(), prompt_profile="full")
        args = {"user_id": "u", "record_id": "r"}
        c = compact.prepare("caffeine", "clinician", **args)
        f = full.prepare("caffeine", "clinician", **args)
        self.assertLess(len(c["messages"][0]["content"]), len(f["messages"][0]["content"]) / 2)
        self.assertEqual(full.answer("caffeine", "clinician", **args)["answer"]["record_id"], "r")


if __name__ == "__main__":
    unittest.main()
