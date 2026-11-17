from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from rag_agent import RAGAgent, ChatClient, ValidationError, validate_answer
from rag_agent.agent import decode_compact, select_history
from rag_agent.retrieval import focus_query
from rag_preprocessing import build_index


class FakeChat:
    def __init__(self, malformed_first=False):
        self.messages = []
        self.malformed_first = malformed_first

    def complete(self, messages, **kwargs):
        self.messages.append(list(messages))
        data = json.loads(messages[1]["content"])
        source = data["evidence"][0]
        if "ref" in source:
            result = {"text": "Подтверждённый вывод [1].", "refused": False, "refs": [1]}
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
        first = agent.answer("кофеин", "clinician", user_id="u", record_id="r1")
        second = agent.answer("кофеин", "biohacker", user_id="u", record_id="r2")
        self.assertEqual(first["answer"]["citations"], second["answer"]["citations"])
        self.assertEqual(first["checks"]["medical_check"], "not_performed")
        self.assertIn("общий набор refs", client.messages[1][0]["content"])

    def test_bounded_repair(self):
        client = FakeChat(malformed_first=True)
        result = RAGAgent(self.index, self.experts, client).answer("caffeine", "clinician", user_id="u", record_id="r")
        self.assertEqual(len(result["attempts"]), 2)

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

    def test_named_factor_wins_over_incidental_meal_context(self):
        self.assertIn("alcohol", focus_query("Вино за ужином влияет на сон?"))
        self.assertNotIn("diet", focus_query("Вино за ужином влияет на сон?"))

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
