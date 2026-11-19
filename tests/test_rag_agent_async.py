"""Offline checks for the standalone async interface."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from rag_agent import AgentBusyError, AsyncRAGAgent, RAGAgent
from rag_preprocessing import build_index


def completion(messages, **_):
    payload = json.loads(messages[1]["content"])
    source = payload["evidence"][0]
    if "ref" in source:
        answer = {"text": "Вывод [1].", "refused": False, "refs": [1]}
    else:
        citation = {key: source[key] for key in ("article_id", "section_id")}
        answer = {"record_id": payload["record_id"], "text": "Вывод [a / s].",
                  "refused": False, "citations": [citation]}
    return {"content": json.dumps(answer), "finish_reason": "stop", "usage": {"total_tokens": 7}, "seconds": 0}


class AsyncAgentTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "tmp" / "async-agent-tests"
        root.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=root)
        cls.addClassCleanup(tmp.cleanup)
        cls.index = Path(tmp.name) / "index.sqlite3"
        build_index([{
            "article_id": "a", "title": "Sleep and caffeine", "authors": ["Synthetic"], "year": 2026,
            "doi": "test:async", "license": {"name": "CC BY"},
            "sections": [{"section_id": "s", "title": "Results", "text": "Caffeine affects sleep."}],
        }], cls.index)

    def agent(self, client=None, *, profile="compact"):
        experts = [{"expert_id": "clinician", "role": "role", "speech_style": "style"}]
        return RAGAgent(self.index, experts, client, prompt_profile=profile)

    async def test_blocking_client_matches_sync_for_both_profiles_and_repairs(self):
        class RepairClient:
            def complete(self, messages, **kwargs):
                result = completion(messages, **kwargs)
                if "Проверка не пройдена" not in messages[0]["content"]:
                    result["content"] = "{"
                return result

        for profile in ("compact", "full"):
            with self.subTest(profile=profile):
                agent = self.agent(RepairClient(), profile=profile)
                args = {"user_id": "u", "record_id": "r"}
                reference = agent.answer("caffeine", "clinician", **args)
                async with AsyncRAGAgent(agent) as async_agent:
                    result = await async_agent.answer("caffeine", "clinician", **args)
                self.assertEqual(len(result["attempts"]), 2)
                for key in ("answer", "retrieved", "checks", "attempts", "history_messages"):
                    self.assertEqual(result[key], reference[key])

    async def test_native_async_provider_keeps_admission_bounded(self):
        both_entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def complete(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                both_entered.set()
            await release.wait()
            return completion(messages, **kwargs)

        async with AsyncRAGAgent(self.agent(), async_complete=complete, max_pending=2, provider_concurrency=2) as agent:
            async with asyncio.TaskGroup() as tasks:
                first = tasks.create_task(agent.answer("caffeine", "clinician", user_id="a", record_id="1"))
                second = tasks.create_task(agent.answer("caffeine", "clinician", user_id="b", record_id="2"))
                try:
                    await asyncio.wait_for(both_entered.wait(), timeout=5)
                    with self.assertRaises(AgentBusyError):
                        await agent.prepare("caffeine", "clinician", user_id="c", record_id="3")
                finally:
                    release.set()
        self.assertEqual([first.result()["answer"]["record_id"], second.result()["answer"]["record_id"]], ["1", "2"])

    async def test_cancelled_blocking_call_holds_slot_but_retrieval_can_continue(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()

        class BlockingClient:
            calls = 0

            def complete(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(timeout=5):
                        raise TimeoutError("test provider was not released")
                return completion(messages, **kwargs)

        client = BlockingClient()
        async with AsyncRAGAgent(self.agent(client)) as agent:
            first = asyncio.create_task(agent.answer("caffeine", "clinician", user_id="a", record_id="1"))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                prepared = await asyncio.wait_for(
                    agent.prepare("caffeine", "clinician", user_id="b", record_id="2"), timeout=5,
                )
                self.assertEqual(prepared["allowed"], {("a", "s")})
                with self.assertRaises(TimeoutError):
                    await agent.answer("caffeine", "clinician", user_id="b", record_id="2",
                                       deadline=time.monotonic() + 0.1)
                self.assertEqual(client.calls, 1)
            finally:
                release.set()
                if not first.done():
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)
            result = await agent.answer("caffeine", "clinician", user_id="b", record_id="2")
            self.assertEqual(result["answer"]["record_id"], "2")
        self.assertEqual(client.calls, 2)
