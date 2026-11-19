"""Async entry points for standalone RAG without an HTTP framework."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from functools import partial
import time


class AgentBusyError(RuntimeError):
    """The standalone async agent has no free admission slots."""


def _remaining(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("RAG deadline exceeded")
    return remaining


class _BlockingWorker:
    """One dedicated worker with no executor backlog."""

    def __init__(self, name):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.slot = asyncio.Semaphore(1)
        self.in_flight = set()

    async def run(self, function, *args, deadline=None, **kwargs):
        async with asyncio.timeout(_remaining(deadline)):
            await self.slot.acquire()
            try:
                _remaining(deadline)
                call = partial(function, *args, **kwargs)
                future = asyncio.get_running_loop().run_in_executor(self.executor, copy_context().run, call)
            except BaseException:
                self.slot.release()
                raise
            self.in_flight.add(future)
            future.add_done_callback(self._finished)
            # Cancellation of a caller must not free a slot that still has a live thread.
            return await asyncio.shield(future)

    def _finished(self, future):
        self.in_flight.discard(future)
        self.slot.release()
        if not future.cancelled():
            future.exception()  # Observe errors even if the caller has stopped waiting.

    async def close(self):
        try:
            await asyncio.gather(*(asyncio.shield(task) for task in self.in_flight), return_exceptions=True)
        finally:
            self.executor.shutdown(wait=False)


class AsyncRAGAgent:
    """Use within one event loop and an async context manager.

    An injected async_complete must accept the same parameters and return the
    same envelope as ChatClient.complete. Without it, the existing blocking
    client runs on one dedicated provider thread, separate from retrieval.
    """

    def __init__(self, agent, *, async_complete=None, max_pending=8, provider_concurrency=1):
        if min(max_pending, provider_concurrency) < 1:
            raise ValueError("Async capacity must be positive")
        if async_complete is None and provider_concurrency != 1:
            raise ValueError("The blocking client requires one provider slot")
        self._agent = agent
        self._async_complete = async_complete
        self._max_pending = max_pending
        self._provider_concurrency = provider_concurrency
        self._active = 0
        self._accepting = False
        self._loop = None

    async def __aenter__(self):
        if self._loop is not None:
            raise RuntimeError("An async agent context cannot be reused")
        self._loop = asyncio.get_running_loop()
        self._local = _BlockingWorker("rag-retrieval")
        self._provider = _BlockingWorker("rag-provider") if self._async_complete is None else None
        self._provider_slots = asyncio.Semaphore(self._provider_concurrency)
        self._idle = asyncio.Event()
        self._idle.set()
        self._accepting = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self._accepting = False
        try:
            await self._idle.wait()
        finally:
            if self._provider is None:
                await self._local.close()
            else:
                await asyncio.gather(self._local.close(), self._provider.close())

    @contextmanager
    def _admit(self, deadline):
        if not self._accepting or asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Use AsyncRAGAgent inside its active async context")
        _remaining(deadline)
        if self._active >= self._max_pending:
            raise AgentBusyError("Too many active async RAG requests")
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def prepare(self, question, expert_id, *, user_id, record_id, at=None, deadline=None, shared_evidence=None):
        with self._admit(deadline):
            return await self._prepare(question, expert_id, user_id, record_id, at, deadline, shared_evidence)

    async def _prepare(self, question, expert_id, user_id, record_id, at, deadline, shared_evidence):
        call = partial(self._agent.prepare, question, expert_id, user_id=user_id, record_id=record_id,
                       at=at, deadline=deadline, shared_evidence=shared_evidence)
        return await self._local.run(call, deadline=deadline)

    async def prepare_group(self, records, *, deadline=None):
        with self._admit(deadline):
            return await self._local.run(partial(self._agent.prepare_group, records, deadline=deadline), deadline=deadline)

    async def _complete(self, generation, deadline):
        if self._provider is not None:
            call = partial(self._agent.client.complete, generation.messages,
                           deadline=deadline, max_tokens=generation.max_tokens)
            return await self._provider.run(call, deadline=deadline)
        async with asyncio.timeout(_remaining(deadline)):
            async with self._provider_slots:
                _remaining(deadline)
                return await self._async_complete(
                    generation.messages, deadline=deadline, max_tokens=generation.max_tokens,
                )

    async def answer(self, question, expert_id, *, user_id, record_id, at=None, deadline=None, shared_evidence=None):
        started = time.monotonic()
        with self._admit(deadline):
            prepared = await self._prepare(question, expert_id, user_id, record_id, at, deadline, shared_evidence)
            generation = self._agent._generation(prepared, question=question, expert_id=expert_id,
                                                 record_id=record_id, started=started)
            while True:
                response = await self._complete(generation, deadline)
                result = await self._local.run(generation.accept, response, deadline=deadline)
                if result is not None:
                    return result
