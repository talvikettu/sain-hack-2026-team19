"""Synchronous public interface for retrieval, prompts and generation."""

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
import time

from rag_preprocessing import pack_context
from .generation import Generation
from .history import group_search_query, select_history
from .prompts import PERSONA_FILES, render_messages
from .retrieval import merge_rankings, references_history, search_rankings
# Compatibility for existing direct imports from rag_agent.agent.
from .validation import ValidationError, clean_answer_text, decode_compact, safety_flags, validate_answer, validate_compact_text


class RAGAgent:
    def __init__(self, index, experts, client, *, prompts_dir=None, history=None,
                 top_k=None, context_chars=None, max_repairs=1, prompt_profile="compact"):
        if max_repairs not in (0, 1):
            raise ValueError("At most one repair is supported")
        self.index, self.client = Path(index), client
        self.experts = {expert["expert_id"]: expert for expert in experts}
        self.history = history or []
        if prompt_profile not in {"compact", "full"}:
            raise ValueError("Unknown prompt profile")
        self.prompt_profile = prompt_profile
        self.prompts_dir = Path(prompts_dir or Path(__file__).resolve().parents[1] / "prompts")
        system_file = "system_compact.md" if prompt_profile == "compact" else "system.md"
        self.system = (self.prompts_dir / system_file).read_text(encoding="utf-8-sig")
        self.top_k = top_k if top_k is not None else (3 if prompt_profile == "compact" else 5)
        self.context_chars = context_chars if context_chars is not None else (4500 if prompt_profile == "compact" else 12000)
        self.max_repairs = max_repairs
        self.evidence_cache = OrderedDict()
        self._cache_lock = Lock()

    def _evidence_for(self, key, question, deadline):
        with self._cache_lock:
            if key in self.evidence_cache:
                self.evidence_cache.move_to_end(key)
                return self.evidence_cache[key]
        evidence = self._retrieve(question, deadline)
        with self._cache_lock:
            self.evidence_cache[key] = evidence
            self.evidence_cache.move_to_end(key)
            if len(self.evidence_cache) > 128:
                self.evidence_cache.popitem(last=False)
        return evidence

    def _retrieve(self, question, deadline):
        rankings = [ranking[:self.top_k] for ranking in search_rankings(self.index, question, deadline=deadline)]
        return pack_context(self.index, merge_rankings(rankings, self.top_k), max_chars=self.context_chars)

    def group_query(self, records):
        return group_search_query(records, self.history)

    def prepare_group(self, records, *, deadline=None):
        """Select shared evidence before generating the first expert answer."""
        # Pick a common anchor only after retrieving across the question's topics.
        # Requiring multiple sources can force every expert to cite weak matches.
        evidence = self._retrieve(self.group_query(records), deadline)
        return evidence[:1]

    def prepare(self, question, expert_id, *, user_id, record_id, at=None, deadline=None, shared_evidence=None):
        """Build an inspectable request without calling the model."""
        if expert_id not in PERSONA_FILES or expert_id not in self.experts:
            raise ValueError("Unknown expert_id")
        if not all(isinstance(v, str) and v.strip() for v in (question, user_id, record_id)):
            raise ValueError("Question, user ID and record ID must be non-empty strings")
        if len(question) > 12000:
            raise ValueError("Question exceeds local input guard")
        needs_history = references_history(question)
        history = select_history(self.history, user_id, at,
                                 limit=4 if self.prompt_profile == "compact" else 6,
                                 max_chars=2000 if self.prompt_profile == "compact" else 3500)
        if self.prompt_profile == "compact" and not needs_history:
            history = []
        # Resolve explicit references using recent USER messages, not prior assistant claims.
        retrieval_query = question
        if needs_history:
            retrieval_query += "\n" + "\n".join(m["text"] for m in history if m["role"] == "user")
        key = sha256(json.dumps([user_id, question, history], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if shared_evidence is None:
            evidence = self._evidence_for(key, retrieval_query, deadline)
        else:
            evidence = shared_evidence
        if not evidence:
            raise ValidationError("No evidence found; expand retrieval before calling the LLM")
        allowed = {(block["article_id"], block["section_id"]) for block in evidence}
        required = None
        if shared_evidence is not None:
            required = list(dict.fromkeys((block["article_id"], block["section_id"]) for block in evidence))
        messages = render_messages(
            profile=self.prompt_profile, system=self.system, prompts_dir=self.prompts_dir,
            expert=self.experts[expert_id], expert_id=expert_id, question=question, record_id=record_id,
            history=history, evidence=evidence, required=required,
        )
        return {"messages": messages, "evidence": evidence, "allowed": allowed,
                "required": required, "cache_key": key, "history_messages": len(history)}

    def _generation(self, prepared, *, question, expert_id, record_id, started):
        return Generation(
            prepared, question=question, expert_id=expert_id, record_id=record_id,
            profile=self.prompt_profile, system=self.system, max_repairs=self.max_repairs, started=started,
        )

    def answer(self, question, expert_id, *, user_id, record_id, at=None, deadline=None, shared_evidence=None):
        started = time.monotonic()
        prepared = self.prepare(question, expert_id, user_id=user_id, record_id=record_id,
                                at=at, deadline=deadline, shared_evidence=shared_evidence)
        generation = self._generation(prepared, question=question, expert_id=expert_id,
                                      record_id=record_id, started=started)
        while True:
            response = self.client.complete(generation.messages, deadline=deadline, max_tokens=generation.max_tokens)
            result = generation.accept(response)
            if result is not None:
                return result
