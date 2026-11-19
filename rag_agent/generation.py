"""Request-local completion and repair state shared by both agent interfaces."""

import time

from .validation import ValidationError, parse_completion, safety_flags


class Generation:
    def __init__(self, prepared, *, question, expert_id, record_id, profile, system, max_repairs, started):
        self.prepared = prepared
        self.question = question
        self.expert_id = expert_id
        self.record_id = record_id
        self.profile = profile
        self.system = system
        self.remaining_repairs = max_repairs
        self.request_started = started
        self.started = time.monotonic()
        self.validation_seconds = 0.0
        self.attempts = []
        self.messages = [dict(message) for message in prepared["messages"]]
        self.max_tokens = 1800 if profile == "compact" else 2500

    def accept(self, response):
        """Return the validated result, or update messages for one more attempt."""
        attempt = {key: response[key] for key in ("usage", "finish_reason", "seconds")}
        if "wait_seconds" in response:
            attempt["wait_seconds"] = response["wait_seconds"]
        self.attempts.append(attempt)
        started = time.monotonic()
        try:
            answer = parse_completion(
                response, self.record_id, self.prepared, profile=self.profile, system=self.system,
            )
        except ValidationError as error:
            if not self.remaining_repairs:
                raise
            self.remaining_repairs -= 1
            # Do not forward model output that failed validation.
            repair = "Сформируй ответ заново по исходным данным. Проверка не пройдена: " + str(error)
            self.messages = [
                {**self.messages[0], "content": self.messages[0]["content"] + "\n\n" + repair},
                *self.messages[1:],
            ]
            return None
        finally:
            self.validation_seconds += time.monotonic() - started
        return self._result(answer)

    def _result(self, answer):
        return {
            "answer": answer,
            "expert_id": self.expert_id,
            "question": self.question,
            "history_messages": self.prepared["history_messages"],
            "retrieved": [{"article_id": block["article_id"], "section_id": block["section_id"]}
                          for block in self.prepared["evidence"]],
            "checks": {
                "schema_and_sources": "passed",
                "medical_check": "not_performed",
                "advisory_flags": safety_flags(answer),
            },
            "attempts": self.attempts,
            "seconds": round(time.monotonic() - self.started, 3),
            "timing": {
                "retrieval_and_prompt_seconds": round(self.started - self.request_started, 4),
                "llm_seconds": round(sum(attempt["seconds"] for attempt in self.attempts), 4),
                "validation_seconds": round(self.validation_seconds, 4),
                "total_seconds": round(time.monotonic() - self.request_started, 4),
            },
        }
