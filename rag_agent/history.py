"""Build context from prior messages scoped to one user and timestamp."""

from datetime import datetime

from .retrieval import references_history


def select_history(history, user_id, at=None, *, limit=6, max_chars=3500):
    cutoff = datetime.fromisoformat(at) if at else None
    if cutoff is not None and cutoff.tzinfo is None:
        raise ValueError("Request timestamp must include timezone")
    candidates = []
    for message in history:
        if message.get("user_id") != user_id:
            continue
        stamp = datetime.fromisoformat(message["at"])
        if stamp.tzinfo is None:
            raise ValueError("History timestamp must include timezone")
        if cutoff is None or stamp < cutoff:
            if message.get("role") in {"user", "agent", "assistant"} and isinstance(message.get("text"), str):
                candidates.append((stamp, {"role": message["role"], "text": message["text"]}))
    ordered = [item[1] for item in sorted(candidates, key=lambda item: item[0])]
    selected, used = [], 0
    for message in reversed(ordered[-limit:]):
        if used + len(message["text"]) > max_chars:
            break
        selected.append(message)
        used += len(message["text"])
    return list(reversed(selected))


def group_search_query(records, history):
    """Combine the requests and relevant history for a shared search."""
    questions = []
    for record in records:
        utterance = record["utterance"]
        question = utterance["text"]
        questions.append(question)
        if references_history(question):
            questions.extend(
                item["text"] for item in select_history(history, record["user_id"], utterance["at"])
                if item["role"] == "user"
            )
    return "\n".join(dict.fromkeys(questions))
