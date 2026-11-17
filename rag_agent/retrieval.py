"""Small transparent topic heuristics and evidence prioritization, no LLM calls."""

import re


def references_history(question):
    return bool(re.search(
        r"как (?:раньше|в прошлый раз)|я (?:вам |тебе )?(?:писал|писала|говорил|говорила)|"
        r"а это|а почему|а за сколько|в прошл\w+ сообщени|помнишь|помните", question, re.I))


def focus_query(question):
    # Prefer a named factor over incidental meal/context words. No dataset IDs,
    # expected answers, article IDs, or dev reference are used for routing.
    patterns = [
        (r"алког|\bвин[ао]\b|пив[ао]|alcohol|ethanol", "alcohol ethanol sleep recovery"),
        (r"коф|caffein", "caffeine dose timing bedtime"),
        (r"жар|душно|батаре|температур|thermal", "heat thermal temperature sleep awakening"),
        (r"трениров|пробеж|спорт|\bбег(?:ать|аю|аюсь|у|а|ом)?\b|exercise", "evening exercise training sleep results"),
        (r"утрен\w* свет|утрен\w* прогул|утрен\w* солн|morning light", "morning light circadian sleep timing"),
        (r"телефон|экран|смартфон|планшет|smartphone", "evening smartphone screen sleep onset"),
        (r"свет|ламп|освещ|light", "evening light exposure circadian sleep"),
        (r"будн|выходн|график|режим|фазы сна|хронотип|chronotype", "irregular sleep wake patterns regularity social jetlag"),
        (r"дневн\w* с[но]|дрем|\bnap", "daytime nap sleep alertness"),
    ]
    for pattern, query in patterns:
        if re.search(pattern, question, re.I):
            return query
    return question


def prioritize_evidence(hits, top_k):
    def score(hit):
        path = " ".join(hit["metadata"].get("heading_path", [])).lower()
        # Source relevance remains the primary signal. Factors bias comparable
        # matches toward findings rather than introductory or methods fragments.
        factor = 1.0
        if re.search(r"\b(results?|findings?|conclusions?)\b", path):
            factor = 1.35
        elif "discussion" in path:
            factor = 1.2
        elif "methods" in path:
            factor = 0.65
        elif "introduction" in path or "background" in path:
            factor = 0.75
        return hit["bm25_score"] * factor
    return sorted(hits, key=lambda hit: (-score(hit), hit["chunk_id"]))[:top_k]
