"""Small transparent topic heuristics and evidence prioritization, no LLM calls."""

import re

from rag_preprocessing import search


CANDIDATE_DEPTH = 40
RERANK_DEPTH = 20
ROUTINE_SEARCH_DEPTH = 10

_SCREEN = re.compile(r"планшет|телефон|смартфон|экран|\b(?:tablets?|smartphones?|screens?|e[- ]?readers?)\b", re.I)
_PAPER = re.compile(r"бумаж|печатн|кни[гж]|\b(?:paper|printed|physical|books?)\b", re.I)
_DAYLIGHT = re.compile(
    r"дн[её]м|дневн|ярк\w*\s+свет|\b(?:daytime|daylight|bright light|preceding day)\b",
    re.I,
)
_PAPER_IN_EVIDENCE = re.compile(
    r"\b(?:(?:physical|printed|paper)\s+(?:books?|reading)|hard\s*cop(?:y|ies))\b",
    re.I,
)
_DAYLIGHT_IN_EVIDENCE = re.compile(
    r"\b(?:daytime|daylight|preceding day|prior light|previous light|bright light)\b",
    re.I,
)
_ROUTINE = r"будн|выходн|рабоч\w*\s+дн|суббот|воскресень|график|расписан|режим|weekdays?|weekends?|sleep schedule|social jetlag"
_DELAYED_TIMING = (
    r"бессонниц|задержк\w*\s+фаз|сдвиг\w*\s+(?:времени|фаз[ыа])\s+сна|"
    r"засып\w*\s+под\s+утро|delayed sleep|insomnia"
)

# Match the conditions specified in the question against passages in the
# already selected article. Keep the main topic when a more precise hit is absent.
_GROUP_FOCUS = (
    (
        r"(?:жар|душно|температур|heat).*(?:перв\w* половин|начал\w* ноч|early night|first half)",
        (r"\b(?:heat|thermal)\b", r"\b(?:initial|early|first) (?:part|half|segment|portion)\b",
         r"\b(?:wakefulness|awakenings?)\b"),
    ),
    (
        r"(?:будн|выходн|weekday|weekend).*(?:чем.{0,30}плох|вред|последств|harm|consequence)",
        (r"\b(?:irregular sleep|irregular sleepers|sleep regularity)\b",
         r"\b(?:delayed|later) circadian (?:timing|rhythms?|phase)\b",
         r"\b(?:associated|association|linked)\b"),
    ),
    (
        r"(?:пробеж|бег\w*|нагрузк|exercise|running).*(?:ложусь|до сна|bedtime)",
        (r"\bmoderate\b", r"\b(?:exercise.to.sleep|exercise sleep latency|buffer|interval)\b",
         r"\b(?:study|laboratory|experiment)\b"),
    ),
)


def references_history(question):
    return bool(re.search(
        r"как (?:раньше|в прошлый раз)|я (?:вам |тебе )?(?:писал|писала|говорил|говорила)|"
        r"а это|а почему|а за сколько|в прошл\w+ сообщени|помнишь|помните", question, re.I))


def topic_queries(question):
    # Search each named factor without discarding the original question.
    patterns = [
        (r"алког|\bвин[ао]\b|пив[ао]|alcohol|ethanol", "alcohol ethanol sleep recovery"),
        (r"коф|caffein", "caffeine dose timing bedtime"),
        (r"жар|душно|батаре|температур|thermal", "heat thermal temperature sleep awakening"),
        (r"трениров|пробеж|спорт|\bбег(?:ать|аю|аюсь|у|а|ом)?\b|exercise", "evening exercise training sleep results"),
        (r"утрен\w* свет|утрен\w* прогул|утрен\w* солн|morning light", "morning light circadian sleep timing"),
        (r"телефон|экран|смартфон|планшет|smartphone", "evening smartphone screen sleep onset"),
        (r"свет|ламп|освещ|light", "evening light exposure circadian sleep"),
        (_ROUTINE + r"|фазы сна|хронотип|chronotype", "irregular sleep wake patterns regularity social jetlag"),
        (r"дневн\w* с[но]|дрем|\bnap", "daytime nap sleep alertness"),
    ]
    return [query for pattern, query in patterns if re.search(pattern, question, re.I)]


def screen_reading_query(question):
    """Preserve the two options and, when given, daytime lighting conditions."""
    if not (_SCREEN.search(question) and _PAPER.search(question)):
        return None
    query = "tablet computer screen physical book evening reading sleep"
    daylight = bool(_DAYLIGHT.search(question))
    if daylight:
        query += " bright light preceding day"
    return query, daylight


def supplementary_queries(question):
    """Broaden coverage for named concepts after the main search paths."""
    queries = []
    if re.search(_DELAYED_TIMING, question, re.I):
        queries.append(("delayed sleep phase late bedtime circadian timing morning light evening light insomnia",
                        CANDIDATE_DEPTH))
    if re.search(r"мелатонин|melatonin", question, re.I):
        queries.append(("evening screen blue light melatonin sleep latency circadian exposure", CANDIDATE_DEPTH))
    if re.search(_ROUTINE, question, re.I):
        queries.append(("sleep hygiene education regular bedtime wake time sleep schedule", ROUTINE_SEARCH_DEPTH))
    return queries


def _rank_candidate_pool(hits, *, comparison_daylight=False):
    # Keep the wider pool available without changing the cutoff used for
    # primary evidence ranking. Tail hits contribute to article coverage.
    shortlist = hits[:RERANK_DEPTH]
    return prioritize_evidence(shortlist, len(shortlist), comparison_daylight=comparison_daylight) + hits[RERANK_DEPTH:]


def search_rankings(index, question, *, deadline=None):
    """Collect a wide pool while preserving the priority of primary evidence."""
    def fetch(query, depth=CANDIDATE_DEPTH):
        return search(index, query, top_k=depth, candidate_k=depth, deadline=deadline)

    rankings = []
    deferred = []
    comparison = screen_reading_query(question)
    if comparison is not None:
        query, daylight = comparison
        hits = fetch(query)
        shortlist = hits[:RERANK_DEPTH]
        matched = [hit for hit in shortlist if _SCREEN.search(hit["text"]) and _PAPER_IN_EVIDENCE.search(hit["text"])]
        if matched:
            ordered = prioritize_evidence(matched, len(matched), comparison_daylight=daylight)
            matched_ids = {hit["chunk_id"] for hit in matched}
            rankings.append(ordered + prioritize_evidence(
                [hit for hit in shortlist if hit["chunk_id"] not in matched_ids], len(shortlist)
            ) + hits[RERANK_DEPTH:])
        elif hits:
            # Keep the search candidates available without giving the comparison priority.
            deferred = _rank_candidate_pool(hits)
    queries = dict.fromkeys([*topic_queries(question), question])
    for query in queries:
        rankings.append(_rank_candidate_pool(fetch(query)))
    if deferred:
        rankings.append(deferred)
    for query, depth in supplementary_queries(question):
        if query not in queries and (comparison is None or query != comparison[0]):
            rankings.append(_rank_candidate_pool(fetch(query, depth)))
    return rankings


def merge_rankings(rankings, top_k):
    """Give each topic a slot before filling with lower-ranked evidence."""
    selected = {}
    for ranking in rankings:
        if ranking:
            hit = ranking[0]
            selected.setdefault(hit["chunk_id"], hit)
    for position in range(1, max(map(len, rankings), default=0)):
        for ranking in rankings:
            if position < len(ranking):
                hit = ranking[position]
                selected.setdefault(hit["chunk_id"], hit)
    return list(selected.values())[:top_k]


def select_group_hits(rankings, question, top_k):
    """Prefer a passage that covers the question's conditions within the main article."""
    primary = merge_rankings([ranking[:top_k] for ranking in rankings], top_k)
    if not primary:
        return primary
    for trigger, required_terms in _GROUP_FOCUS:
        if not re.search(trigger, question, re.I | re.S):
            continue
        for ranking in rankings:
            for hit in ranking:
                if (hit["article_id"] == primary[0]["article_id"]
                        and all(re.search(term, hit["text"], re.I) for term in required_terms)):
                    return [hit, *(other for other in primary if other["chunk_id"] != hit["chunk_id"])]
    return primary


def prioritize_evidence(hits, top_k, *, comparison_daylight=False):
    def score(hit):
        metadata = hit["metadata"]
        path = " ".join(metadata.get("heading_path", [])).casefold()
        section_type = metadata.get("section_type")
        # Abstracts can be split into separate chunks with the same generic
        # heading. Inspect their structured labels to tell methods from findings.
        labels = path
        if section_type == "abstract":
            abstract_labels = re.findall(
                r"\b(background|introduction|methods?|results?|findings?|conclusions?)\s*:",
                hit["text"].casefold(),
            )
            labels += " " + " ".join(abstract_labels)
        if re.search(r"\bconclusions?\b", labels):
            factor = 1.6
        elif re.search(r"\bresults?\b", labels) or (section_type == "abstract" and "findings" in labels):
            factor = 1.65 if section_type == "abstract" else 1.45
        elif re.search(r"\bmethods?\b", labels):
            factor = 0.65
        elif re.search(r"\b(?:introduction|background)\b", labels):
            factor = 0.75
        elif section_type == "discussion":
            factor = 1.12
        else:
            factor = 1.0
        if comparison_daylight and _DAYLIGHT_IN_EVIDENCE.search(hit["text"]):
            factor *= 1.35
        return hit["bm25_score"] * factor
    return sorted(hits, key=lambda hit: (-score(hit), hit["chunk_id"]))[:top_k]
