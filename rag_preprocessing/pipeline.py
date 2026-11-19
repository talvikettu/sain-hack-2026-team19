"""Deterministic preprocessing, SQLite storage, BM25, and optional vector search.

No runtime dependencies beyond Python 3.11's standard library. SQLite stores
vectors; cosine search is implemented here, without an external DB server.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import tempfile
import time
import unicodedata

from .embeddings import normalize

VERSION = "1.0.0"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Config:
    mode: str = "sections"  # sections | structural | semantic
    max_words: int = 250
    overlap_words: int = 40
    semantic_min_words: int = 80
    semantic_threshold: float = 0.65
    batch_size: int = 16
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self):
        if self.mode not in {"sections", "structural", "semantic"}:
            raise ValueError("Unknown chunking mode")
        if self.max_words < 1 or not 0 <= self.overlap_words < self.max_words:
            raise ValueError("Require 0 <= overlap_words < max_words")
        if not 1 <= self.semantic_min_words <= self.max_words:
            raise ValueError("Invalid semantic minimum")
        if not -1 <= self.semantic_threshold <= 1 or self.batch_size < 1:
            raise ValueError("Invalid semantic threshold or batch size")
        if not math.isfinite(self.k1) or self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("Invalid BM25 parameters")


def load_articles(source):
    """Accept an articles list, dataset dict, API envelope, or a JSON path.

    Retain only the article fields. References, user data, and judge instructions
    never enter the scientific retrieval corpus. This validates the consumed
    fields, not the entire organizer Case4Dataset schema.
    """
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    if isinstance(source, dict) and "dataset" in source:
        source = source["dataset"]
    if isinstance(source, dict):
        source = source.get("articles")
    if not isinstance(source, list) or not source:
        raise ValueError("Expected a non-empty articles list or dataset")
    articles, article_ids = [], set()
    for article in source:
        if not isinstance(article, dict):
            raise ValueError("Invalid article")
        aid = article.get("article_id")
        if not isinstance(aid, str) or not aid or aid in article_ids:
            raise ValueError("Missing or duplicate article ID")
        article_ids.add(aid)
        for key in ("title", "doi"):
            if not isinstance(article.get(key), str):
                raise ValueError("Article title and DOI must be strings")
        if (not isinstance(article.get("authors"), list)
                or any(not isinstance(x, str) for x in article["authors"])
                or type(article.get("year")) is not int
                or not isinstance(article.get("license"), dict)):
            raise ValueError("Invalid article metadata")
        if not isinstance(article.get("sections"), list) or not article["sections"]:
            raise ValueError("Article must contain sections")
        section_ids, sections = set(), []
        for section in article["sections"]:
            if not isinstance(section, dict):
                raise ValueError("Invalid section")
            sid = section.get("section_id")
            if not isinstance(sid, str) or not sid or sid in section_ids:
                raise ValueError("Missing or duplicate section ID within article")
            section_ids.add(sid)
            if not all(isinstance(section.get(k), str) for k in ("title", "text")):
                raise ValueError("Section title/text must be strings")
            sections.append({k: section[k] for k in ("section_id", "title", "text")})
        articles.append({**{k: article[k] for k in
                               ("article_id", "title", "authors", "year", "doi", "license")},
                         "sections": sections})
    return sorted(articles, key=lambda article: article["article_id"])


def clean_text(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\n\t")
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def heading_path(title):
    return [re.sub(r"\s*\(\d+/\d+\)\s*$", "", part).strip()
            for part in title.split("›") if part.strip()]


def section_type(path):
    words = " ".join(path).lower()
    for pattern, kind in ((r"\babstract\b", "abstract"), (r"\bintroduction\b", "introduction"),
                          (r"\blimitations?\b", "limitations"), (r"\bmethods?\b", "methods"),
                          (r"\bresults?\b", "results"), (r"\bdiscussion\b", "discussion"),
                          (r"\bconclusions?\b", "conclusion")):
        if re.search(pattern, words):
            return kind
    return "other"


def sentence_spans(text):
    """Conservative sentence/paragraph boundaries; never splits within a token.

    This is a deterministic heuristic, not a scientific-domain NLP model.
    """
    start, spans = 0, []
    boundaries = re.finditer(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ\"“])|\n\s*\n", text)
    abbreviations = re.compile(r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Dr|Mr|Mrs|vs|No)\.)$", re.I)
    for match in boundaries:
        if abbreviations.search(text[max(start, match.start() - 15):match.start()]):
            continue
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("RAG preprocessing deadline exceeded")


def embed_batched(provider, texts, config, deadline):
    vectors, dimensions = [], None
    for start in range(0, len(texts), config.batch_size):
        check_deadline(deadline)
        batch = texts[start:start + config.batch_size]
        result = provider.embed(batch, deadline=deadline)
        if len(result) != len(batch):
            raise ValueError("Embedding count mismatch")
        for vector in result:
            vector = normalize(vector)
            dimensions = dimensions or len(vector)
            if len(vector) != dimensions:
                raise ValueError("Embedding dimensions changed between batches")
            vectors.append(vector)
    return vectors


def chunk_spans(text, config, provider=None, deadline=None):
    if not text:
        return []
    if config.mode == "sections":
        return [(0, len(text))]
    units = sentence_spans(text)
    lengths = [len(text[a:b].split()) for a, b in units]
    vectors = None
    if config.mode == "semantic":
        if provider is None:
            raise ValueError("Semantic chunking requires an approved embedding provider")
        vectors = embed_batched(provider, [text[a:b] for a, b in units], config, deadline)
    output, start = [], 0
    while start < len(units):
        end, count = start, 0
        while end < len(units):
            if end > start:
                if count + lengths[end] > config.max_words:
                    break
                if vectors and count >= config.semantic_min_words:
                    similarity = sum(a * b for a, b in zip(vectors[end - 1], vectors[end]))
                    if similarity < config.semantic_threshold:
                        break
            count += lengths[end]
            end += 1
        output.append((units[start][0], units[end - 1][1]))
        if end == len(units):
            break
        # Whole-sentence overlap, bounded by the configured word budget.
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap + lengths[next_start - 1] <= config.overlap_words:
            next_start -= 1
            overlap += lengths[next_start]
        start = next_start
    return output


def tokenize(text):
    return re.findall(r"\w+(?:[-’']\w+)*", unicodedata.normalize("NFC", text).casefold())


# Transparent, deliberately small lexical expansion, not machine translation.
QUERY_TERMS = {
    "коф": "coffee caffeine", "кофеин": "caffeine", "сон": "sleep",
    "сна": "sleep", "сном": "sleep", "сплю": "sleep", "спать": "sleep",
    "засып": "sleep onset latency", "бессон": "insomnia",
    "алког": "alcohol ethanol", "вино": "alcohol wine", "вин": "wine alcohol",
    "свет": "light", "экран": "screen light", "температур": "temperature",
    "питан": "diet nutrition food", "ужин": "dinner meal", "еда": "food diet",
    "дневн": "daytime nap", "дрем": "nap", "мелатонин": "melatonin",
    "пробужд": "awakening", "устал": "fatigue", "нагруз": "exercise",
    "спорт": "exercise", "хронотип": "chronotype", "режим": "schedule timing",
    "трениров": "exercise training", "пробеж": "exercise running",
    "гуля": "walking light", "утрен": "morning", "телефон": "screen light",
    "жар": "heat temperature thermal", "душно": "heat temperature ventilation",
    "батаре": "heat temperature", "комнат": "room bedroom",
    "шум": "noise sound", "розов": "pink", "рот": "mouth",
    "пластыр": "mouth taping", "храп": "snoring", "апноэ": "apnea",
}


def expand_query(query):
    tokens = tokenize(query)
    extra = set()
    for token in tokens:
        for prefix, english in QUERY_TERMS.items():
            if token.startswith(prefix):
                extra.update(english.split())
    return " ".join(tokens + sorted(extra))


@contextmanager
def _connect_readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE articles(article_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
CREATE TABLE sections(article_id TEXT NOT NULL, section_id TEXT NOT NULL,
    source_text TEXT NOT NULL, clean_text TEXT NOT NULL, metadata TEXT NOT NULL,
    PRIMARY KEY(article_id,section_id));
CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, article_id TEXT NOT NULL,
    section_id TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, length INTEGER NOT NULL);
CREATE TABLE postings(term TEXT NOT NULL, chunk_id TEXT NOT NULL, tf INTEGER NOT NULL,
    PRIMARY KEY(term,chunk_id));
CREATE TABLE vectors(chunk_id TEXT PRIMARY KEY, dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL);
"""


def build_index(source, output, config=None, *, provider=None, deadline=None):
    """Build or reuse an atomic on-disk index. No network without provider.

    Pass a monotonic absolute deadline shared with the enclosing /run. Only one
    builder should write a given output path at a time (coordinate in backend).
    """
    started = time.monotonic()
    config = config or Config()
    articles = load_articles(source)
    check_deadline(deadline)
    if config.mode == "semantic" and provider is None:
        raise ValueError("Semantic chunking requires an approved embedding provider")
    identity = provider.identity if provider else None
    corpus_hash = digest(articles)
    build_hash = digest({"version": VERSION, "corpus": corpus_hash,
                         "config": asdict(config), "embedding": identity})
    output = Path(output)
    if output.is_file():
        try:
            with _connect_readonly(output) as connection:
                meta = dict(connection.execute("SELECT key,value FROM metadata"))
                if meta.get("build_hash") == build_hash and connection.execute("PRAGMA quick_check").fetchone()[0] == "ok":
                    report = json.loads(meta["report"])
                    return {**report, "cache_hit": True, "build_seconds": round(time.monotonic() - started, 4)}
        except (sqlite3.Error, ValueError, KeyError):
            pass  # Invalid/incomplete cache: rebuild, preserving old file on failure.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".rag-build-", suffix=".sqlite3", dir=output.parent)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(SCHEMA)
        chunk_records, embedding_texts, warnings = [], [], []
        section_count, word_count = 0, 0
        for article in articles:
            article_meta = {k: v for k, v in article.items() if k != "sections"}
            connection.execute("INSERT INTO articles VALUES (?,?)", (article["article_id"], canonical(article_meta)))
            sections = article["sections"]
            for order, section in enumerate(sections):
                check_deadline(deadline)
                section_count += 1
                aid, sid = article["article_id"], section["section_id"]
                cleaned = clean_text(section["text"])
                path = heading_path(section["title"])
                meta = {"section_title": section["title"], "heading_path": path,
                        "section_type": section_type(path), "section_order": order,
                        "previous_section_id": sections[order - 1]["section_id"] if order else None,
                        "next_section_id": sections[order + 1]["section_id"] if order + 1 < len(sections) else None,
                        "source_hash": digest(section["text"]), "clean_hash": digest(cleaned)}
                connection.execute("INSERT INTO sections VALUES (?,?,?,?,?)", (aid, sid, section["text"], cleaned, canonical(meta)))
                spans = chunk_spans(cleaned, config, provider, deadline)
                if not spans:
                    warnings.append("empty_section")  # Counts only, never closed-dataset IDs/text.
                for part, (start, end) in enumerate(spans):
                    body = cleaned[start:end]
                    chunk_id = digest([VERSION, aid, sid, meta["clean_hash"], asdict(config), identity, start, end])
                    cm = {**meta, "chunk_order": part, "start_char": start, "end_char": end,
                          "corpus_hash": corpus_hash, "pipeline_version": VERSION}
                    if len(body.split()) > config.max_words and config.mode != "sections":
                        warnings.append("oversized_sentence")
                    retrieval_text = article["title"] + "\n" + " > ".join(path) + "\n" + body
                    terms = tokenize(retrieval_text)
                    word_count += len(terms)
                    connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (chunk_id, aid, sid, body, canonical(cm), len(terms)))
                    connection.executemany("INSERT INTO postings VALUES (?,?,?)",
                                           ((term, chunk_id, tf) for term, tf in Counter(terms).items()))
                    chunk_records.append(chunk_id)
                    if provider:
                        embedding_texts.append(retrieval_text)
        if not chunk_records:
            raise ValueError("Corpus has no indexable non-empty text")
        dimensions = None
        if provider:
            vectors = embed_batched(provider, embedding_texts, config, deadline)
            dimensions = len(vectors[0])
            connection.executemany("INSERT INTO vectors VALUES (?,?,?)", (
                (cid, dimensions, struct.pack("<" + "f" * dimensions, *vector))
                for cid, vector in zip(chunk_records, vectors)))
        check_deadline(deadline)
        report = {"articles": len(articles), "sections": section_count, "chunks": len(chunk_records),
                  "dimensions": dimensions, "mode": config.mode, "cache_hit": False,
                  "warnings": dict(Counter(warnings)), "corpus_hash": corpus_hash,
                  "build_seconds": round(time.monotonic() - started, 4)}
        values = {"build_hash": build_hash, "corpus_hash": corpus_hash, "version": VERSION,
                  "config": canonical(asdict(config)), "embedding_identity": identity or "",
                  "average_length": str(word_count / len(chunk_records)), "report": canonical(report)}
        connection.executemany("INSERT INTO metadata VALUES (?,?)", values.items())
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Index integrity check failed")
        connection.close()
        connection = None
        check_deadline(deadline)
        os.replace(temporary, output)
        return report
    finally:
        if connection is not None:
            connection.close()
        if Path(temporary).exists():
            Path(temporary).unlink()


def search(index, query, *, top_k=5, candidate_k=20, provider=None, deadline=None,
           expand=True, rrf_k=60):
    """Return evidence with source IDs; ranking is not proof of relevance.

    The caller must validate evidence sufficiency before answering. No chat
    calls, history processing, or user authorization are performed here.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be non-empty text")
    if not 1 <= top_k <= candidate_k or rrf_k < 1:
        raise ValueError("Require 1 <= top_k <= candidate_k and positive rrf_k")
    check_deadline(deadline)
    with _connect_readonly(index) as connection:
        meta = dict(connection.execute("SELECT key,value FROM metadata"))
        config = json.loads(meta["config"])
        total = json.loads(meta["report"])["chunks"]
        average = float(meta["average_length"])
        scores = defaultdict(float)
        query_terms = set(tokenize(expand_query(query) if expand else query))
        for term in sorted(query_terms):
            matches = connection.execute(
                "SELECT p.chunk_id,p.tf,c.length FROM postings p JOIN chunks c USING(chunk_id) WHERE p.term=?", (term,)).fetchall()
            idf = math.log(1 + (total - len(matches) + 0.5) / (len(matches) + 0.5))
            for row in matches:
                denominator = row["tf"] + config["k1"] * (1 - config["b"] + config["b"] * row["length"] / average)
                scores[row["chunk_id"]] += idf * row["tf"] * (config["k1"] + 1) / denominator
        rankings = [sorted(scores, key=lambda cid: (-scores[cid], cid))[:candidate_k]]
        vector_scores = {}
        if provider is not None:
            if not meta["embedding_identity"] or meta["embedding_identity"] != provider.identity:
                raise ValueError("Query embedding provider does not match index")
            vector = embed_batched(provider, [query], Config(), deadline)[0]
            for row in connection.execute("SELECT chunk_id,dimensions,embedding FROM vectors"):
                if row["dimensions"] != len(vector):
                    raise ValueError("Query embedding dimensions do not match index")
                stored = struct.unpack("<" + "f" * len(vector), row["embedding"])
                vector_scores[row["chunk_id"]] = sum(a * b for a, b in zip(vector, stored))
            rankings.append(sorted(vector_scores, key=lambda cid: (-vector_scores[cid], cid))[:candidate_k])
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, cid in enumerate(ranking, 1):
                fused[cid] += 1 / (rrf_k + rank)
        output, seen = [], set()
        for cid in sorted(fused, key=lambda item: (-fused[item], item)):
            check_deadline(deadline)
            row = connection.execute("SELECT * FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
            # Do not merge equal text from different sources or lose their citations.
            key = (row["article_id"], row["section_id"], row["text"])
            if key in seen:
                continue
            seen.add(key)
            article_meta = json.loads(connection.execute("SELECT metadata FROM articles WHERE article_id=?", (row["article_id"],)).fetchone()[0])
            output.append({"chunk_id": cid, "article_id": row["article_id"],
                           "section_id": row["section_id"], "text": row["text"],
                           "metadata": {**article_meta, **json.loads(row["metadata"])},
                           "rrf_score": fused[cid], "bm25_score": scores.get(cid, 0),
                           "cosine_score": vector_scores.get(cid)})
            if len(output) == top_k:
                break
        return output


def pack_context(index, hits, *, max_chars=12000):
    """Merge overlapping evidence spans within each original section.

    Budget is characters, NOT model tokens. Never truncates a scientific passage
    mid-sentence: skips a block that does not fit. Callers should measure actual
    tokens and verify sufficient evidence before generation.
    """
    if max_chars < 1:
        raise ValueError("Context budget must be positive")
    grouped = {}
    for hit in hits:
        pair = hit["article_id"], hit["section_id"]
        entry = grouped.setdefault(pair, {"spans": [], "metadata": hit["metadata"]})
        entry["spans"].append((hit["metadata"]["start_char"], hit["metadata"]["end_char"]))
    blocks, used = [], 0
    with _connect_readonly(index) as connection:
        for (aid, sid), entry in grouped.items():
            row = connection.execute("SELECT clean_text FROM sections WHERE article_id=? AND section_id=?", (aid, sid)).fetchone()
            if row is None:
                raise ValueError("Evidence section does not exist in index")
            merged = []
            for start, end in sorted(set(entry["spans"])):
                if not 0 <= start < end <= len(row[0]):
                    raise ValueError("Invalid evidence offsets")
                if merged and start <= merged[-1][1]:
                    merged[-1] = merged[-1][0], max(merged[-1][1], end)
                else:
                    merged.append((start, end))
            for start, end in merged:
                text = row[0][start:end]
                label = f"[{aid} / {sid}] {entry['metadata']['title']} > {entry['metadata']['section_title']}\n"
                if used + len(label) + len(text) > max_chars:
                    continue
                blocks.append({"article_id": aid, "section_id": sid, "text": text,
                               "context_text": label + text,
                               "metadata": {**entry["metadata"], "start_char": start, "end_char": end}})
                used += len(label) + len(text)
    return blocks
