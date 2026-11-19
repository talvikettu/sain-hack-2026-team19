import copy
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError

from rag_preprocessing import Config, build_index, load_articles, pack_context, search
from rag_preprocessing.embeddings import ProxyEmbeddings, normalize
from rag_preprocessing.pipeline import chunk_spans, clean_text, heading_path


def corpus():
    return [{"article_id": "a1", "title": "Sleep and caffeine", "authors": ["Synthetic Author"],
             "year": 2026, "doi": "synthetic:test1", "license": {"name": "CC BY"},
             "sections": [
                 {"section_id": "s001", "title": "3. Results › Caffeine (1/2)",
                  "text": "Caffeine affects sleep. The observation does not prove causation. Participants were healthy adults."},
                 {"section_id": "s002", "title": "4. Limitations",
                  "text": "Small sample limits generalization. These results are not treatment instructions."}]},
            {"article_id": "a2", "title": "Daytime naps", "authors": ["Synthetic Author"],
             "year": 2026, "doi": "synthetic:test2", "license": {"name": "CC BY"},
             "sections": [{"section_id": "s001", "title": "Abstract",
                           "text": "Daytime naps may affect alertness. More evidence is needed."}]}]


class FakeEmbedding:
    identity = "synthetic-v1"

    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, deadline=None):
        self.calls += 1
        return [[1.0, 0.0] if "caffeine" in text.casefold() else [0.0, 1.0] for text in texts]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "tmp" / "rag-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.path = Path(self.temp.name) / "index.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_consumes_only_articles_from_envelope(self):
        envelope = {"dataset": {"articles": corpus(), "users": ["PRIVATE"],
                                 "judge_prompts": ["PRIVATE"]}, "reference": "PRIVATE"}
        self.assertEqual(load_articles(envelope), load_articles(corpus()))
        build_index(envelope, self.path)
        self.assertNotIn(b"PRIVATE", self.path.read_bytes())

    def test_duplicate_ids_rejected_but_section_id_is_article_scoped(self):
        self.assertEqual(len(load_articles(corpus())), 2)
        bad = corpus()
        bad[0]["sections"].append(copy.deepcopy(bad[0]["sections"][0]))
        with self.assertRaises(ValueError):
            load_articles(bad)

    def test_cleaning_preserves_evidence(self):
        raw = "No\x00 effect:\t 0.05 mg; 95% CI −0.3–0.2.\r\n\r\n\r\nNot guaranteed [12]."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "No effect: 0.05 mg; 95% CI −0.3–0.2.\n\nNot guaranteed [12].")
        self.assertEqual(clean_text(cleaned), cleaned)

    def test_structural_chunks_cover_sentences_and_preserve_offsets(self):
        text = "First evidence is limited. Second result is not causal. Third observation needs verification."
        config = Config(mode="structural", max_words=10, overlap_words=5, semantic_min_words=3)
        spans = chunk_spans(text, config)
        self.assertGreater(len(spans), 1)
        covered = set()
        for start, end in spans:
            self.assertTrue(text[start:end].endswith("."))
            covered.update(range(start, end))
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        self.assertEqual(heading_path("3. Results › Caffeine (1/2)"), ["3. Results", "Caffeine"])

    def test_build_cache_metadata_and_changed_corpus(self):
        report = build_index(corpus(), self.path)
        self.assertEqual((report["articles"], report["sections"], report["chunks"]), (2, 3, 3))
        self.assertTrue(build_index(corpus(), self.path)["cache_hit"])
        before = self.path.read_bytes()
        changed = corpus()
        changed[0]["sections"][0]["text"] += " New evidence."
        self.assertFalse(build_index(changed, self.path)["cache_hit"])
        self.assertNotEqual(before, self.path.read_bytes())
        with closing(sqlite3.connect(self.path)) as db:
            for aid, sid, text, meta in db.execute("SELECT article_id,section_id,text,metadata FROM chunks"):
                meta = json.loads(meta)
                section = db.execute("SELECT clean_text FROM sections WHERE article_id=? AND section_id=?", (aid, sid)).fetchone()[0]
                self.assertEqual(text, section[meta["start_char"]:meta["end_char"]])

    def test_russian_expansion_and_unmatched_query(self):
        build_index(corpus(), self.path)
        self.assertEqual(search(self.path, "кофеин")[0]["article_id"], "a1")
        self.assertEqual(search(self.path, "zzzzunmatched"), [])
        self.assertEqual(search(self.path, "кофеин", expand=False), [])

    def test_context_has_sources_and_respects_budget(self):
        build_index(corpus(), self.path)
        hits = search(self.path, "caffeine")
        blocks = pack_context(self.path, hits + hits)
        self.assertEqual(len(blocks), len(hits))
        self.assertIn("[a1 / s001]", blocks[0]["context_text"])
        self.assertEqual(pack_context(self.path, hits, max_chars=1), [])

    def test_overlap_is_merged_without_losing_text(self):
        data = corpus()
        config = Config(mode="structural", max_words=12, overlap_words=6, semantic_min_words=3)
        build_index(data, self.path, config)
        hits = [x for x in search(self.path, "caffeine healthy observation", top_k=10, candidate_k=20)
                if x["article_id"] == "a1" and x["section_id"] == "s001"]
        blocks = pack_context(self.path, hits)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], clean_text(data[0]["sections"][0]["text"]))

    def test_vectors_hybrid_and_cache_never_reembeds(self):
        provider = FakeEmbedding()
        report = build_index(corpus(), self.path, provider=provider)
        self.assertEqual(report["dimensions"], 2)
        calls = provider.calls
        self.assertTrue(build_index(corpus(), self.path, provider=provider)["cache_hit"])
        self.assertEqual(provider.calls, calls)
        hits = search(self.path, "caffeine", provider=provider)
        self.assertEqual(hits[0]["article_id"], "a1")
        self.assertAlmostEqual(hits[0]["cosine_score"], 1.0)
        other = FakeEmbedding()
        other.identity = "different-model"
        with self.assertRaises(ValueError):
            search(self.path, "caffeine", provider=other)

    def test_failed_rebuild_preserves_old_index_and_cleans_temporary_file(self):
        build_index(corpus(), self.path)
        before = self.path.read_bytes()

        class Broken(FakeEmbedding):
            def embed(self, texts, *, deadline=None):
                raise RuntimeError("Synthetic failure")

        with self.assertRaises(RuntimeError):
            build_index(corpus(), self.path, provider=Broken())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".rag-build-*")), [])

    def test_semantic_mode_and_deadline(self):
        with self.assertRaises(ValueError):
            build_index(corpus(), self.path, Config(mode="semantic"))
        config = Config(mode="semantic", max_words=50, overlap_words=0, semantic_min_words=2)
        spans = chunk_spans("Caffeine affects sleep. Naps affect alertness.", config, FakeEmbedding())
        self.assertEqual(len(spans), 2)
        with self.assertRaises(TimeoutError):
            build_index(corpus(), self.path, deadline=time.monotonic() - 1)

    def test_provider_security_and_vector_validation(self):
        for endpoint in ("http://proxy.test/v1/embeddings", "https://other.test/v1/embeddings",
                         "https://proxy.test/v1/embeddings?token=SECRET",
                         "https://user:SECRET@proxy.test/v1/embeddings"):
            with self.assertRaises(ValueError):
                ProxyEmbeddings(endpoint, "approved-test-model", "SECRET", "https://proxy.test/v1")
        for vector in ([0, 0], [math.nan, 1], [math.inf, 1], [True, 0], []):
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_bad_dimensions_never_published(self):
        class Wrong(FakeEmbedding):
            def embed(self, texts, *, deadline=None):
                return [[1, 0] if i == 0 else [1, 0, 0] for i, _ in enumerate(texts)]

        with self.assertRaises(ValueError):
            build_index(corpus(), self.path, provider=Wrong())
        self.assertFalse(self.path.exists())

    def test_proxy_reorders_response_and_does_not_leak_http_errors(self):
        provider = ProxyEmbeddings("https://proxy.test/v1/embeddings", "approved-test-model",
                                   "SECRET", "https://proxy.test/v1", min_interval=0)
        response = MagicMock()
        response.read.return_value = json.dumps({"data": [
            {"index": 1, "embedding": [0, 3]}, {"index": 0, "embedding": [2, 0]}
        ]}).encode()
        provider.opener = MagicMock()
        provider.opener.open.return_value.__enter__.return_value = response
        self.assertEqual(provider.embed(["first", "second"]), [[1, 0], [0, 1]])
        provider.opener.open.side_effect = HTTPError("https://PRIVATE", 429, "SECRET", {}, None)
        with self.assertRaises(RuntimeError) as error:
            provider.embed(["third"])
        self.assertNotIn("SECRET", str(error.exception))
        self.assertNotIn("PRIVATE", str(error.exception))
        self.assertIn("429", str(error.exception))

    def test_configuration_change_invalidates_cache(self):
        build_index(corpus(), self.path)
        report = build_index(corpus(), self.path, Config(mode="structural"))
        self.assertFalse(report["cache_hit"])
        self.assertTrue(build_index(corpus(), self.path, Config(mode="structural"))["cache_hit"])


if __name__ == "__main__":
    unittest.main()
