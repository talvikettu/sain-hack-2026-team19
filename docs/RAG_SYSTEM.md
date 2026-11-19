# RAG system architecture and recent changes

This guide describes the implementation in `dinoden` through commit `44b2309`. The application answers questions about sleep in Russian using the scientific corpus supplied for Sain Case 4. Retrieval runs locally. Answer generation uses the configured organizer proxy.

## Request flow

```text
Organizer articles
  -> normalize text and retain original source IDs
  -> build or reuse the SQLite index

Question + expert ID + relevant user history
  -> original, topical, and supplementary searches
  -> BM25 ranking and section reranking
  -> select and pack evidence
  -> render the system prompt, expert role, and request data
  -> generate through the organizer proxy
  -> clean and validate the answer and citations
  -> return the evaluation response or chat text with source cards
```

| Component | Responsibility |
|---|---|
| [rag_preprocessing/pipeline.py](../rag_preprocessing/pipeline.py) | Article normalization, chunking, SQLite index construction, BM25, and context packing |
| [rag_agent/retrieval.py](../rag_agent/retrieval.py) | Search expansions, candidate merging, and reranking |
| [rag_agent/history.py](../rag_agent/history.py) | History selection and shared group queries |
| [rag_agent/agent.py](../rag_agent/agent.py) | Synchronous entry points, preparation, evidence cache, and orchestration |
| [rag_agent/prompts.py](../rag_agent/prompts.py) | Profile and expert selection, plus model message assembly |
| [rag_agent/generation.py](../rag_agent/generation.py) and [validation.py](../rag_agent/validation.py) | Per-answer state, repair, decoding, validation, and cleanup |
| [rag_agent/client.py](../rag_agent/client.py) and [async_agent.py](../rag_agent/async_agent.py) | Proxy transport and an independently usable async interface |
| [main.py](../main.py) | HTTP integration, official batches, chat history, and source lookup |
| [scripts/evaluate_retrieval.py](../scripts/evaluate_retrieval.py) | Offline article coverage and selected evidence report |

## Corpus ingestion and index lifecycle

The indexer reads the article list from the supplied JSON and normalizes text while retaining original article and section IDs, section headings, offsets, and article metadata. An ID such as `art_12 / s015` identifies section `s015` within article `art_12`. A section ID can recur across articles, so citations use both IDs. The dev corpus contains 20 articles and 609 original sections.

The serving path indexes whole sections by default. Structural mode can split longer passages on sentence boundaries with a configurable overlap. Semantic chunking requires a configured embedding provider. The library can also combine BM25 and vector rankings with reciprocal rank fusion. The agent in this branch uses the local lexical path. It builds and queries SQLite without an embedding provider.

The index stores corpus and configuration fingerprints and can reuse a matching file after an integrity check. A rebuild writes a temporary SQLite file, validates it, and replaces the target after successful construction. The application owns runtime publication and the lifetime of the index. The current HTTP branch uses a shared configured index path. Immutable runtime revisions and index leases belong to future backend integration.

The organizer's reference answers and evaluator prompts are excluded from the search corpus and model inputs. The evaluation script reads the reference locally to score the articles found by retrieval.

## Retrieval, history, and evidence

1. **Select history.** The agent filters by user ID and the request timestamp. In `compact`, a phrase matcher enables history when the question appears to refer to earlier turns, with a limit of four messages and 2,000 characters. Recent user turns can supply search terms. Scientific claims must come from article passages. The phrase matcher can miss implicit follow-ups.
2. **Build queries.** The original question participates in search alongside recognized topics. A small Russian-to-English dictionary bridges Russian questions and English papers. A screen-versus-paper comparison keeps both options and incorporates daytime lighting when mentioned. Supplementary queries cover topics such as delayed sleep timing, evening light, and sleep hygiene.
3. **Rank and merge.** The primary search takes up to 40 BM25 fragments per query and reranks the first 20. A focused sleep-hygiene search takes up to 10. Section weights favor relevant findings and conclusions over methods-only passages. For a recognized screen-versus-book comparison, passages mentioning both options take priority when available. Ranked lists are interleaved and duplicate chunks removed.
4. **Pack context.** Overlapping spans from the same section are merged and kept with their original IDs. `compact` accepts up to three fragments within 4,500 characters. `full` accepts up to five within 12,000 characters. A block that does not fit can be skipped. The budget counts characters, not model tokens.

The retrieval pool and model context have different purposes. The broad pool improves article coverage. The context contains only selected evidence. The official `/run` currently chooses **one shared section per group** before generating any expert answers. All experts in the group must use the same shared citation set. Questions with independently supported parts can benefit from a future multi-passage selector. The current branch has no such selector on `/run`.

## Prompt profiles and expert selection

**`compact` is the default for the agent, CLI, `/run`, and `/chat`.** Both profiles instruct the model to answer in Russian.

| Expert | Compact file | Distinct contribution |
|---|---|---|
| `clinician` | [somnologist_compact.md](../prompts/somnologist_compact.md) | State the conclusion, explain the mechanism and strength of the evidence, and identify limits of applicability |
| `biohacker` | [biohacker_compact.md](../prompts/biohacker_compact.md) | When useful, propose a safe, measurable observation with one variable and a way to interpret the result |
| `nutritionist` | [nutritionist_compact.md](../prompts/nutritionist_compact.md) | Connect a supported conclusion to one practical substitution that fits an existing routine |

[The compact system prompt](../prompts/system_compact.md) owns language, evidence policy, medical boundaries, refusal semantics, and JSON format. The trusted system message gets one locally maintained persona selected by known `expert_id`. Each persona follows the [organizer's expert descriptions](../data/experts/). Questions, history, headings, and article text are serialized as separate user-role data. The compact path no longer places raw dataset role or style strings into the system message. When a shared citation set is required, the application adds it to the trusted instructions.

The `full` option remains available with `--profile full` or `prompt_profile="full"`. It uses the longer system and persona files, organizer role and style data, and a different model-facing JSON contract. The public answer schema remains the same.

## Generation, validation, and delivery

- **Proxy.** The client reads the endpoint, model, and credentials from environment variables. It checks local call and token limits before dispatch, applies the request deadline and minimum call interval, and records actual usage from provider responses. The process-level limits complement the team's shared quota.
- **Compact wire format.** The model returns `text`, `refused`, and `refs`. Numbered citations such as `[1]` resolve to original article and section IDs in code. The externally returned answer has `record_id`, `text`, `refused`, and `citations`.
- **Validation and cleanup.** The agent checks the finish reason, required fields and types, valid numbered references, inline citation agreement, and any fixed citation set. It cleans formatting artifacts. Compact output checks reject patterns such as HTML, outside links, direct appeals to the evaluator, and long verbatim copies of system text.
- **Repair.** One additional generation can follow a failed output check. The original context remains in place and a short validation reason is added. The rejected text is not replayed. A wrong refusal decision with the correct boolean type can pass. Backend transport retries are a separate policy from model-output repair.
- **Official batch.** `/run` preselects group evidence and returns a complete answer in the original request order. When generation fails and evidence exists, it can return a degraded message with the retrieved source. An empty evidence set cannot satisfy the required citation schema. Degraded results are excluded from the success cache.
- **Browser chat.** `/chat` returns readable prose alongside structured source metadata for source cards. Technical inline IDs are removed from the displayed answer. The organizer response retains its inline citations.

Structural validation confirms that citations belong to the supplied evidence. It cannot prove that a cited passage supports every claim, that a refusal flag matches the prose, or that an answer is medically correct. Medical heuristics are advisory. The compact filters cover specific output patterns and prompt separation reduces risk, while model compliance remains an empirical question.

## Sync and async entry points

The existing backend and CLI use `RAGAgent.answer()`. The standalone `AsyncRAGAgent` adds awaitable `prepare()`, `prepare_group()`, and `answer()` with the same prompt and completion logic. The default compatibility adapter uses one worker for SQLite preparation and validation and one for the existing blocking proxy client. Retrieval can progress while a model call waits. An injected async provider can be used instead, with its own quota policy and configurable concurrency.

The standalone async agent bounds admission and applies a monotonic deadline while waiting for work slots. Cancellation stops the caller waiting for a blocking operation. That operation holds its slot until it actually finishes. Use the adapter inside its async context manager. A [usage example](../rag_agent/README.md) is available in the agent guide. The HTTP routes in this branch still use the synchronous interface, so these changes do not establish concurrent backend throughput.

## Latest changes and measured results

| Revision or run | Change or result |
|---|---|
| `1f4555e` | Search comparison expansion and section reranking improved the chosen source from 57/60 to 60/60 on dev. Widening the candidate pool and adding focused searches increased coverage from 153/183 to 183/183 allowed request–article pairs. |
| `44b2309` | Split history, prompts, generation and validation into focused modules. Added the standalone async API and kept the synchronous contract. Added three compact persona files and consolidated shared policy in the compact system prompt. |
| Offline prompt audit | Estimated average system-message size fell from 974 to 668 tokens with `o200k_base`, about 31.4%. All 60 full-profile previews matched the earlier version. The tokenizer used by the organizer proxy was not confirmed. See the [prompt audit](../prompt_audit.md). |
| Latest live compact dev run | Ten answers from 11 calls including one repair. The proxy reported 20,922 tokens, with 15,057 input and 5,865 output. The active run time was 101.9 seconds. All ten cited permitted articles. Eight refusal flags matched the dev labels. |

The 183 figure counts **allowed request–article pairs**, not distinct articles. All allowed articles were found for 60/60 dev requests across 20 groups. The wider search kept the same chosen section on all 60. Coverage on this labeled set does not measure the strength of individual claims or performance on the hidden dataset.

Manual review of the latest live run found two refusal mismatches. One answer declined to distinguish diagnoses in the prose but kept `refused=false`. Another acknowledged missing evidence for a method without giving the explicit refusal expected by the fixture. Two heat-related answers used a misleading translation of wakefulness. The live run did not include the organizer's medical validation or scoring. The full dialogue and run records are local artifacts.

## Reproduce the checks

Run these commands from the project root with Python 3.11 and the backend test dependencies installed:

```powershell
python -m unittest discover -s tests
python scripts/evaluate_retrieval.py --output tmp/rag_retrieval_check
python -m rag_agent --profile compact --limit 10 --dry-run --output tmp/rag_preview
```

The tests and both diagnostics run without a model. The retrieval check writes Markdown and JSON with allowed, found, and missing articles plus the section selected for each request. The preview writes the prepared messages for local inspection. A real CLI run requires the organizer proxy settings and consumes quota. Keep credentials, raw previews, and local run outputs out of Git.
