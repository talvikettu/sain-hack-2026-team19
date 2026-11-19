# Sain Hackathon — Case 4: Expert as a Playlist

This repository contains expert prompts, a scientific-article preprocessing pipeline, a RAG agent, the FastAPI stand backend, and local evaluation tools for case 4 of the Sain × School 21 hackathon.

The application answers sleep-related questions through three fictional experts: a clinical sleep specialist (`clinician`), an experimental biohacker (`biohacker`), and a common-sense nutritionist (`nutritionist`). Answers must rely on the organizer-provided articles and include original article/section references.

## Start here

- **Configure the HTTP backend:** [Backend guide](docs/BACKEND.md).
- **Understand the pipeline:** [RAG preprocessing design](docs/RAG_PREPROCESSING_PIPELINE_RU.md).
- **Read the end-to-end architecture and latest changes:** [RAG system guide](docs/RAG_SYSTEM.md).
- **Run preprocessing or retrieval:** [Preprocessing guide](rag_preprocessing/README.md).
- **Run the agent on example questions:** [Agent guide](rag_agent/README.md).
- **Check retrieval coverage:** [Offline retrieval check](#offline-retrieval-check).
- **Understand the token optimizations:** [Optimization notes](docs/RAG_TOKEN_OPTIMIZATION_RU.md).

This navigation page is in English. Most detailed project documents and example conversations are currently in Russian.

## Repository map

| Directory | Contents |
|---|---|
| [`prompts/`](prompts/) | Compact system and expert prompts, plus the original full versions. |
| [`rag_preprocessing/`](rag_preprocessing/) | Parsing, cleaning, structural/semantic chunking, metadata, SQLite storage, BM25, optional embeddings, vector search, and context assembly. |
| [`rag_agent/`](rag_agent/) | Expert selection, history filtering, retrieval orchestration, LLM proxy client, response validation, and the example-run CLI. |
| [`backend/`](backend/) | FastAPI routes, runtime state, organizer access, run cache, and chat sessions. `main.py` is the `uvicorn main:app` entrypoint. |
| [`tests/`](tests/) | Synthetic unit/integration tests. These do not call an external LLM or consume team quota. |
| [`data/`](data/) | Organizer dev dataset, reference labels, and API schemas for offline development. Git archives for the stand omit this directory. |
| [`docs/`](docs/) | Pipeline architecture and token-optimization documentation. |
| [`output/pdf/`](output/pdf/) | Russian and English technical-constraints PDFs. |
| `team19/` | Local organizer kit, including credentials and deployment templates. Ignored by Git; never commit this directory or secret values copied from it. |
| `tmp/` | Generated indexes, previews, temporary scripts, and local run results. Ignored by Git. |

The original case specification is available as a PDF in the repository root.

### Prompts

| File | Purpose |
|---|---|
| [`system_compact.md`](prompts/system_compact.md) | Default runtime rules: grounding, medical boundaries, numbered citations, and compact JSON output. |
| [`system.md`](prompts/system.md) | Original detailed system prompt, retained for the `full` profile and comparison. |
| [`somnologist.md`](prompts/somnologist.md) | Original clinical sleep specialist metaprompt; corresponds to expert ID `clinician`. |
| [`biohacker.md`](prompts/biohacker.md) | Original experimental biohacker metaprompt. |
| [`nutritionist.md`](prompts/nutritionist.md) | Original common-sense nutritionist metaprompt. |

The default `compact` profile combines `system_compact.md` with one local expert prompt selected by `expert_id`. The compact role files follow the organizer's descriptions in `data/experts/`. The `full` profile retains the original persona files and earlier prompt assembly.

### Dev materials

See [the data guide](data/README.md) for the complete inventory. Key inputs are:

- [`articles.json`](data/articles.json): 20 articles with 609 original sections in the current dev corpus.
- [`experts.json`](data/experts.json): the three official fictional expert descriptions.
- [`requests.json`](data/requests.json): 60 original example requests, including target expert IDs.
- [`chat_history.json`](data/chat_history.json): available user conversation history.
- [`schemas/`](data/schemas/): organizer JSON schemas.

`data/dataset.json`, `data/reference.json`, and `data/raw/case4-dev-v2.json`
are the organizer's dev fixtures. Use reference labels only to score the
results of offline checks. The deployment archive excludes `data/`.

## Processing flow

```text
Articles → parsing/cleaning → sections or smaller chunks → metadata → search index

Question + expert_id
    → relevant user history, when needed
    → BM25 retrieval and evidence selection
    → compact rules + local expert prompt + evidence + question
    → organizer LLM proxy
    → JSON/reference validation
    → answer with article_id / section_id citations
```

The preprocessing package supports optional remote embeddings and hybrid retrieval. The current compact profile uses **local SQLite/BM25**. It combines the original question with topic-specific searches and preserves the options and constraints in comparisons where recognized. The reranker favors relevant sections with findings or conclusions. Embedding access must be confirmed and explicitly configured; no model weights are downloaded.

## Offline retrieval check

The offline check processed **60 dev requests across 20 groups**. It found **183 of 183 allowed request–article pairs** in the candidate pool, with complete coverage for **60 of 60 requests**. It selected an allowed source for **60 of 60 requests**. These are pairs from the organizer's dev reference, not 183 distinct articles. The reference is used only to score retrieval after the search.

| Metric | Before the wider search | Current |
|---|---:|---:|
| Allowed request–article pairs found | 153/183 | **183/183** |
| Requests with every allowed article in the candidate pool | 42/60 | **60/60** |
| Requests with an allowed article selected for the prompt | 60/60 | **60/60** |

The wider search collects up to **40 fragments per main query** and reranks the first **20**. Supplementary searches expand coverage for sleep timing and evening light. A focused sleep-hygiene query adds up to **10** candidates for schedule questions.

The **`/run`** path sends **one shared section** to the experts in a group and returns the organizer's citation fields. **`/chat`** can use up to three fragments within the compact context budget and returns source-card metadata separately from the displayed text. Compact output checks cover citation structure, external links, HTML, direct appeals to the evaluator, and long verbatim copies of system instructions. These checks do not measure medical correctness or resistance to all prompt injections.

Reproduce the retrieval check without credentials or model calls:

```powershell
python scripts/evaluate_retrieval.py --output tmp/retrieval_evaluation
```

The script saves a local Markdown table and JSON with allowed, found, and missing articles for each request, plus the selected section. These results measure dev retrieval coverage. Generation quality and performance on the closed organizer dataset remain unmeasured by this check.

## Quick start

Run commands from the repository root with Python 3.11 or newer. Install the backend dependencies once with `pip install -r requirements.txt`; for the test suite use `pip install -r requirements-dev.txt` (it includes the runtime requirements and `httpx2` for FastAPI's test client). The preprocessing and agent CLI commands below otherwise use the Python standard library.

### 1. Run tests — no API calls

```powershell
python -m unittest discover -s tests -v
```

### 2. Build a local index

```powershell
python -m rag_preprocessing build --input data/articles.json --output tmp/rag/index.sqlite3
```

For additional sentence-based splitting, add `--mode structural`. See the preprocessing guide for chunk sizes, overlap, optional semantic splitting, and embedding configuration.

### 3. Preview the first 10 requests — no API calls

```powershell
python -m rag_agent --limit 10 --dry-run --profile compact --output tmp/rag_preview
```

Inspect `tmp/rag_preview/preview.json` locally. It contains the prepared prompts and evidence, without credentials. Use `--limit 60` to prepare all dev requests without spending LLM tokens.

### 4. Run the first 10 questions against the LLM

Configure `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in the environment using the organizer-issued settings. Keep actual credentials out of commands shared in chat, source files, and reports.

```powershell
python -m rag_agent --limit 10 --profile compact --token-budget 60000 --output tmp/rag_demo_new
```

This command consumes the team's shared LLM quota. Select a new output directory: the CLI refuses to overwrite an existing run. Results are checkpointed after each question as `report.md`, `run.json`, and `answers.json`. On the first error, the run stops and preserves partial results.

## Integration and hosting limits

FastAPI returns JSON health status at `/` and `/health`. The [backend guide](docs/BACKEND.md) lists the API routes, deployment settings, and quota limits.

`POST /run` receives `{"case": 4, "dataset_code": "..."}`, downloads
`getData` once with the incoming organizer Bearer token, validates the dataset
contract, and returns the full `Case4Answers` JSON in the same response. The
download, candidate index build, retrieval and generation share a 42-minute
deadline inside the stand's 45-minute limit. The candidate dataset is published
only after the run succeeds or a valid cache entry is found, so a failed run
cannot replace the runtime serving `/chat`. A transient proxy failure pauses
and retries later records instead of disabling generation for the run. If one
group cannot use the model, the backend returns a safe extractive answer or a
refusal with a real citation rather than invalid JSON.

`/chat` uses a separate browser credential issued by `POST /session`; send it
as `Authorization: Bearer <browser-token>`. The organizer `/run` token and the
LLM key are never accepted as browser credentials and are never logged. Every
chat response carries a `dataset_generation`; pass it to `/source` to avoid
showing a section from a newer dataset. Optional `turn_id` values make a lost
turn replayable without spending the model budget twice. `/chat` does not wait
for an official `/run`, and chat calls are capped atomically so the run's
quota reserve always survives.

The organizer backend environment has Python 3.11, 512 MB RAM, one CPU core, no GPU, and outbound access restricted to the hackathon API and LLM proxy. `/tmp` is limited to 256 MB and is not persistent across deployments. Rebuild or restore the index after deployment. Run exactly one Uvicorn worker: the runtime, browser sessions and LLM counters are process-local, and `start.sh` starts one worker. The shared chat allowance is 30 requests/minute and 300,000 tokens/day; the CLI's local token budget does not account for other team members' usage.

See the [technical-constraints PDFs](output/pdf/) and the local organizer README for deployment details. Keep `team19/`, environment files, generated SQLite indexes, and run reports out of commits.
