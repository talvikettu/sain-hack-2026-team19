# Sain Hackathon — Case 4: Expert as a Playlist

This branch (`frontend`) contains the web client, FastAPI backend, expert prompts, a scientific-article preprocessing pipeline, a RAG agent, and local evaluation tools for case 4 of the Sain × School 21 hackathon.

The application answers sleep-related questions through three fictional experts: a clinical sleep specialist (`clinician`), an experimental biohacker (`biohacker`), and a common-sense nutritionist (`nutritionist`). Answers must rely on the organizer-provided articles and include original article/section references.

## Start here

- **Run or build the frontend:** [Frontend guide (RU)](web/README.md).
- **Configure the HTTP backend:** [Backend guide](docs/BACKEND.md).
- **Understand the pipeline:** [RAG preprocessing design](docs/RAG_PREPROCESSING_PIPELINE_RU.md).
- **Run preprocessing or retrieval:** [Preprocessing guide](rag_preprocessing/README.md).
- **Run the agent on example questions:** [Agent guide](rag_agent/README.md).
- **Read the latest generated answers:** [First 10 questions — compact run](reports/rag_compact_first10/report.md).
- **Review accuracy, latency, and token usage:** [Latest evaluation](reports/rag_compact_first10/EVALUATION.md).
- **Understand the token optimizations:** [Optimization notes](docs/RAG_TOKEN_OPTIMIZATION_RU.md).

This navigation page is in English. Most detailed project documents and example conversations are currently in Russian.

## Repository map

| Directory | Contents |
|---|---|
| [`prompts/`](prompts/) | Original system and expert prompts, plus the compact runtime system prompt. |
| [`rag_preprocessing/`](rag_preprocessing/) | Parsing, cleaning, structural/semantic chunking, metadata, SQLite storage, BM25, optional embeddings, vector search, and context assembly. |
| [`rag_agent/`](rag_agent/) | Expert selection, history filtering, retrieval orchestration, LLM proxy client, response validation, and the example-run CLI. |
| [`tests/`](tests/) | Synthetic unit/integration tests. These do not call an external LLM or consume team quota. |
| [`data/`](data/) | Organizer dev materials already present in this branch: articles, expert descriptions, example requests, history, API schemas, and local evaluation references. |
| [`docs/`](docs/) | Pipeline architecture and token-optimization documentation. |
| [`reports/rag_compact_first10/`](reports/rag_compact_first10/) | Explicitly published snapshot of the latest real 10-question run: answers, usage, timing, and evaluation. |
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

The default `compact` profile uses `system_compact.md` and the selected expert's organizer-provided `role` and `speech_style` from the dataset. It does **not** concatenate the original long persona files as well. The `full` profile retains the earlier prompt assembly.

### Dev materials

See [the data guide](data/README.md) for the complete inventory. Key inputs are:

- [`articles.json`](data/articles.json): 20 articles with 609 original sections in the current dev corpus.
- [`experts.json`](data/experts.json): the three official fictional expert descriptions.
- [`requests.json`](data/requests.json): 60 original example requests, including target expert IDs.
- [`chat_history.json`](data/chat_history.json): available user conversation history.
- [`dataset.json`](data/dataset.json): the complete dev input dataset.
- [`schemas/`](data/schemas/): organizer JSON schemas.
- [`reference.json`](data/reference.json): offline evaluation reference; never use it as retrieval evidence or include it in generation prompts.

Current dev counts are not a guarantee about hidden datasets. Do not copy credentials into `data/`, expose dev/reference dumps through the client, or add hidden evaluation datasets to Git.

## Processing flow

```text
Articles → parsing/cleaning → sections or smaller chunks → metadata → search index

Question + expert_id
    → relevant user history, when needed
    → BM25 retrieval and evidence selection
    → compact rules + expert role/style + evidence + question
    → organizer LLM proxy
    → JSON/reference validation
    → answer with article_id / section_id citations
```

The preprocessing package supports optional remote embeddings and hybrid retrieval. The current agent demo uses **BM25**, not live embeddings. Embedding access must be confirmed and explicitly configured; no model weights are downloaded.

## Quick start

Run commands from the repository root with Python 3.11 or newer. The implemented packages use the Python standard library; no third-party package installation is needed for these commands.

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

Configure `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in the environment using the organizer-issued settings. Keep actual credentials out of commands shared in chat, source files, reports, and frontend code.

```powershell
python -m rag_agent --limit 10 --profile compact --token-budget 60000 --output tmp/rag_demo_new
```

This command consumes the team's shared LLM quota. Select a new output directory: the CLI refuses to overwrite an existing run. Results are checkpointed after each question as `report.md`, `run.json`, and `answers.json`. On the first error, the run stops and preserves partial results.

## Latest measured run

The committed [compact run snapshot](reports/rag_compact_first10/README.md) used the first 10 unchanged dev questions:

- **10 LLM calls, 21,273 total tokens**, with no repair calls.
- Mean generation-and-validation time: **8.455 seconds per answer**.
- Valid output structure and references to supplied evidence: **10/10**.
- Allowed article IDs and refusal flags matched the dev reference: **10/10**.

These checks do not mean that every answer is correct. The manual review found substantial issues in answers **1, 2, and 6**, including an imprecise section citation and a user/husband mix-up. Medical validation and official S/G/V scoring were not performed. Read the [evaluation](reports/rag_compact_first10/EVALUATION.md) before using these results as a quality claim.

## Integration and hosting limits

This branch includes a FastAPI backend with `/run`, `/experts`, `/chat`, and `/source/{article_id}/{section_id}`, plus a web frontend in `web/`. The committed `web/dist/` build is served by FastAPI at `/`; rebuild it after client changes and restart the backend. See the frontend and backend guides above for configuration and current session/user limitations. Deployment is not performed by creating this branch.

The organizer backend environment has Python 3.11, 512 MB RAM, one CPU core, no GPU, and outbound access restricted to the hackathon API and LLM proxy. `/tmp` is limited to 256 MB and is not persistent across deployments. Rebuild or restore the index after deployment. The shared chat allowance is 30 requests/minute and 300,000 tokens/day; the CLI's local token budget does not account for other team members' usage.

See the [technical-constraints PDFs](output/pdf/) and the local organizer README for deployment details. `team19/`, environment files, generated SQLite indexes, and temporary output must stay out of commits. The published report directory is a deliberate, limited snapshot, not a reason to commit every local run or raw dataset.
