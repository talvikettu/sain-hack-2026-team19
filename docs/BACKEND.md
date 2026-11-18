# Backend (case 4)

FastAPI app for the stand. The organizer contract is `team19/README.md`; the
answer format is `Case4Answers.schema.json`.

## Run locally

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export HACK_API_URL=https://hack-api.be-healthy.tech
export HACK_TEAM_TOKEN="$(cat ~/Documents/team19/api.token)"
export LLM_BASE_URL=https://bifrost.be-healthy.tech/v1
export LLM_API_KEY="$(cat ~/Documents/team19/llm.key)"
export LLM_MODEL=gpt-6-luna

./start.sh
```

Do not put the token or key values into the repository, logs, or chat.

## Endpoints

| Method | Purpose |
|---|---|
| `GET /` | JSON health page, or `web/dist/index.html` when the client is present |
| `GET /health` | `{"team":"team19","status":"ok"}` |
| `POST /run` | complete online/final run; fetches `getData` with the incoming Bearer token |
| `GET /experts` | the three experts from the loaded dataset |
| `POST /chat` | one interactive client question: `{expert_id, message, session_id?, user_id?}` |
| `GET /source/{article_id}/{section_id}` | real article/section text for the source card |

`/chat` returns clean prose in `text` and source-card metadata in `citations`.
Display the sources separately from the message. Inline citation IDs remain
in the `/run` response for organizer evaluation.

`POST /run` returns exactly:

```json
{"requests": [{"record_id": "...", "text": "...", "refused": false,
               "citations": [{"article_id": "art_10", "section_id": "s002"}]}]}
```

No extra fields. The token, dataset URL and dataset content are not logged.
`group_id` is used to keep the citations of a three-expert question identical.

## Quota and repeated runs

The shared LLM quota is 30 requests/minute and 300k tokens/day for the whole
team. The backend therefore keeps one `ChatClient` per process with these
limits:

| Limit | Value |
|---|---:|
| LLM calls per process | 150 |
| LLM tokens per process | 260,000 |
| Minimum interval between calls | 2.1 s |

Counters survive repeated `POST /run` calls. A fully successful run is cached
in process memory, so calling `/run` again with the same dataset content costs
no LLM tokens. The key is a SHA-256 digest of `requests`, `experts`,
`chat_history`, `articles`, `users`, the cache version and `LLM_MODEL`. It
excludes `reference`, `meta`, the dataset code and the API token. A cache hit
still calls `getData` with the incoming Bearer token, so access is validated on
every request.

Only complete runs are cached. If any request used a fallback answer, hit the
deadline or exhausted the local budget, the result is returned but not stored.

Before generation, /run searches the original question and its recognized topics.
It collects up to 40 candidates per main query and reranks the first 20.
Supplementary searches can widen coverage for sleep timing and evening light.
The sleep-hygiene query for sleep schedules retrieves up to 10 candidates.
The reranker favors relevant sections containing findings or conclusions and
distinguishes those sections from abstracts that only describe the methods.
For each `group_id`, it selects a common section and passes it to all experts.
The same section is available for a degraded response if generation fails.
A degraded response identifies the section as a reading suggestion and does not
claim a scientific conclusion. If retrieval finds no section, the format's
required citation cannot be supplied and the run reports an error.

Cached results live only in the process; a stand restart starts with an empty
cache. `RUN_CACHE_VERSION` in `main.py` must be bumped when prompts or answer
logic change in a way that should invalidate old cached answers.

## Tests

The automated tests do not call the organizer LLM proxy and spend no team
quota:

```bash
python -m unittest discover -s tests
```

`tests/test_backend_contract.py` covers the exact `/run` response shape,
identical citations inside every three-expert group, fallback answers, cache
hits, non-caching of partial runs, the single process-wide client, and
evidence selection before generation. The tests use synthetic data, plus the
dev dataset already present in the repository when it is available.

An end-to-end contract check can replace the LLM with a local deterministic
stub and run `scripts/dev_run.py` against it. That verifies the HTTP path,
grouping, cache and `/case/4/validate`, but not `gpt-6-luna` answer quality.
The stub is for local development only and is not part of the repository.

## Local client development

Before `/run` has loaded a dataset, the client endpoints are empty. To work on
the frontend against the dev corpus:

```bash
export DEV_DATASET_CODE=case4-dev-v2
```

The backend then lazy-loads the dev dataset with `HACK_TEAM_TOKEN`. On the stand
`DEV_DATASET_CODE` is unset; `/run` supplies the online/final dataset.

When the frontend is ready, put its build in `web/dist/` and commit it: the
stand installs Python dependencies only and does not run Node.

## Before the final submission

The case's customer wishes (`Кейс 4.pdf`, section 6) still need README numbers:

- time from adding a new article with one command to experts citing it;
- cost of adding a new expert.

These are documentation tasks for the final README, not runtime code.
