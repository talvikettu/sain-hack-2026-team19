# Backend (case 4)

FastAPI app for the stand. The organizer contract is `team19/README.md`; the
answer format is `Case4Answers.schema.json`.

## Code layout

`main.py` is only the stable Uvicorn entrypoint (`uvicorn main:app`). The
implementation lives in `backend/`:

| Module | Responsibility |
|---|---|
| `backend/config.py` | environment limits, deadlines, cache and dataset caps |
| `backend/state.py` | active runtime registry, index lifecycle, shared LLM client |
| `backend/organizer.py` | `getData` download, socket deadlines, dataset contract validation |
| `backend/cache.py` | content-addressed completed-run cache |
| `backend/answering.py` | retrieval, retry policy, fallbacks, output validation |
| `backend/chat.py` | browser sessions, quota ceilings, idempotent turns, `/session` and `/chat` |
| `backend/routes_run.py` | official `/run` handler and transactional publication |
| `backend/catalog.py` | `/experts` and `/source` |
| `backend/app.py` | FastAPI assembly, admission middleware, health and static client |

Tests patch the owning module (`backend.state`, `backend.chat`, …) instead of
`main`. Move a function only together with the state it owns; runtime, sessions
and quota counters must stay process-local.

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
| `POST /session` | issue a short-lived browser token for `/chat` |
| `POST /run` | complete online/final run; fetches `getData` with the incoming Bearer token |
| `GET /experts` | the three experts and the active `dataset_generation` |
| `POST /chat` | one interactive client question: `{expert_id, message, session_id?, turn_id?}` |
| `GET /source/{article_id}/{section_id}?generation=...` | real article/section text for the source card |

`/experts`, `/chat` citations and `/source` include `dataset_generation`. The
value is derived from the dataset content digest. Send it back to `/source`;
if a new dataset has been published since the citation was produced, `/source`
returns `409` instead of showing text from a different corpus. `/run` is
serialized with other `/run` requests, while `/chat`, `/health`, `/experts` and
`/source` never wait for the official run.

`/chat` is not public. Call `POST /session` first, then send the returned
`token` as `Authorization: Bearer <browser-token>` with every `/chat`
request:

```bash
curl -s -X POST http://127.0.0.1:8000/session
# {"session_id":"...","token":"...","user_id":"...","expires_in":7200}

curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Authorization: Bearer <browser-token>' \
  -H 'Content-Type: application/json' \
  -d '{"expert_id":"clinician","message":"Как кофеин влияет на сон?"}'
```

This browser token is separate from the organizer `/run` token and
from the LLM key. The server stores only a SHA-256 hash of it, binds it to the
active dataset and to a server-selected demo user, and drops it when the
session expires or a different dataset is published. The client cannot send a
`user_id` or choose another dataset.

`/chat` returns clean prose in `text` and source-card metadata in `citations`.
Display the sources separately from the message. Inline citation IDs remain
in the `/run` response for organizer evaluation. The demo user is selected
server-side; the client cannot choose whose dataset history is sent to the
model. Chat uses only the current browser session's messages, never the
dataset's stored `chat_history`. Sessions keep at most 40 messages and
20,000 characters, the store is capped at 100 sessions, and chat is limited
to 12 requests/minute per session and 30/minute overall. Session issuance has
its own 20/minute bucket, so opening a tab never consumes a question. When all
100 slots are held, only a session idle for at least five minutes is replaced;
otherwise `/session` returns `503` instead of silently dropping a fresh
browser. `expires_in` is an idle timeout and an authenticated `/chat` refreshes
it.

Send an optional `turn_id` with every question. Retrying a lost request with
the same `turn_id` returns the stored response without another LLM call; using
the same ID with different content returns `409`. Only one turn per browser
session generates at a time (a second concurrent turn gets `409`), while
different browsers and the official run proceed independently. Responses
include `turn_id` and `dataset_generation`.

Chat is always limited by per-call ceilings: every proxy call re-checks the
allowed call count and token count atomically inside the client lock. It can
never spend the calls or tokens reserved for a retry of the official `/run`,
including after a run has already completed.

`POST /run` returns exactly:

```json
{"requests": [{"record_id": "...", "text": "...", "refused": false,
               "citations": [{"article_id": "art_10", "section_id": "s002"}]}]}
```

No extra fields. The token, dataset URL and dataset content are not logged.
`group_id` is used to keep the citations of a three-expert question identical.

Before anything is indexed or published, `/run` validates every field the
backend consumes: unique user/expert/article/section/record IDs, non-empty
section text, known expert personas, timezone-aware timestamps, matching
request experts, a bounded request count and the input size caps. A contract
violation returns `502` with only the field path in the log; the dataset
content is never echoed. The dev dataset validates unchanged.

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

Only complete, schema-valid runs are cached. If any request used a fallback
answer, a group needed the local retrieval fallback, hit the deadline or
exhausted the local budget, the result is returned but not stored. Every
generated answer is checked at the backend boundary: exact fields, matching
`record_id`, non-empty text, boolean `refused`, at least one unique citation
and only citations that exist in the submitted dataset. A malformed answer
becomes a counted local fallback; a malformed batch is never cached or
returned.

The cache has four entries, a 4 MiB per-entry cap and a 16 MiB aggregate cap.
A same-content `/run` reuses the active runtime, preserving its browser
sessions and index. A different dataset is downloaded, validated and indexed
as a candidate; it is published only after the run succeeds or a valid cache
entry is found. A failed fetch, validation, index build or generation leaves
the previously active runtime and its `/chat` sessions untouched. Changing the
dataset content invalidates old browser tokens; the same content does not.
Content-addressed index files retain the active one and one previous file, so a
failed build or a rapid dataset switch cannot fill `/tmp`.

A transient LLM failure (timeout, HTTP 429/5xx, unavailable proxy) is retried
with the configured delays and a server-provided `Retry-After` (capped at
120 s). After a failed record, the next record waits for a bounded cooldown
(4 s, doubling to 30 s) and probes the model again, so a short outage does not
disable generation for the rest of the run. Only a confirmed local budget
stop, an authentication failure (HTTP 401/403), or a malformed proxy response
whose token usage cannot be accounted for stops further generation; the rest
of the run is then returned as fallbacks. The proxy client classifies failures
with typed `ChatTerminalError` / `ChatTransientError` status codes; message
checks remain only as a fallback for non-client errors.

The dataset download, index build and generation share one 42-minute `/run`
deadline, leaving five seconds for response serialization inside the stand's
45-minute limit. Socket reads are bounded by the remaining deadline, so a
slow organizer API cannot hold the request forever. A second `/run` returns
`503` with `Retry-After` immediately instead of waiting behind the first.
If one group's retrieval fails, `/run` selects a topical section locally
instead of aborting the whole run.

Before generation, /run searches the original question and its recognized topics.
It collects up to 40 candidates per main query and reranks the first 20.
Supplementary searches can widen coverage for sleep timing and evening light.
The sleep-hygiene query for sleep schedules retrieves up to 10 candidates.
The reranker favors relevant sections containing findings or conclusions and
distinguishes those sections from abstracts that only describe the methods.
For each `group_id`, it selects a common section and passes it to all experts.
The same section is available for a degraded response if generation fails.
A degraded response never invents text:

- for an ordinary question with a topically matching section, `/run` returns
  short verbatim sentences from that section, marks the answer as not refused,
  and adds the required citation;
- for a diagnostic, treatment, dosage or prompt-injection request, it returns
  a refusal (`refused: true`) with a reading suggestion and a valid citation;
- if the cited section has no short, safe, relevant sentence, it refuses rather
  than returning a non-refusal with no substance.

If the group search fails before a section is selected, a local index search
retries without generation; if the index is unavailable too, a keyword scan
over the dataset can select a section. A zero-overlap section is not cited.
When no topical section exists at all, the schema-required citation is filled
from the first available section, but the answer stays a refusal and the case
is counted as `unsupported` in the run log. That group is never sent to the
model with unrelated evidence. Fallbacks do not spend LLM calls.

Cached results live only in the process; a stand restart starts with an empty
cache. `RUN_CACHE_VERSION` in `backend/config.py` must be bumped when prompts
or answer logic change in a way that should invalidate old cached answers.

## Concurrency and process model

Run **one** Uvicorn worker. The active runtime, browser sessions, idempotency
records and the shared LLM counters are process-local; `start.sh` starts one
worker and the app logs a warning if `WEB_CONCURRENCY` asks for more. Health,
experts, source lookup and chat are served from the active runtime while an
official run builds its candidate in the background, so they do not wait for
the 42-minute run. Admission limits the number of concurrent mutating HTTP
requests to 16; excess `/run`, `/chat` or `/session` calls receive `503`
before a worker thread is allocated.

Every response carries `X-Request-ID` (the caller's safe ID or a generated
16-hex ID). Failure logs contain the request ID, stage (`fetch`, `validate`,
`build`, `generate`, `commit`, `publish`), elapsed time, and call/token deltas
for that request. They do not contain dataset codes, tokens, prompts,
questions, source text or model output.

## Tests

The automated tests do not call the organizer LLM proxy and spend no team
quota:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
```

`tests/test_backend_contract.py` covers the actual FastAPI handlers, the exact
`/run` response shape, identical citations inside every three-expert group,
safe extractive and refusal fallbacks, unrelated-evidence rejection,
retrieval-failure recovery, transient LLM recovery, cache hits and runtime
activation, non-caching of partial runs, the single process-wide client,
browser-session issuance and ownership, bounded chat storage and rate limits,
and evidence selection before generation. It also covers transactional
candidate publication, same-content runtime reuse, dataset contract rejection,
backend-side answer validation, per-call chat ceilings, turn idempotency,
generation-scoped sources, cache size bounds, session issuance limits, request
IDs and terminal proxy errors. The tests use synthetic data, plus the dev
dataset when it is present locally. The dataset and reference files are local
evaluation material and are not part of the submitted package.

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
