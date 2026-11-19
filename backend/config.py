"""Environment-driven limits shared by the backend modules."""

from __future__ import annotations

import os
import re
from pathlib import Path

TEAM = os.environ.get("TEAM", "team19")
CASE_ID = 4
RUN_BUDGET_SECONDS = 42 * 60  # the stand waits 45 minutes
CHAT_BUDGET_SECONDS = 5 * 60
CHAT_MAX_SESSIONS = 100
CHAT_MAX_MESSAGES = 40
CHAT_MAX_CHARS = 20_000
CHAT_SESSION_TTL_SECONDS = 2 * 60 * 60
CHAT_RATE_LIMIT = 12
CHAT_GLOBAL_RATE_LIMIT = 30
CHAT_RATE_WINDOW = 60
CHAT_SESSION_ISSUE_LIMIT = 20
CHAT_SESSION_EVICT_GRACE_SECONDS = 5 * 60
CHAT_TURN_CACHE_MAX = 200
# The official run has priority. Chat is checked against a hard ceiling at
# every individual proxy call, so a repair attempt can never consume the
# calls/tokens that a retry of the official run needs.
RUN_RESERVED_CALLS = 105
RUN_RESERVED_TOKENS = 220_000
INDEX_PATH = Path(os.environ.get("CASE4_INDEX_PATH", "/tmp/case4-index.sqlite3"))
INDEX_MAX_RETAINED = 2

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CALLS = 150
LLM_TOKEN_BUDGET = 260_000
LLM_MIN_INTERVAL_SECONDS = 2.1
# A transient proxy outage must not permanently disable generation. After a
# failed record, pause before the next probe so later records can recover.
TRANSIENT_COOLDOWN_BASE_SECONDS = 4.0
TRANSIENT_COOLDOWN_MAX_SECONDS = 30.0
RUN_CACHE_VERSION = "10"
RUN_CACHE_MAX_ENTRIES = 4
RUN_CACHE_MAX_ENTRY_BYTES = 4 * 1024 * 1024
RUN_CACHE_MAX_BYTES = 16 * 1024 * 1024
MAX_DATASET_BYTES = 32 * 1024 * 1024
DATASET_READ_CHUNK = 64 * 1024
MAX_DATASET_REQUESTS = 2000
RUN_SERIALIZATION_RESERVE_SECONDS = 5
MAX_INFLIGHT_REQUESTS = 16
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_REQUEST_ID_HEADER = "X-Request-ID"
