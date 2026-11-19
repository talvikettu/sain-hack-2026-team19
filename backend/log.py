"""Process-wide logging setup for stand and local runs."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("team19.case4")
# Uvicorn configures only its own loggers. Add one plain handler so the
# aggregate run/chat metrics are visible on the stand without request data.
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
logger.setLevel(LOG_LEVEL)

if os.environ.get("WEB_CONCURRENCY", "").strip() not in ("", "1"):
    logger.warning(
        "WEB_CONCURRENCY is set; this backend requires one worker because "
        "runtime, sessions and quota are process-local"
    )
