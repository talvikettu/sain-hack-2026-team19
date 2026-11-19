"""Team19 backend for the Sain x School 21 hackathon, case 4.

Contract (team19/README.md, sample-app/main.py):
- ``requirements.txt`` and ``start.sh`` live in the repository root;
- Python 3.11, the server listens on ``0.0.0.0:$PORT``;
- ``GET /`` answers within 60 seconds;
- ``POST /run`` receives ``{"case": 4, "dataset_code": "..."}``, fetches the
  dataset with the incoming Bearer token, answers every request, and returns
  the complete Case4Answers JSON in the same HTTP response.

Secrets, dataset URLs and dataset content are never written to logs.

The application is deliberately a single-worker process: the active runtime,
browser sessions and the shared LLM budget are process-local. Run it with one
Uvicorn worker until cross-process state has its own store. Implementation
lives in the ``backend`` package; this file is the stable entrypoint.
"""

from backend.app import app

__all__ = ["app"]
