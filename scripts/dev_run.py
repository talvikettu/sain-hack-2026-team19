"""Local /run smoke test: POST /run, save the answer, call /case/4/validate.

The token is read from --token-file or HACK_TEAM_TOKEN and is never printed.
The dataset code is not printed either.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_preprocessing.embeddings import NoRedirect


def _token(path: str | None) -> str:
    if path:
        value = Path(path).read_text(encoding="utf-8").strip()
    else:
        value = os.environ.get("HACK_TEAM_TOKEN", "").strip()
    if not value:
        raise SystemExit("Set HACK_TEAM_TOKEN or pass --token-file")
    return value


def _post(url: str, token: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        return response.status, response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset-code", default="case4-dev-v2")
    parser.add_argument("--token-file")
    parser.add_argument("--answer", default="answer.json")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--timeout", type=float, default=45 * 60 + 30)
    args = parser.parse_args()

    token = _token(args.token_file)
    payload = json.dumps({"case": 4, "dataset_code": args.dataset_code}).encode("utf-8")
    started = time.monotonic()
    try:
        status, raw = _post(args.base_url.rstrip("/") + "/run", token, payload, args.timeout)
    except HTTPError as exc:
        print(f"/run: HTTP {exc.code}")
        return 1
    except URLError as exc:
        print(f"/run: unavailable ({type(exc.reason).__name__})")
        return 1

    elapsed = time.monotonic() - started
    answer = json.loads(raw)
    requests = answer.get("requests")
    if not isinstance(requests, list):
        print("/run: response has no requests array")
        return 1
    Path(args.answer).write_bytes(raw)
    print(f"/run: HTTP {status}, {len(requests)} answers, {elapsed:.1f}s, saved {args.answer}")

    if args.no_validate:
        return 0
    api = os.environ.get("HACK_API_URL", "").rstrip("/")
    if not api:
        print("validate: HACK_API_URL is not set, skipped")
        return 0
    try:
        status, raw = _post(api + "/case/4/validate", token, raw, 120)
    except HTTPError as exc:
        print(f"validate: HTTP {exc.code}")
        return 1
    except URLError as exc:
        print(f"validate: unavailable ({type(exc.reason).__name__})")
        return 1
    result = json.loads(raw)
    print(f"validate: HTTP {status}, valid={result.get('valid')}")
    if not result.get("valid"):
        for error in result.get("errors", [])[:20]:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
