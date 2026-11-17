"""Local CLI. Prints evidence text only when search is explicitly requested."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

from .embeddings import ProxyEmbeddings
from .pipeline import Config, build_index, search


def main():
    parser = argparse.ArgumentParser(description="Prepare case 4 corpus without downloading models")
    commands = parser.add_subparsers(dest="command", required=True)
    default = str(Path(tempfile.gettempdir()) / "sain-rag" / "index.sqlite3")
    build = commands.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--output", default=default)
    build.add_argument("--mode", choices=("sections", "structural", "semantic"), default="sections")
    build.add_argument("--max-words", type=int, default=250)
    build.add_argument("--overlap-words", type=int, default=40)
    build.add_argument("--batch-size", type=int, default=16)
    build.add_argument("--embeddings", action="store_true")
    build.add_argument("--timeout", type=float, default=120)
    find = commands.add_parser("search")
    find.add_argument("--index", default=default)
    find.add_argument("--query", required=True)
    find.add_argument("--top-k", type=int, default=5)
    find.add_argument("--embeddings", action="store_true")
    find.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        deadline = time.monotonic() + args.timeout
        provider = ProxyEmbeddings.from_env() if args.embeddings else None
        if args.command == "build":
            result = build_index(args.input, args.output,
                                 Config(mode=args.mode, max_words=args.max_words,
                                        overlap_words=args.overlap_words, batch_size=args.batch_size),
                                 provider=provider, deadline=deadline)
            result["index_bytes"] = Path(args.output).stat().st_size
        else:
            result = search(args.index, args.query, top_k=args.top_k,
                            candidate_k=max(20, args.top_k), provider=provider, deadline=deadline)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, TimeoutError, OSError, sqlite3.Error):
        # Error details can include local paths, closed dataset IDs, or URLs.
        print("RAG operation failed: check input/configuration, deadline, and provider availability. No secrets or response bodies logged.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
