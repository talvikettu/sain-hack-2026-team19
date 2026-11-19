#!/bin/sh
# Stand entrypoint: listen on 0.0.0.0 and the organizer-provided PORT.
# uvicorn keeps a long-running /run request alive by default; the access log is
# disabled so request data is not written to the stand logs.
set -eu
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --no-access-log
