#!/usr/bin/env bash
# Run inside the container (the entrypoint already dispatches, but this is
# handy if you exec into a shell).
set -euo pipefail
cd /app
python -m src.pipeline "$@"
