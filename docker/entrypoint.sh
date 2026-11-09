#!/usr/bin/env bash
# Container entrypoint: dispatch to the pipeline, or drop into a shell.
set -euo pipefail

cd /app

if [[ "${1:-}" == "bash" || "${1:-}" == "shell" ]]; then
    exec /bin/bash
fi

# Anything else is forwarded to the orchestrator.
exec python -m src.pipeline "$@"
