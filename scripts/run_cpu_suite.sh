#!/usr/bin/env bash
# Portable sequential CPU suite; safe default is to print the exact commands.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" -m paper_reproduction train "$@"
