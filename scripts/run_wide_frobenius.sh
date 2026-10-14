#!/usr/bin/env bash
# Optional wide-only run. Use a fresh directory, never archived paper results.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RUNS="${RUNS:-reproduction/new-runs}"
THREADS="${THREADS:-2}"
for divisor in 1 2; do
  "$PYTHON" scripts/reproduce.py --experiment f21_wide --output "$RUNS" --step-divisor "$divisor" --threads "$THREADS"
done
for divisor in 1 2; do
  "$PYTHON" scripts/stage2_from_checkpoint.py --runs "$RUNS" --step-divisor "$divisor" --threads "$THREADS"
done
