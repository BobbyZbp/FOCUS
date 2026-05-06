#!/usr/bin/env bash
# Print the path of the highest-numbered checkpoint under a given root.
# Usage:
#   bash scripts/find_latest_checkpoint.sh /workspace/checkpoints/calql_large_diverse_seed0
#
# Prints something like:
#   /workspace/checkpoints/calql_large_diverse_seed0/wsrl/_antmaze-large-diverse-v2_calql_seed0_*/checkpoint_1000000

set -e

ROOT=${1:-/workspace/checkpoints/calql_large_diverse_seed0}

if [ ! -d "$ROOT" ]; then
  echo "ERROR: directory not found: $ROOT" >&2
  exit 1
fi

CKPT=$(find "$ROOT" -type d -name "checkpoint_*" 2>/dev/null \
       | awk -F'checkpoint_' '{print $NF, $0}' \
       | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint_* directory found under $ROOT" >&2
  exit 1
fi

echo "$CKPT"
