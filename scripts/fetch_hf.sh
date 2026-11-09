#!/bin/bash
# Retry a Hugging Face snapshot download until it succeeds (the proxy to huggingface.co is intermittent).
# usage: bash scripts/fetch_hf.sh <repo> [allow_pattern...]
repo=$1; shift
for attempt in $(seq 1 200); do
  if timeout 3600 "$(dirname "$0")/../envs/esmc/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
pats = sys.argv[2:] or None
print(snapshot_download(sys.argv[1], allow_patterns=pats))
" "$repo" "$@"; then echo "ok $repo"; exit 0; fi
  echo "attempt $attempt failed $(date)"; sleep 120
done
exit 1
