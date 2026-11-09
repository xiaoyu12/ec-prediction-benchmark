#!/bin/bash
# Usage: when_ready.sh <model-dir-name> <queue lines file>: wait for features/plm/<model> assembly, then queue the lines.
until grep -q "^wrote .*features/plm/$1:" features/logs/extract_$1.log 2>/dev/null; do sleep 60; done
cat "$2" >> features/logs/queue.txt
echo "queued $2 at $(date)"
