#!/bin/bash
# Sequential job queue: runs lines of features/logs/queue.txt in order (append lines any time).
# Each line is a shell command run from the repo root; finished lines are logged to queue.done.
cd "$(dirname "$0")/.."
Q=features/logs/queue.txt; D=features/logs/queue.done
touch $Q $D
while true; do
  n=$(wc -l < $D)
  line=$(sed -n "$((n + 1))p" $Q)
  if [ -z "$line" ]; then sleep 30; continue; fi
  echo "$(date +%H:%M) start: $line" >> features/logs/queue.log
  bash -c "$line"
  echo "$line" >> $D
  echo "$(date +%H:%M) end (exit $?): $line" >> features/logs/queue.log
done
