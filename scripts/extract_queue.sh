#!/bin/bash
# Run the remaining extractions one after another on the bus-01 GPU, after esmc-600m (pid 2507266).
cd "$(dirname "$0")/.."
while kill -0 2507266 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
bash scripts/extract_plm.sh prott5 --batch-tokens 24000
bash scripts/extract_plm.sh esm2-3b --batch-tokens 16000
bash scripts/extract_plm.sh esmc-6b --batch-tokens 16000
until grep -q "^ok" features/logs/fetch_prostt5.log; do sleep 300; done
bash scripts/extract_plm.sh prostt5 --batch-tokens 24000
