#!/bin/bash
# Pooled PLM embeddings for every split100 and test protein; memory-capped, resumable.
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/extract_plm.sh <model> [extract_plm.py args...]
set -u
cd "$(dirname "$0")/.."
model=$1; shift
export HF_HUB_OFFLINE=1
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 -q \
  envs/esmc/bin/python code/extract_plm.py --model "$model" --csvdata features/domains/all_union.csv \
    --out-dir features/plm/"$model" "$@" >> features/logs/extract_"$model".log 2>&1
