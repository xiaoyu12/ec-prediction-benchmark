#!/bin/bash
# Domain-attention model on the id50 fold (validation only), each run memory-capped.
# usage: [CUDA_VISIBLE_DEVICES=...] bash domattn_id50.sh <name> [train_domattn.py args...]
set -u
cd "$(dirname "$0")/../code"
F=../features
name=$1; shift
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -q \
  ../.venv/bin/python train_domattn.py --npzdata $(for i in $(seq -w 0 15); do echo $F/shards/split100_$i.npz; done) \
    --csvdata $F/split100_min5.csv --domains $F/domains/domain_esm.npz \
    --val-entries ../scripts/folds/id50_val.txt --exclude-entries ../scripts/folds/id50_exclude.txt \
    --output-dir ../runs/domattn/id50_$name "$@" > $F/logs/domattn_id50_$name.log 2>&1
