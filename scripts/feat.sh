#!/bin/bash
# Train a train_feat.py model; memory-capped. Output runs/feat/<protocol>_<name>, log features/logs/feat_<protocol>_<name>.log
# usage: CUDA_VISIBLE_DEVICES=0 bash scripts/feat.sh <protocol id50|full> <name> [train_feat.py args...]
set -u
cd "$(dirname "$0")/../code"
protocol=$1; name=$2; shift 2
systemd-run --user --scope -p MemoryMax=${MEM:-40G} -p MemorySwapMax=0 -q \
  ../.venv/bin/python train_feat.py --protocol "$protocol" --output-dir ../runs/feat/${protocol}_$name "$@" \
  > ../features/logs/feat_${protocol}_$name.log 2>&1
