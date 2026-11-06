#!/bin/bash
# Domain-attention model on all of split100, scored on the six test sets; memory-capped.
# usage: [CUDA_VISIBLE_DEVICES=...] bash domattn_full.sh <name> [train_domattn.py args...]
set -u
cd "$(dirname "$0")/../code"
F=../features
name=$1; shift
EVAL=""; for n in new price uniprot-all uniprot-multi test70 test50; do EVAL="$EVAL --eval-set $F/tests/$n.npz $F/tests/$n.csv"; done
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -q \
  ../.venv/bin/python train_domattn.py --npzdata $(for i in $(seq -w 0 15); do echo $F/shards/split100_$i.npz; done) \
    --csvdata $F/split100_min5.csv --domains $F/domains/domain_esm.npz $EVAL \
    --output-dir ../runs/domattn/full_$name "$@" > $F/logs/domattn_full_$name.log 2>&1
