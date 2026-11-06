#!/bin/bash
# Train on split100 minus the 50 %-identity validation fold (scripts/folds/id50_*) (test sets scored for information only).
# usage: bash id50_train.sh <name> [train.py args...]
set -u
cd "$(dirname "$0")/../code"
F=../features
name=$1; shift
EVAL=""; for n in new price uniprot-all uniprot-multi test70 test50; do EVAL="$EVAL --eval-set $F/tests/$n.npz $F/tests/$n.csv"; done
../.venv/bin/python train.py --model milbc --npzdata $(for i in $(seq -w 0 15); do echo $F/shards/split100_$i.npz; done) \
    --csvdata $F/split100_min5.csv --val-entries ../scripts/folds/id50_val.txt --exclude-entries ../scripts/folds/id50_exclude.txt $EVAL --eval-every 10 \
    --output-dir ../runs/id50/$name "$@" > $F/logs/id50_$name.log 2>&1
