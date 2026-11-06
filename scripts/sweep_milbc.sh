#!/bin/bash
# MIL-BC optimiser sweep on the fold-0 validation split (test sets scored for information only).
# usage: bash sweep_milbc.sh <name> [train.py args...]
set -u
cd "$(dirname "$0")/../code"
F=../features
name=$1; shift
EVAL=""; for n in new price uniprot-all uniprot-multi test70 test50; do EVAL="$EVAL --eval-set $F/tests/$n.npz $F/tests/$n.csv"; done
../.venv/bin/python train.py --model milbc --npzdata $(for i in $(seq -w 0 15); do echo $F/shards/split100_$i.npz; done) \
    --csvdata $F/split100_min5.csv --num-folds 5 --fold 0 $EVAL --eval-every 10 \
    --output-dir ../runs/sweep/$name "$@" > $F/logs/sweep_$name.log 2>&1
