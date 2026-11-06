#!/bin/bash
# Train every model on split100 with the paper's settings, scoring the six test sets as it goes.
set -u
# WSL2: without this the CUDA driver refuses new allocations once the caching allocator fragments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../code"
PY=../.venv/bin/python
F=../features
LOG=$F/logs
TRAIN="--npzdata $(ls $F/shards/split100_[0-9][0-9].npz | tr '\n' ' ') --csvdata $F/split100_min5.csv"
EVAL=""
for name in new price uniprot-all uniprot-multi test70 test50; do
    EVAL="$EVAL --eval-set $F/tests/$name.npz $F/tests/$name.csv"
done

run() {  # run <model> <done-file> <extra args...>
    local model=$1 target=$2
    shift 2
    for attempt in 1 2 3; do
        [ -s "$target" ] && return 0
        echo "[$(date '+%F %T')] $model attempt $attempt" | tee -a $LOG/train_$model.log
        $PY train.py --model $model $TRAIN $EVAL --output-dir ../runs/$model "$@" >> $LOG/train_$model.log 2>&1 \
            && [ -s "$target" ] && return 0
        sleep 10
    done
    echo "[$(date '+%F %T')] $model FAILED" | tee -a $LOG/train_$model.log
    return 1
}

run milbc          ../runs/milbc/model.pth
run milbc_plus     ../runs/milbc_plus/model.pth
run esm2_bc        ../runs/esm2_bc/model.pth
run milcl_backbone ../runs/milcl_backbone/backbone.pth || exit 1
run milcl_plus     ../runs/milcl_plus/model.pt --backbone ../runs/milcl_backbone/backbone.pth \
                   --augment-npz $F/split100_orphans.npz
echo "[$(date '+%F %T')] training done"
