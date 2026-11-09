#!/bin/bash
# Recipe sweep on the ESM2-650M global embedding (id50 protocol).
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
F="--features esm2-650m:last --epochs 100 --eval-every 20"
bash scripts/feat.sh id50 esm2_bce_e100 $F
bash scripts/feat.sh id50 esm2_asl_e100 $F --loss asl
bash scripts/feat.sh id50 esm2_bce_e100_lr3e3 $F --lr 3e-3
bash scripts/feat.sh id50 esm2_focal_e100 $F --loss focal
bash scripts/feat.sh id50 esm2_asl_e100_h2048l2 $F --loss asl --hidden 2048 --layers 2
