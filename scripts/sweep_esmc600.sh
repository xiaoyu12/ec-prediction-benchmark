#!/bin/bash
# Feature comparison on ESMC-600M (id50, default recipe), after extraction finishes.
cd "$(dirname "$0")/.."
until [ -f features/plm/esmc-600m/entries.txt ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
R="--epochs 100 --eval-every 25"
bash scripts/feat.sh id50 esmc600_last --features esmc-600m:last $R
bash scripts/feat.sh id50 esmc600_mid --features esmc-600m:mid $R
bash scripts/feat.sh id50 esmc600_saemean --features esmc-600m:sae_mean $R
bash scripts/feat.sh id50 esmc600_saemax --features esmc-600m:sae_max $R
bash scripts/feat.sh id50 esmc600_mid_saemean --features esmc-600m:mid esmc-600m:sae_mean $R
bash scripts/feat.sh id50 esmc600_mid_esm2 --features esmc-600m:mid esm2-650m:last $R
