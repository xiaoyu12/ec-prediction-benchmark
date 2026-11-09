#!/bin/bash
# After ProstT5 extraction: queue ProstT5 feature runs, then Foldseek (ProstT5 3Di) homology for both protocols.
cd "$(dirname "$0")/.."
bash scripts/when_ready.sh prostt5 features/logs/next_prostt5.txt
cd code
for protocol in id50 full; do
  systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -q ../.venv/bin/python homology.py --tool foldseek \
    --protocol $protocol --output-dir ../runs/feat/${protocol}_foldseek > ../features/logs/foldseek_$protocol.log 2>&1
done
echo done
