#!/bin/bash
# Domain features for the domain-attention model: pool ESM2 per DCTdomain domain while dctfast.py
# segments, then a final pass once its output exists. Both steps are resumable.
set -u
cd "$(dirname "$0")/../code"
D=../features/domains
until [ -f $D/all-dct.npz ]; do
  ../.venv/bin/python domain_esm.py --csvdata $D/all_union.csv --dct $D/all-dct.npz.partial --output $D/domain_esm.npz
  sleep 120
done
../.venv/bin/python domain_esm.py --csvdata $D/all_union.csv --dct $D/all-dct.npz --output $D/domain_esm.npz
