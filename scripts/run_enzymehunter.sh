#!/bin/bash
# Run EnzymeHunter (https://github.com/cgxbio/EnzymeHunter, PyPI enzymehunter 0.1.15) on the six test
# sets in all-enzymes mode, as done on 2026-11-05. Scoring: scripts/bootstrap_ci.py --external.
#
# Everything lives outside the repo in $EH (default ~/workspace/enzymehunter).
set -euo pipefail
EH=${EH:-$HOME/workspace/enzymehunter}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$EH"/{download,inputs,results,tmp}
cd "$EH"

# 1. Environment. Python 3.9 (the authors') has no biotite wheel here, so 3.11. Pins: the package's
#    max-separation code needs pandas<3 (positional Series indexing) and tm-vec needs transformers<5
#    (T5Tokenizer.batch_encode_plus).
[ -x .venv/bin/python ] || uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python enzymehunter==0.1.15 "pandas>=2.0,<3" "transformers>=4.40,<5"

# 2. Assets. Hugging Face was unreachable from this machine, so the Zenodo archive (record 18598241,
#    EnzymeHunter_v2.zip, 6.3 GB). It lacks two things the HF assets include: the ESM2 3B/650M
#    checkpoints and torchvision's ResNet101 -- the standard public files, linked from the torch cache.
if [ ! -d assets/model ]; then
  curl -sS -L --retry 5 -C - -o download/EnzymeHunter_v2.zip \
       https://zenodo.org/api/records/18598241/files/EnzymeHunter_v2.zip/content
  unzip -q download/EnzymeHunter_v2.zip 'EnzymeHunter/model/*' 'EnzymeHunter/data/*' -d download/unpacked
  mv download/unpacked/EnzymeHunter assets && rmdir download/unpacked
fi
C=$HOME/.cache/torch/hub/checkpoints   # fetched by fair-esm / torchvision on first use
mkdir -p assets/model/esm2_model assets/model/diamond_tmvec_database/resnet_model
for f in esm2_t36_3B_UR50D esm2_t33_650M_UR50D; do
  ln -sf "$C/$f.pt" assets/model/esm2_model/$f.pt
  ln -sf "$C/$f-contact-regression.pt" assets/model/esm2_model/$f-contact-regression.pt
done
ln -sf "$C/resnet101-cd907fc2.pth" assets/model/diamond_tmvec_database/resnet_model/resnet101-cd907fc2.pth

# 3. Local fix: all-enzymes mode never writes {name}_pred_enz.{csv,fasta}, which the EC step reads
#    (same bug in the repo's EnzymeHunter.py). Write them as the classifier branch would.
P=.venv/lib/python3.11/site-packages/enzymehunter/pipeline.py
if ! grep -q "PATCH (local)" "$P"; then
  cp -n "$P" "$P.orig"
  python3 - "$P" <<'PY'
import sys; from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
a = "        enz_subset.to_csv(enz_nonenz_path, sep='\\t', index=False)\n"
assert s.count(a) == 1
p.write_text(s.replace(a, a + "        # PATCH (local): write {test_name}_pred_enz.{csv,fasta}, which predict_ec_numbers reads\n"
                              "        enz_subset = prepare_enz_subset(test_name, enz_subset)\n"))
PY
fi

# 4. One run on the union of the six test sets (predictions are per protein).
"$REPO/.venv/bin/python" - "$REPO" "$EH/inputs/tests_union.fasta" <<'PY'
import csv, sys; from pathlib import Path
repo, out = Path(sys.argv[1]), Path(sys.argv[2]); seqs = {}
for n in ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]:
    for r in csv.DictReader((repo / f"features/tests/{n}.csv").read_text().splitlines(), delimiter="\t"):
        seqs[r["Entry"]] = r["Sequence"].strip()
out.write_text("".join(f">{e}\n{s}\n" for e, s in seqs.items()))
print(len(seqs), "enzymes")
PY
# tmvec-search is called via PATH; /tmp is too small for the contact maps, so TMPDIR on /home.
PATH="$EH/.venv/bin:$PATH" TMPDIR="$EH/tmp" .venv/bin/enzymehunter predict inputs/tests_union.fasta \
    --model-dir assets --all-are-enzymes --device cuda:0 -o results/tests_union_allenz.csv
cp results/tests_union_allenz.csv "$REPO/features/external/enzymehunter_tests_union_allenz.csv"
