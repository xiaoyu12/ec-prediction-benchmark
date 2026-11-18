# EC prediction benchmark: protein language models, structure and homology

A head-to-head comparison of ways to predict Enzyme Commission (EC) numbers, all trained on the same data,
selected on the same validation fold and scored once on the same six test sets:

- **Homology transfer:** by sequence (MMseqs2) and by structure (Foldseek, with ProstT5-predicted or AlphaFold DB 3Di).
- **Classifiers on embeddings of seven PLMs:** ESM2-650M, ESM2-3B, ESMC-600M, ESMC-6B, ProtT5-XL, and two
  structure-aware models (ProstT5, SaProt-650M).
- **Other single models:** classifiers on sparse-autoencoder (SAE) features of ESMC, and a fine-tuned ESMC-600M.
- **Combinations:** concatenations and ensembles of PLM heads, and identity-gated blends of PLM heads with homology.
- **Seven published predictors, re-run under the same protocol:** CLEAN, EnzHier, CLEAN-Contact, EnzymeHunter,
  MIL-BC+, MIL-CL+ and ESM2-BC.

The manuscript is [`reports/manuscript/manuscript.pdf`](reports/manuscript/manuscript.pdf):
*Protein language models complement, but do not replace, homology search for EC number prediction.* Every model
that was tried, including the ones that did not help, is logged in
[`reports/AUTORESEARCH.md`](reports/AUTORESEARCH.md).

## Main results

Per-enzyme F1. "id50" is the validation fold used for every choice. "Test mean" is the mean over the six test
sets. "vs homology" counts the test sets where a method is significantly better / worse than homology alone
(MMseqs2 + Foldseek-AFDB best hit), by paired bootstrap with 2,000 resamples.

| Method | id50 | Test mean | vs homology |
|---|---:|---:|:---:|
| **Homology alone** (MMseqs2 + Foldseek-AFDB, best hit) | 0.897 | 0.604 | – |
| ESM2-3B head | 0.888 | 0.617 | 0 / 0 |
| ESMC-600M SAE features, max-pooled | 0.892 | 0.605 | 0 / 1 |
| ProstT5 head (structure-aware) | 0.890 | 0.579 | 0 / 4 |
| SaProt-650M head, pLDDT-masked (structure-aware) | 0.863 | 0.569 | 0 / 4 |
| ESMC-600M, fine-tuned | 0.908 | 0.586 | 0 / 4 |
| Six-head PLM ensemble | 0.901 | 0.622 | 1 / 0 |
| ESM2-3B ⊕ ESMC-600M ⊕ SAE head + homology | 0.916 | 0.632 | 4 / 0 |
| **Six-head ensemble + homology (best hybrid)** | 0.914 | **0.638** | **5 / 0** |
| *Published, re-run:* ESM2-BC (retrained) | 0.877 | 0.574 | 0 / 4 |
| MIL-BC+ (released) | 0.868 | 0.572 | 0 / 4 |
| CLEAN-Contact | – | 0.568 | 0 / 3 |
| MIL-CL+ (released) | – | 0.551 | 0 / 5 |
| EnzymeHunter | – | 0.547 | 0 / 4 |
| CLEAN | – | 0.536 | 0 / 5 |
| EnzHier | – | 0.503 | 0 / 5 |

What this shows:

- **No PLM embedding beats homology on its own.** The best single embedding (ESM2-3B) matches homology
  without being significantly better. SAE features tie it. Structure-aware PLMs and a fine-tuned ESMC fall
  below it.
- **Structure helps through search, not embeddings.** Foldseek on AlphaFold DB 3Di is as strong as MMseqs2.
  ProstT5 and SaProt embeddings are weaker than plain ESM2.
- **The two are complementary.** PLM classifiers are better on common ECs. Homology is better on rare ECs and
  close homologues. Blending them by the top hit's identity gives the best model.
- **All seven published methods fall below homology alone.** None is significantly better on any set.
- **The metric matters.** The support-weighted F1 used in earlier benchmarks rewards calling many ECs per
  enzyme. This benchmark reports per-enzyme F1, calls per enzyme and paired bootstrap intervals instead.

## Protocol

| | |
|---|---|
| Training | split100: 227,360 enzymes, 5,242 EC numbers (`checkpoints/classes.npy`) |
| Model selection | `id50`: 10,422 split100 enzymes held out by 50% identity clustering, with leak-excluded neighbours removed (`scripts/folds/`) |
| Final scores | `full`: retrain the chosen configuration on all of split100, then score each test set once |
| Test sets | New-392, Price-149, UniProt-all (987), UniProt-multi (77), Test-70 (1,057), Test-50 (732); 3,394 enzymes |
| Decision rule | sigmoid ≥ 0.5, at least one and at most five calls (max-separation for centroid methods, as published) |
| Primary metric | per-enzyme F1; micro and support-weighted F1 are stored too |

The training and test sets are those of MIL-BC, on [Zenodo](https://zenodo.org/records/22598911), and are
included in `dataset/`.

## Repository layout

```text
code/                 training, homology transfer, ensembles (run from code/ or by path)
  extract_plm.py        pooled PLM embeddings and ESMC SAE features
  train_feat.py         classifier heads on fixed embeddings, both protocols
  homology.py           MMseqs2 / Foldseek EC transfer
  ensemble.py           probability averaging and the identity-gated homology blend
  knn_feat.py           nearest-neighbour transfer in embedding space
  stack.py              gradient-boosted re-ranker over models and homology
  finetune_plm.py       partial fine-tuning of ESMC-600M
  process.py, infer.py, train.py, models.py, ...   MIL-BC family (see docs/MIL_MODELS.md)
scripts/
  feat.sh, extract_plm.sh   memory-capped wrappers for train_feat.py and extract_plm.py
  bootstrap_ci.py           paired bootstrap of scores and differences, incl. external prediction files
  stratify.py, rare_eval.py, identity_bins.py, bins_from_predictions.py   error analysis by identity and rarity
  feat_table.py             results table from runs/feat/*/metrics.json
  manuscript_data.py        numbers behind the manuscript's tables and figures
  folds/                    the id50 validation fold
  baselines/                drivers for the CLEAN, EnzHier and CLEAN-Contact re-runs
dataset/              split100 and the six test sets
checkpoints/          released and retrained MIL-BC family checkpoints, EC vocabulary
reports/
  AUTORESEARCH.md       log of every experiment
  manuscript/           LaTeX source, figures, figure data and make_figures.py
docs/MIL_MODELS.md    the inherited MIL-BC / MIL-CL+ / ESM2-BC code (upstream README)
```

Large generated data stays out of git: `features/` (embeddings, homology databases, logs), `runs/` (one
directory per model, with `metrics.json`, `logits.npz` and `predictions.csv`), `envs/`, `models/` and the
AlphaFold DB tarball.

## Setup

- **Main environment:** `uv sync` creates `.venv` from `pyproject.toml` (torch, numpy, scipy, fair-esm, faiss).
  It runs heads, homology transfer, ensembles and scoring.
- **Embedding environment:** `envs/esmc`, with Biohub's `esm` package and `transformers`. It is separate
  because fair-esm also installs a package called `esm`. `extract_plm.py`, `stack.py` and the figure script
  run here.
- **Search tools:** MMseqs2 and Foldseek. `code/homology.py` looks for them under `~/.local/opt/`; edit
  `MMSEQS` and `FOLDSEEK` at its top to point elsewhere.
- **Structures:** the AlphaFold DB Swiss-Prot v6 PDB tarball (`swissprot_pdb_v6.tar`) and the Foldseek
  `Alphafold/Swiss-Prot` database, for AFDB 3Di, pLDDT masking and CLEAN-Contact's contact maps.

The wrappers in `scripts/` run jobs under `systemd-run --user --scope -p MemoryMax=...` so that a large
job cannot exhaust host memory. Pick the GPU with `CUDA_VISIBLE_DEVICES`.

## Reproducing

All proteins (split100 and the six test sets) are listed once in `features/domains/all_union.csv`
(`Entry`, `Sequence`; 228,888 proteins). Every feature file follows its row order.

**1. Embeddings.** One run per PLM; resumable, and `--shard i/n` splits a model over GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_plm.sh esmc-600m   # -> features/plm/esmc-600m/{last,mid,sae_mean,sae_max}.npy
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_plm.sh esm2-3b
```

**2. Homology transfer**, for each protocol:

```bash
cd code
python homology.py --protocol full --output-dir ../runs/feat/full_mmseqs
python homology.py --protocol full --tool foldseek --states ../features/plm/afdb/3di.fasta \
                   --output-dir ../runs/feat/full_foldseek_afdb      # 3Di from scripts/afdb_3di.py
```

**3. Classifier heads.** Choose on `id50`, then retrain on `full`:

```bash
bash scripts/feat.sh id50 esm2_3b_last --features esm2-3b:last
bash scripts/feat.sh full esm2_3b_last --features esm2-3b:last
```

`train_feat.py --help` lists the head options: architecture, loss, SAE `--log1p` blocks, seeds and more.
The arguments of each run are in `runs/feat/<run>/config.json` and in `reports/AUTORESEARCH.md`.

**4. Ensembles and the hybrid.** With `--protocol id50`, the identity gate (`lo`, `hi`, `amax`) is chosen on
the fold. With `full`, pass the chosen values.

```bash
python ensemble.py --protocol id50 --name hyb_ens6_6b_msaf --homology mmseqs foldseek_afdb \
    --runs esm2_bce_e100 esmc600_last esmc600_saemax esmc600_mid_esm2 esmc600_last_saemax_esm2 esmc6b_saemax
```

**5. Scores, significance and error analysis.**

```bash
python scripts/feat_table.py --protocol full                       # results table
python scripts/bootstrap_ci.py --external CLEAN=features/external/clean_tests.csv ...   # paired bootstrap
python scripts/stratify.py --hybrid hyb_ens6_6b_msaf --model ens6_6b \
    --single esm2_3b_last_esmc600_saemax --homology mmseqs foldseek_afdb                # by top-hit identity
python scripts/rare_eval.py --runs esm2_3b_last --protocol full                            # by EC rarity
```

**6. Published methods.** MIL-BC+, MIL-CL+ and ESM2-BC use `code/infer.py` and `code/train.py`
([docs/MIL_MODELS.md](docs/MIL_MODELS.md)). EnzymeHunter runs through `scripts/run_enzymehunter.sh`. CLEAN,
EnzHier and CLEAN-Contact run from their released split100 models
([scripts/baselines/README.md](scripts/baselines/README.md)):
[CLEAN](https://github.com/tttianhao/CLEAN) (weights on
[Google Drive](https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view?usp=sharing)),
[EnzHier](https://github.com/TangTZscut/EnzHier) (model
[in its repository](https://github.com/TangTZscut/EnzHier/blob/main/data/model/split100_triplet_withEC_7000.pth))
and [CLEAN-Contact](https://github.com/PNNL-CompBio/CLEAN-Contact).

**7. Manuscript.**

```bash
python scripts/manuscript_data.py                                   # -> reports/manuscript/data/*.json
envs/esmc/bin/python reports/manuscript/make_figures.py             # -> reports/manuscript/fig*.pdf
cd reports/manuscript && pdflatex manuscript && pdflatex manuscript && pdflatex manuscript
```

## Credits

This repository started from [omics-ai/enzyme-mil](https://github.com/omics-ai/enzyme-mil), the code of
MIL-BC ("Domain-aware enzyme classification via multiple instance learning", *Bioinformatics*,
[btag804](https://academic.oup.com/bioinformatics/article/42/11/btag804/9006040)), which provides the
datasets and the MIL-BC family of models. Domain segmentation uses
[DCTdomain](https://github.com/mgtools/DCTdomain), bundled in `DCTdomain/`. The manuscript cites every
PLM, SAE, search tool and published method used here.
