# MIL-BC, MIL-BC+, MIL-CL+ and ESM2-BC (inherited models)

> This page is the README of the upstream repository [omics-ai/enzyme-mil](https://github.com/omics-ai/enzyme-mil), kept for the
> four multiple-instance-learning models this benchmark re-runs. Commands run from `code/`, as upstream.

Predict EC numbers for enzyme sequences with multiple-instance learning over [DCTdomain](https://github.com/mgtools/DCTdomain) domain embeddings and ESM2.

Each enzyme is a _bag_ and its structural domains are _instances_. A gated-attention pooler learns one attention distribution per enzyme and produces independent scores over a fixed vocabulary of
**5,242 EC numbers** learned from split100. An enzyme whose true EC is outside that vocabulary can never be called correctly.

```text
example.csv  (Entry, EC number, Sequence)
     |
     |  process.py   DCTdomain -> domain vectors + mask
     |               ESM2      -> whole-sequence embedding (1280-d)
     v
example.npz
     |
     |  infer.py     one of four models
     v
outputs/example_<model>.csv   (Entry, predicted_ec, n_predicted [, predicted_score])
```

`train.py` takes the same archive, built from `split100.csv`, and retrains the models, including the ESM2-BC baseline.
See [Train](#train).

## Dataset

All the datasets are available at [Zenodo](https://zenodo.org/records/22598911).

- training set: `dataset/split100.csv`
- four test sets
  - test set 1: `dataset/new.csv`
  - test set 2: `dataset/price.csv`
  - test set 3: `dataset/uniprot-all.csv`
  - test set 4: `dataset/uniprot-multi.csv`
- sequence-simlirity constrained test sets
  - test set 5: `dataset/test70.csv`
  - test set 6: `dataset/test50.csv`

| dataset       | enzymes | EC per enzyme | EC classes present (of 5,242) |
| :------------ | ------: | ------------: | ----------------------------: |
| NEW           |     392 |          1.28 |                           177 |
| Price         |     149 |          1.02 |                            56 |
| UniProt-all   |     987 |          1.09 |                           488 |
| UniProt-multi |      77 |          2.16 |                           110 |
| Test70        |   1,057 |          1.11 |                           492 |
| Test50        |     732 |          1.06 |                           339 |

## Models

| `--model`    | Checkpoint                            | Inputs                   | Decision rule            |
| ------------ | ------------------------------------- | ------------------------ | ------------------------ |
| `milbc`      | `checkpoints/milbc/model_v1.pth`      | domain embeddings only   | sigmoid ≥ 0.5            |
| `milbc_plus` | `checkpoints/milbc_plus/model_v1.pth` | domain embeddings + ESM2 | sigmoid ≥ 0.5            |
| `milcl_plus` | `checkpoints/milcl_plus/model_v2.pt`  | domain embeddings + ESM2 | CLEAN maximum separation |
| `esm2_bc`    | `checkpoints/retrained/esm2_bc/model.pth` | ESM2 only            | sigmoid ≥ 0.5            |

`milcl_plus` additionally needs `checkpoints/milcl_plus/centroids.npy`, the split100 EC centroids its projection is compared against;
`infer.py` picks that up automatically from beside the checkpoint.

`esm2_bc` is the paper's sequence-level baseline. It applies MIL-BC's classifier head to the
whole-sequence ESM2 embedding alone, so comparing it with `milbc_plus` isolates what the bag of
domains adds. The paper released no checkpoint for it; the one here was retrained (below).

`checkpoints/retrained/<model>/` holds all four models retrained from scratch on split100 with
`train.py` and the paper's settings (one run each, seed 0; `config.json` and `history.json` beside
each checkpoint record the run). Use them exactly like the released ones, e.g.
`--model milcl_plus --checkpoint ../checkpoints/retrained/milcl_plus/model.pt`. F1 on the six test
sets (released checkpoints in parentheses):

| Model        | New-392       | Price-149     | Uniprot25-all | Uniprot25-multi | Test-70       | Test-50       |
| ------------ | ------------- | ------------- | ------------- | --------------- | ------------- | ------------- |
| `milbc`      | 0.436 (0.518) | 0.303 (0.509) | 0.644 (0.696) | 0.481 (0.611)   | 0.484 (0.553) | 0.424 (0.499) |
| `milbc_plus` | 0.524 (0.554) | 0.454 (0.530) | 0.727 (0.747) | 0.583 (0.689)   | 0.577 (0.611) | 0.509 (0.555) |
| `milcl_plus` | 0.528 (0.541) | 0.442 (0.486) | 0.704 (0.698) | 0.544 (0.511)   | 0.562 (0.570) | 0.490 (0.511) |
| `esm2_bc`    | 0.500 (0.504) | 0.394 (0.468) | 0.724 (0.712) | 0.595 (0.589)   | 0.572 (0.564) | 0.502 (0.481) |

(`esm2_bc`'s parenthesised numbers are the paper's Tables 3–4.) The retrained `milbc` falls short of
the released one by a consistent margin: three seeds agree within 0.02, and neither 200 epochs nor
initialising the output bias to the class prior closes the gap, so the released model's recipe must
differ in some detail the paper does not state.

## Install

```bash
pip install -r requirements.txt
```

DCTdomain is bundled in `DCTdomain/`. Compile its `RecCut` domain cutter once and `process.py` refuses to start without it:

```bash
cd DCTdomain
g++ -o src/RecCut src/RecCut.cpp
```

That is the whole setup. `process.py` runs DCTdomain in this same interpreter, so `requirements.txt`
covers both: `torch`, `numpy` and `fair-esm` for the models, plus `scipy` and `faiss` for DCTdomain.
(`faiss` only builds a similarity index nothing here reads, but DCTdomain imports it
unconditionally. Conda users can `conda install -c conda-forge faiss` instead of `faiss-cpu`.)

The first run downloads the ESM2 `esm2_t33_650M_UR50D` weights (2.6 GB) into the torch hub cache.

## Use

**1. Build features.** Input is a tab-separated table with `Entry` and `Sequence`
columns. An `EC number` column may be present and is ignored, so this works on unannotated proteins.

```bash
cd code
python process.py --csvdata ../example/example.csv
```

Writes `../example/example.npz`. Useful flags: `--output` and `--device`.

**2. Predict.**

```bash
python infer.py --npzdata ../example/example.npz \
                --model milbc_plus \
                --checkpoint ../checkpoints/milbc_plus/model_v1.pth
```

Writes `../outputs/example_milbc_plus.csv` and prints a summary:

```json
{
  "model": "milbc_plus",
  "rule": "sigmoid 0.5",
  "checkpoint": "../checkpoints/milbc_plus/model_v1.pth",
  "enzymes": 10,
  "classes": 5242,
  "output": "../outputs/example_milbc_plus.csv"
}
```

```csv
Entry,predicted_ec,n_predicted
A0A0H2Z7X0,2.7.7.65,1
A0A1B3PEJ0,2.7.8.7,1
```

**3. Confidence scores (optional).** `--confscore` adds a `predicted_score` column with one score
per called EC in the same order as `predicted_ec`:

```bash
python infer.py --npzdata ../example/example.npz \
                --model milbc_plus \
                --checkpoint ../checkpoints/milbc_plus/model_v1.pth \
                --confscore
```

## Train

`code/train.py` retrains every released model from `split100.csv` with the paper's settings, and
writes checkpoints that `infer.py` reads unchanged. Labels come from the `EC number` column
(`;`-separated for multifunctional enzymes), joined to the feature archive on `Entry`. The two
split100 sequences shorter than five residues are dropped, which leaves the paper's 227,360 enzymes.

**1. Build training features.** This is the same `process.py` step as for inference:

```bash
cd code
python process.py --csvdata ../dataset/split100.csv          # -> ../dataset/split100.npz
```

A `process.py` archive keeps every token in float64, so the full split100 archive takes about 28 GB
of memory to load. To lower the peak, split the CSV into shards, run `process.py` on each, and pass
all of them: `--npzdata a.npz b.npz ...`. The first time `train.py` loads an archive it writes a
compact `<name>.compact.npz` cache beside it (about 5 GB for split100) and reads that afterwards.

**2. MIL-BC, MIL-BC+ and the ESM2-BC baseline.** MIL-BC and MIL-BC+ use gated-attention MIL with
a binary cross-entropy loss: 80 epochs, AdamW, learning rate 2e-4, weight decay 0.01, batch 512.
ESM2-BC uses the same classifier head and the same recipe, but its only input is the ESM2 embedding.

```bash
python train.py --model milbc      --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv
python train.py --model milbc_plus --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv
python train.py --model esm2_bc    --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv
# -> ../runs/milbc/model.pth, ../runs/milbc_plus/model.pth, ../runs/esm2_bc/model.pth
```

Expect the loss to plateau for the first few thousand steps. With 5,242 independent sigmoids, the
model first pushes every negative logit down before it separates the classes.

**3. MIL-CL+.** This model trains in two stages, as the released checkpoint was built. The first
stage trains the fused domain + ESM2 backbone (`MILBCPlusBackbone` in `models.py`) with BCE, using the
same settings as step 2. The second stage refines it end to end with CLEAN's SupCon-Hard
contrastive loss: 1,500 epochs, Adam, learning rate 5e-4, temperature 0.1. Each epoch draws one
anchor per EC, 9 positives from that EC and 30 negatives from its 100 nearest EC centroids, and the
negatives are re-mined every 100 epochs. The classifier BCE term stays on as a joint loss
(`--bce-weight`, 0 to disable). At the end, `centroids.npy` is written beside `model.pt` for
max-separation inference.

```bash
python train.py --model milcl_backbone --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv

# optional, CLEAN's orphan-EC augmentation: 10 masked copies of every enzyme with a one-member EC
python orphans.py --csvdata ../dataset/split100.csv          # -> ../dataset/split100_orphans.csv
python process.py --csvdata ../dataset/split100_orphans.csv  # -> ../dataset/split100_orphans.npz

python train.py --model milcl_plus --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv \
                --backbone ../runs/milcl_backbone/backbone.pth \
                --augment-npz ../dataset/split100_orphans.npz
# -> ../runs/milcl_plus/model.pt + centroids.npy
```

Without `--augment-npz`, an orphan EC's positive is the anchor itself, and dropout supplies the
second view.

**Validation and test sets.** `--num-folds 5 --fold k` holds out fold *k* of the training data for
5-fold cross-validation. `--eval-set NPZ CSV`, which you can repeat, scores an independent test
set during training. Both use the decision rule `infer.py` uses and CLEAN's support-weighted
precision, recall and F1. The scores go to the log and to `history.json`.

```bash
python process.py --csvdata ../dataset/new.csv
python train.py --model milbc_plus --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv \
                --eval-set ../dataset/new.npz ../dataset/new.csv --eval-every 5
```

Every hyperparameter can be overridden: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`,
`--n-pos`, `--n-neg`, `--temperature` and the rest; see `python train.py --help`. `--amp` turns on
bfloat16 autocast. `--data-device cpu` keeps the feature tensors in host memory when the GPU is
small. Each run also writes `config.json` and `history.json` to `--output-dir`, which defaults to
`../runs/<model>`.

```bash
python infer.py --npzdata ../example/example.npz --model milbc_plus --checkpoint ../runs/milbc_plus/model.pth
python infer.py --npzdata ../example/example.npz --model milcl_plus --checkpoint ../runs/milcl_plus/model.pt
python infer.py --npzdata ../example/example.npz --model esm2_bc    --checkpoint ../runs/esm2_bc/model.pth
```

## Files

```text
enzyme-mil/
├── requirements.txt
├── code/
│   ├── models.py                architectures for the three checkpoints + ESM2-BC
│   ├── process.py               sequences -> feature archive
│   ├── infer.py                 feature archive -> EC calls
│   ├── train.py                 feature archive + EC labels -> checkpoints
│   └── orphans.py               mutated copies of orphan-EC enzymes for MIL-CL+
├── runs/                        where train.py writes
├── checkpoints/
│   ├── classes.npy              the 5,242 EC numbers, in model output order
│   ├── milbc/model_v1.pth
│   ├── milbc_plus/model_v1.pth
│   ├── milcl_plus/model_v2.pt + centroids.npy
│   └── retrained/               the four models retrained on split100 with train.py
├── DCTdomain/                   cloned from https://github.com/mgtools/DCTdomain;
│                                compile src/RecCut before first use
├── example/
│   ├── example.csv              10 enzymes with sequences and true EC numbers
│   └── example.npz              features file, ready for infer.py
└── outputs/                     where infer.py writes
```

## Citation

- **DCTdomain** — Iovino, Tang and Ye, _Genome Research_, 2024. <https://github.com/mgtools/DCTdomain>
- **ESM2** — Lin et al., _Science_, 2023. <https://github.com/facebookresearch/ESM>
- **CLEAN** max-separation — Yu et al., _Science_, 2023. <https://github.com/tttianhao/CLEAN>

---

## pretrained models

[CLEAN](https://github.com/tttianhao/CLEAN) shared pretrained model on [Google Drive](https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view?usp=sharing)

[EnzHier](https://github.com/labxscut/EnzHier) shared pretrained model on [GitHub](https://github.com/TangTZscut/EnzHier/blob/main/data/model/split100_triplet_withEC_7000.pth)
