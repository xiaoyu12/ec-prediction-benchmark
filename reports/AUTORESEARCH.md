# Autoresearch: new EC-prediction models

Branch `main` (started 2026-11-06 on `autoresearch`, merged 2026-11-18). Goal: beat the existing models (ESM2-BC, MIL-BC+, the
domain-attention model, EnzymeHunter) on EC prediction with new features, architectures and
hyperparameters, **on the same training and test data**.

## Protocol (fixed for every model)

- **Training data:** `features/split100_min5.csv`, the 5,242-EC split100 vocabulary
  (`checkpoints/classes.npy`).
- **Model selection:** the `id50` protocol. Train on split100 minus the identity-clustered validation
  fold and its leak-excluded neighbours (`scripts/folds`), then score the 10,422-enzyme fold. All
  choices (features, head, loss, epochs, k, thresholds) are made on id50, never on the test sets.
- **Final numbers:** the `full` protocol. Retrain the chosen configuration on all of split100, then
  score the six test sets: New-392, Price-149, UniProt-all, UniProt-multi, Test-70 and Test-50.
- **Decision rule:** the paper's `call_ec` for every model: sigmoid ≥ 0.5, at least one call, at
  most five.
- **Primary metric:** per-enzyme F1 (`sample`), which charges for false positives. Micro F1 and the
  paper's support-weighted F1 are also stored in each run's `metrics.json`. The weighted F1 rewards
  over-calling (see `HANDOFF_MILBC.md`).
- **Uncertainty:** paired bootstrap over enzymes (`scripts/bootstrap_ci.py --external`) for the final
  comparisons. Test sets are small (77–1,057 enzymes); Price in particular swings ±0.05 between
  epochs of one run.

## Summary (through entry 27)

- **Best model (id50-selected):** `hyb_ens6_6b_msaf`, entry 16. It averages six MLP heads on frozen
  embeddings (ESM2-650M; ESMC-600M final layer; ESMC-600M SAE-max; ESMC-600M layer 27 ⊕ ESM2;
  ESMC-600M final layer ⊕ SAE-max ⊕ ESM2; ESMC-6B SAE-max). It then blends in homology transfer
  from MMseqs2 and Foldseek on AlphaFold-DB 3Di, gated by top-hit identity. id50 0.914; full
  protocol New 0.640, Price 0.425, U-all 0.787, U-multi 0.751, T-70 0.653, T-50 0.574 (mean
  0.638). It is significantly better than retrained ESM2-BC on all six test sets, than released
  MIL-BC+ on four, and than EnzymeHunter on the four large sets (entry 15). It ties EnzymeHunter on
  New and Price.
- **Best single network:** ESM2-3B last ⊕ ESMC-600M last ⊕ ESMC-600M SAE-max (full test mean
  0.619). The fine-tuned ESMC-600M is the best single model on id50 (0.908).
- **Simplest strong model (entry 25):** the c3 head plus MMseqs2/Foldseek-AFDB homology
  (`hyb_c3_msaf`): id50 0.916, full test mean 0.632. It matches the best model on the five
  non-U-multi sets (mean 0.616 both) and trails only on multi-EC U-multi (0.716 vs 0.751).
- **Best single embedding (entry 26):** ESM2-3B last, full test mean 0.617 over 3 seeds
  (ESMC-600M SAE-max 0.605). With homology, it ties ESMC-600M SAE-max (0.625 vs 0.627).
- **What helped:** sparse-autoencoder (SAE) max-pooled features of ESMC, ensembling different
  feature heads, identity-gated homology, and structural homology (Foldseek).
- **What did not:** ESMC-6B hidden states, alternative losses, EC-hierarchy loss, block dropout,
  hidden-space kNN, a GBT stacker, seed averaging, real-vs-predicted 3Di, and fancier heads. Deeper,
  residual, SiLU/ReLU, other dropout values and gated heads all land within seed noise; block
  attention is worse (entry 20). Rare-class rebalancing (logit adjustment, loss reweighting) and a
  class-mean prototype output layer (entries 21–22). SaProt-650M-AF2 embeddings (entries 23–24): below
  ESM2-650M alone (0.569 vs 0.588), a tie when added to c3 or the hybrid, with or without SaProt's
  pLDDT < 70 masking.
- **Plateau:** every hybrid since entry 13 scores 0.913–0.916 on id50 and 0.633–0.642 on the full
  test mean. The seed noise is about 0.003.
- **Where the error is (entry 19):** 85% of test error is on queries below 50% identity to
  training, and half is on queries whose EC has fewer than 10 training examples. The id50 fold
  under-represents both, so it is a weak guide for the gains still available.

## Code

| file | what |
|---|---|
| `code/extract_plm.py` | Per-protein pooled embeddings: ESMC-600M/6B (final and SAE-layer hidden states, plus SAE features), ProtT5, ProstT5, ESM2-3B; SaProt-650M (in `.venv`). Runs in `envs/esmc` (Biohub `esm`), resumable, `--shard i/n` to split over GPUs. |
| `code/train_feat.py` | Heads on fixed embeddings, single or combined (MLP / residual MLP / block attention / gated / linear; GELU/SiLU/ReLU; BCE / ASL / focal; EMA; block dropout; EC-hierarchy aux loss; hidden-space kNN). Both protocols. |
| `code/knn_feat.py` | Cosine nearest-neighbour EC transfer on the same embeddings. |
| `code/homology.py` | MMseqs2 homology transfer; `--tool foldseek` for structural (ProstT5 3Di + AA) homology. |
| `code/prostt5_from_gguf.py` | Rebuilds ProstT5 (encoder + 3Di CNN) from Foldseek's GGUF copy, since huggingface.co is unreachable here. It agrees with Foldseek's own 3Di prediction on 99.8% of residues. |
| `code/ensemble.py` | Probability averaging plus the identity-gated homology blend. |
| `code/stack.py` | Gradient-boosted re-ranker over model and homology evidence (cluster-grouped CV). |
| `code/finetune_plm.py` | Partial fine-tuning of ESMC-600M (implemented, smoke-tested, deprioritized). |
| `scripts/feat.sh`, `scripts/extract_plm.sh` | Memory-capped wrappers. |
| `scripts/afdb_3di.py` | 3Di from AlphaFold-DB Swiss-Prot structures (Foldseek DB), ProstT5 fallback. |
| `scripts/afdb_plddt.py` | Per-residue pLDDT from `swissprot_pdb_v6.tar` (streamed) and pLDDT-masked 3Di for SaProt. |
| `scripts/rare_eval.py` | Per-enzyme F1 by training count of the query's rarest EC; post-hoc logit adjustment of rare classes (`--adjust tau:n0`, `--write`). |
| `scripts/stratify.py` | Per-enzyme F1 by top-hit identity bin (no hit / <30 / 30–50 / 50–70 / ≥70%), with each bin's share of the remaining error. |
| `scripts/feat_table.py` | Builds the results tables from `runs/feat/*/metrics.json`. |

Features live in `features/plm/<model>/<key>.npy` (float16, rows in `features/domains/all_union.csv`
order). `esm2-650m:last` is the original global ESM2 embedding, for like-for-like comparisons.

## Reference models (per-enzyme F1, from earlier work)

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESM2-BC (retrained) | 0.877 | 0.585 | 0.367 | 0.734 | 0.667 | 0.590 | 0.502 |
| MIL-BC+ (released) | 0.868 | 0.560 | 0.397 | 0.713 | 0.684 | 0.578 | 0.500 |
| DomAttn-flat | 0.877 | 0.577 | 0.479 | 0.728 | 0.638 | 0.598 | 0.514 |
| EnzymeHunter | – | 0.620 | 0.416 | 0.656 | 0.596 | 0.539 | 0.453 |

## Log

<!-- newest last; one entry per tested model -->

### 1. Training recipe for fixed-embedding heads (ESM2-650M global embedding, id50)

`train_feat.py` on the original ESM2-650M embedding. Per-enzyme F1; test columns come from the
id50-trained model and are for orientation only.

| run | head | id50 | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| esm2_mlp (40 ep) | MLP 1024, BCE | 0.863 | 0.541 | 0.308 | 0.707 | 0.599 | 0.565 | 0.494 |
| **esm2_bce_e100** | MLP 1024, BCE, 100 ep | **0.879** | 0.607 | 0.376 | 0.734 | 0.660 | 0.590 | 0.502 |
| esm2_bce_e100_lr3e3 | same, lr 3e-3 | 0.877 | 0.599 | 0.425 | 0.729 | 0.674 | 0.593 | 0.504 |
| esm2_focal_e100 | focal loss | 0.870 | 0.585 | 0.438 | 0.733 | 0.677 | 0.596 | 0.509 |
| esm2_asl_e100 | asymmetric loss | 0.868 | 0.596 | 0.365 | 0.715 | 0.667 | 0.582 | 0.497 |
| esm2_asl_e100_h2048l2 | ASL, 2 x 2048 | 0.868 | 0.593 | 0.370 | 0.712 | 0.684 | 0.575 | 0.485 |

- 40 epochs under-trains: mean-reduced BCE over 5,242 labels learns positives slowly. 100 epochs at
  lr 1e-3 with a cosine schedule matches ESM2-BC on id50 (0.879 vs 0.877).
- ASL and focal loss call more ECs: micro F1 drops below per-enzyme F1, and both lose on id50.
- **Default recipe from here on:** MLP (512 projection, one 1024 hidden layer, dropout 0.3, input
  dropout 0.1), BCE, AdamW lr 1e-3 with weight decay 0.05, cosine schedule, 100 epochs, batch 512.

### 2. Nearest-neighbour transfer on raw ESM2 embeddings (`knn_feat.py`, id50)

Cosine kNN with softmax-weighted votes; k and T are chosen on id50 (k=20, T=0.01, effectively the
first neighbour). Results: id50 0.806, New 0.534, Price 0.334, U-all 0.643, U-multi 0.623, T-70
0.488, T-50 0.376, far below the MLP. Raw PLM embedding space is a poor metric for EC identity, so
a supervised head (or a learned metric) is needed.

### 3. MMseqs2 homology transfer (`homology.py`, id50)

The top hit's ECs, plus any EC whose best hit scores at least 95% of the top bit score. With
id50-train as the database:

| | id50 | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-enzyme F1 | **0.893** | 0.618 | 0.393 | 0.740 | 0.684 | 0.602 | 0.506 |
| queries with a hit (e ≤ 1e-3) | 0.978 | 0.969 | 0.953 | 0.919 | 0.922 | 0.907 | 0.866 |

**Plain homology transfer beats every learned model** on id50 and on all six test sets. The id50
fold allows no training neighbour above 50% identity with 50% coverage, yet sensitive MMseqs2 still
finds informative remote homologs. None of the earlier models (MIL-BC/BC+, ESM2-BC, DomAttn) were
compared against this baseline.

### 4. Hybrid: ESM2 MLP + homology, gated by top-hit identity (`ensemble.py`, id50)

`p = (1 - a) p_MLP + a p_hom`, with `a = clip((id - lo)/(hi - lo), 0, 1) * amax`. The grid search on
id50 chose lo 0, hi 0.8, amax 0.75.

| | id50 | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hyb_esm2_mmseqs | **0.906** | 0.631 | 0.414 | 0.755 | 0.697 | 0.623 | 0.535 |

That is +0.027 over the MLP and +0.013 over homology on id50, and the gain carries to every test
set. The blend is close to "trust homology in proportion to identity", and the model fills in where
hits are weak or missing.

### 5. Full protocol: ESM2 MLP, homology, and their hybrid (test sets)

Each model was retrained, or its database rebuilt, on all of split100; the hybrid reuses the
gate chosen on id50 (lo 0, hi 0.8, amax 0.75). Per-enzyme F1:

| model | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|
| ESM2-BC (reference) | 0.585 | 0.367 | 0.734 | 0.667 | 0.590 | 0.502 |
| EnzymeHunter (reference) | 0.620 | 0.416 | 0.656 | 0.596 | 0.539 | 0.453 |
| full_esm2_bce_e100 (MLP) | 0.595 | 0.414 | 0.738 | 0.682 | 0.594 | 0.505 |
| full_mmseqs (homology) | 0.622 | 0.385 | 0.754 | 0.684 | 0.610 | 0.514 |
| **full_hyb_esm2_mmseqs** | **0.631** | 0.400 | **0.765** | **0.744** | **0.626** | **0.540** |

The hybrid is best on five of the six sets. It is the first model here to beat EnzymeHunter on
New-392, and it beats ESM2-BC by 0.03–0.08 everywhere. Bootstrap CIs will follow in the final
comparison.

### 6. ESMC-600M features and ESMC-600M SAE features (id50 protocol, default recipe)

Features are mean-pooled over residues: `last` is the final hidden state, `mid` the layer-27
hidden state (the SAE's input layer). `sae_mean` / `sae_max` are the TopK-64 codes of
`ESMC-600M-sae-layer27-k64-codebook16384` (16,384 features), mean- or max-pooled over residues and
`log1p`-transformed.

| run | features | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| esm2_bce_e100 (s0 / s1) | ESM2-650M | 0.879 / 0.876 | 0.607 | 0.376 | 0.734 | 0.660 | 0.590 | 0.502 | 0.578 |
| esmc600_last (s0 / s1) | ESMC last | **0.897 / 0.890** | 0.571 | 0.379 | 0.729 | 0.713 | 0.581 | 0.483 | 0.576 |
| esmc600_mid | ESMC layer 27 | 0.878 | 0.514 | 0.300 | 0.716 | 0.623 | 0.569 | 0.501 | 0.537 |
| esmc600_saemean | SAE mean | 0.881 | 0.574 | 0.346 | 0.728 | 0.659 | 0.589 | 0.512 | 0.568 |
| esmc600_saemax | SAE max | 0.893 | 0.594 | 0.404 | **0.755** | 0.674 | 0.612 | **0.537** | 0.596 |
| esmc600_last_saemean | last + SAE mean | 0.894 | 0.591 | 0.361 | 0.744 | 0.659 | 0.598 | 0.513 | 0.578 |
| esmc600_last_esm2 | ESMC last + ESM2 | **0.898** | 0.591 | 0.405 | 0.742 | 0.674 | 0.600 | 0.511 | 0.587 |
| esmc600_mid_esm2 | ESMC mid + ESM2 | 0.878 | 0.596 | **0.443** | 0.746 | 0.685 | **0.612** | 0.532 | **0.602** |

Test columns come from id50-trained models. Training-seed noise is about ±0.004 on id50 and about
±0.01–0.03 on the small test sets.

- **ESMC-600M final layer beats ESM2-650M on id50** (+0.015 over two seeds), the largest single
  feature gain so far. It does **not** carry over to the test sets: it is worse on New and T-50, and
  the full-protocol retrain gives New 0.561 vs 0.595. The id50 fold (split100 clusters) and the test
  sets (newer Swiss-Prot entries, Price, identity-filtered sets) disagree, so features chosen by id50
  alone may not transfer.
- **SAE max-pooled features are the most robust single feature.** id50 0.893, close to ESMC last,
  and the best test mean of any single feature (0.596), led by U-all, T-70 and T-50. Max pooling
  over residues keeps local motif detectors, such as active-site features, that mean pooling dilutes.
  Mean-pooled SAE codes are clearly worse.
- The layer-27 hidden state is worse than the final layer for EC prediction.
- Concatenations help modestly. last + ESM2 is the best id50 model (0.898), and mid + ESM2 has the
  best test mean (0.602) despite a mediocre id50. Combining two PLMs mostly helps the remote sets.

### 7. Feature ensembles, with and without homology (id50 protocol)

Probability averages of the id50 models from entries 1 and 6, optionally blended with MMseqs2 by
top-hit identity. The gate is chosen on id50. Test columns are id50-trained.

| ensemble | id50 | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ens2 = ESM2 + SAE-max | 0.898 | 0.611 | 0.413 | 0.759 | 0.693 | 0.620 | 0.543 |
| ens4 = ESM2 + ESMC-last + SAE-max + (ESMC-mid ⊕ ESM2) | 0.899 | 0.612 | 0.444 | 0.762 | 0.704 | 0.628 | 0.546 |
| hyb_ens2 (+ homology; lo .1, hi .8, amax .75) | 0.913 | 0.621 | 0.422 | 0.771 | 0.727 | 0.634 | 0.552 |
| **hyb_ens4** (+ homology; lo 0, hi 1, amax .9) | **0.914** | 0.629 | 0.432 | **0.774** | **0.751** | **0.642** | **0.563** |
| hyb_esm2_mmseqs (entry 4, for reference) | 0.906 | 0.631 | 0.414 | 0.755 | 0.697 | 0.623 | 0.535 |

Averaging models on different PLM features adds about +0.02 per-enzyme F1 on the remote sets
(T-70, T-50, U-all) over the best single model. Homology adds another +0.015 on id50 and on most
test sets. Full-protocol members are queued.

### 8. Stacked re-ranker (`stack.py`): gradient-boosted trees over model and homology evidence

Candidates per query are the top 10 ECs by mean model probability plus every EC carried by an
MMseqs2 hit. Per candidate, a HistGradientBoosting classifier sees each model's probability, rank
and margin, the bit-score ratio of the EC's best hit, the top hit's identity and bits, the number of
hit ECs, and the EC's training frequency. It is trained on id50 predictions from the id50-protocol
bases (ESM2 MLP + ESMC-last MLP + MMseqs2).

| | id50 (out of fold) | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stack, id50 bases | 0.922 | 0.627 | 0.395 | 0.766 | 0.701 | 0.625 | 0.538 |
| stack, full bases | – | 0.633 | 0.412 | 0.768 | 0.716 | 0.631 | 0.543 |
| full_hyb_esm2_mmseqs (entry 5) | – | 0.631 | 0.400 | 0.765 | 0.744 | 0.626 | 0.540 |

- **Pitfall caught:** with protein-level 5-fold CV the stacker reported id50 0.952. Members of one
  sequence cluster landed on both sides of a fold, and the trees memorized EC-level shortcuts.
  Cluster-grouped folds (MMseqs2 at 30% identity, 80% coverage) give the honest 0.922.
- On the test sets the stacker is about level with the three-parameter identity gate (±0.01), so
  the extra machinery buys little. The simple gate stays the default combiner; the stacker is kept as
  an alternative.

### 9. SAE-max follow-ups and three-way feature concatenation (id50 protocol)

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAE-max (log1p, entry 6) | 0.893 | 0.594 | 0.404 | 0.755 | 0.674 | 0.612 | 0.537 | 0.596 |
| SAE-max, seed 1 | 0.890 | 0.595 | 0.405 | 0.747 | 0.691 | 0.610 | 0.527 | 0.596 |
| SAE-max, no log1p | 0.895 | 0.591 | 0.393 | 0.749 | 0.660 | 0.606 | 0.526 | 0.587 |
| SAE-max, proj 1024 / hidden 2048 | 0.893 | 0.601 | 0.417 | 0.751 | 0.680 | 0.615 | 0.536 | 0.600 |
| SAE-max ⊕ ESM2 | 0.895 | 0.600 | 0.404 | 0.755 | 0.677 | 0.611 | 0.529 | 0.596 |
| **ESMC-last ⊕ SAE-max ⊕ ESM2** | **0.903** | 0.603 | 0.419 | **0.761** | 0.684 | **0.621** | **0.537** | **0.604** |

- Seed noise on id50 is about ±0.003 to 0.004, so the log1p, width and SAE ⊕ ESM2 variants are all
  ties. log1p is kept: it scores about 0.01 higher on the test sets.
- Concatenating three blocks (ESMC final layer, SAE-max, ESM2) gives the best single network so
  far (id50 0.903, test mean 0.604). It still trails averaging the separate models (ens4: 0.899
  id50, test mean 0.616). Ensembling separately trained heads remains the better way to combine
  features.

### 10. Full protocol: best ensemble and paired bootstrap CIs

Full-protocol members are retrained on all of split100. The gate settings are copied from the id50
choice.

| model (full protocol) | New | Price | U-all | U-multi | T-70 | T-50 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESM2 MLP | 0.595 | 0.414 | 0.738 | 0.682 | 0.594 | 0.505 | 0.588 |
| ESMC-600M last | 0.561 | 0.405 | 0.739 | 0.692 | 0.585 | 0.493 | 0.579 |
| ESMC-600M SAE-max | 0.600 | **0.425** | 0.762 | 0.682 | 0.619 | 0.537 | 0.604 |
| ESMC-mid ⊕ ESM2 | 0.604 | 0.403 | 0.756 | 0.691 | 0.613 | 0.536 | 0.600 |
| MMseqs2 | 0.622 | 0.385 | 0.754 | 0.684 | 0.610 | 0.514 | 0.595 |
| hyb ESM2 + MMseqs2 (entry 5) | 0.631 | 0.400 | 0.765 | **0.744** | 0.626 | 0.540 | 0.618 |
| hyb_ens2 (ESM2 + SAE-max + hom) | 0.626 | 0.394 | **0.777** | 0.714 | 0.634 | 0.550 | 0.616 |
| **hyb_ens4** (4 feature heads + hom) | **0.633** | 0.409 | 0.775 | 0.733 | **0.637** | **0.554** | **0.624** |

Per-enzyme F1 with 95% paired-bootstrap CIs (2,000 resamples;
`features/logs/bootstrap_autoresearch.txt`):

| | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---|---|---|---|---|---|
| released MIL-BC+ | .560 [.515,.603] | .397 [.329,.462] | .713 [.689,.739] | .684 [.620,.749] | .578 [.550,.603] | .500 [.466,.534] |
| EnzymeHunter | .620 [.574,.667] | .416 [.345,.488] | .656 [.628,.685] | .596 [.523,.674] | .539 [.512,.567] | .453 [.419,.486] |
| hyb_ens4 | .633 [.587,.676] | .409 [.334,.490] | .775 [.752,.800] | .733 [.669,.800] | .637 [.609,.664] | .554 [.518,.589] |
| hyb_ens4 − ESM2-BC (retrained) | **+.048*** | +.043 | **+.041*** | **+.067*** | **+.047*** | **+.053*** |
| hyb_ens4 − released MIL-BC+ | **+.073*** | +.013 | **+.062*** | +.049 | **+.060*** | **+.054*** |
| hyb_ens4 − EnzymeHunter | +.012 | −.006 | **+.119*** | **+.138*** | **+.098*** | **+.101*** |
| hyb_ens4 − hyb ESM2+MMseqs2 | +.002 | +.009 | **+.010*** | −.011 | **+.011*** | **+.015*** |
| hyb_ens4 − MMseqs2 | +.011 | +.025 | **+.022*** | **+.050*** | **+.028*** | **+.041*** |
| SAE-max alone − ESM2-BC (retrained) | +.016 | **+.058*** | **+.027*** | +.016 | **+.029*** | **+.036*** |

`*` marks a 95% interval that excludes 0.

- hyb_ens4 is significantly better than released MIL-BC+ on 4 of 6 sets and better than
  EnzymeHunter on the 4 large sets. On New and Price it ties both (Price has only 149 enzymes).
- Adding the ESMC and SAE heads to the ESM2 + homology hybrid gives a small but significant gain
  (+0.010 to +0.015) on U-all, T-70 and T-50.
- On its own, the ESMC-600M SAE-max head beats the retrained ESM2 baseline on 4 sets without any
  homology. The sparse dictionary features carry real EC signal.

### 11. Hierarchy auxiliary loss, hidden-space kNN, block dropout, larger ensembles (id50 protocol)

`--hier w` adds linear heads on the shared hidden layer that predict the EC prefixes at levels 1, 2
and 3, each trained with BCE at weight `w`. `--hknn` replaces the output layer with cosine kNN over
training proteins in the head's hidden space; k and T are chosen on id50. `--block-dropout 0.2`
randomly drops whole feature blocks during training.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM2 MLP (entry 1) | 0.879 | 0.607 | 0.376 | 0.734 | 0.660 | 0.590 | 0.502 | 0.578 |
| ESM2 MLP + hier 0.3 | 0.857 | 0.563 | 0.362 | 0.708 | 0.656 | 0.573 | 0.493 | 0.559 |
| ESM2 MLP + hier 1.0 | 0.855 | 0.558 | 0.318 | 0.693 | 0.664 | 0.558 | 0.477 | 0.545 |
| last ⊕ SAE-max ⊕ ESM2 (entry 9) | 0.903 | 0.603 | 0.419 | 0.761 | 0.684 | 0.621 | 0.537 | 0.604 |
| same, block dropout 0.2 | 0.904 | 0.607 | 0.411 | 0.762 | 0.684 | 0.624 | 0.541 | 0.605 |
| same, hidden-space kNN | 0.903 | 0.609 | 0.404 | 0.766 | 0.700 | 0.625 | 0.545 | 0.608 |
| hyb_ens4 (entry 7) | 0.914 | 0.629 | 0.432 | 0.774 | 0.751 | 0.642 | 0.563 | 0.632 |
| hyb_ens5 = ens4 + (last ⊕ SAE-max ⊕ ESM2), + hom | 0.915 | 0.628 | 0.446 | 0.778 | 0.742 | 0.647 | 0.569 | 0.635 |
| hyb_ens6 = ens5 + its hidden-kNN, + hom | 0.916 | 0.632 | 0.442 | 0.777 | 0.733 | 0.647 | 0.570 | 0.634 |

- The hierarchy loss hurts, by −0.02 on id50 and on every test set. The aux losses average over a
  few hundred prefixes, while the main loss averages over 5,242 mostly-negative ECs, so even
  weight 0.3 dominates the gradient. A run at 0.03 is queued to test that explanation. (The SAE-max
  + hier run hit a CUDA OOM while the ProtT5 extraction peaked at 35 GB and was not rerun.)
- Block dropout and hidden-space kNN tie the plain head on id50. kNN is about +0.004 on the test
  sets.
- Adding more heads built on the same two PLMs (ESMC-600M, ESM2-650M) is saturating: ens4 → ens6
  gains ≤ 0.003. More gains likely need new information, either different PLMs (ProtT5, ESM2-3B,
  ESMC-6B; extraction in progress) or structure.

### 12. ProtT5-XL (UniRef50) features (mean-pooled encoder output, 1024-d)

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ProtT5 MLP | 0.870 | 0.564 | 0.337 | 0.731 | 0.631 | 0.571 | 0.489 | 0.554 |
| ProtT5 MLP, full protocol | – | 0.574 | 0.393 | 0.744 | 0.649 | 0.589 | 0.509 | 0.576 |
| **ProtT5 ⊕ ESMC-last ⊕ SAE-max** | **0.905** | 0.600 | **0.449** | 0.764 | 0.684 | 0.626 | **0.550** | **0.612** |
| ESM2 MLP + hier 0.03 (entry 11 follow-up) | 0.877 | 0.581 | 0.390 | 0.724 | 0.702 | 0.589 | 0.503 | 0.582 |
| hyb_ens5p = ens4 + ProtT5, + hom | 0.915 | 0.632 | 0.431 | 0.772 | 0.751 | 0.642 | 0.562 | 0.632 |
| hyb_ens6p = ens5 + (ProtT5 ⊕ ESMC-last ⊕ SAE-max), + hom | 0.915 | 0.625 | 0.436 | 0.774 | 0.742 | 0.641 | 0.562 | 0.630 |
| hyb_ens7p = ens6p + ProtT5, + hom | 0.914 | 0.625 | 0.426 | 0.777 | 0.742 | 0.642 | 0.562 | 0.629 |

- On its own, ProtT5 is weaker than ESM2-650M (−0.009 id50, −0.024 test mean). As one block of a
  concatenated network, though, it yields the best single network so far (id50 0.905, test mean
  0.612, Price 0.449).
- Adding ProtT5 heads to the homology hybrid changes nothing (all within ±0.003 of hyb_ens5). Once
  homology is in, ensemble gains from yet another sequence PLM are exhausted.
- At weight 0.03, the hierarchy loss ties the plain head. That fits the explanation that the aux
  loss dominated the gradient at larger weights. It gives no gain, so the idea is dropped.

### 13. Structure: ProstT5 3Di + Foldseek homology transfer

Huggingface.co is unreachable from this host. ProstT5 was rebuilt from the GGUF weights that
Foldseek serves from its own mirror (`code/prostt5_from_gguf.py`). The rebuilt model agrees with
Foldseek's own 3Di prediction on 99.8% of 53k residues, and runs about 2× faster batched in bf16.
`extract_plm.py --model prostt5` writes the ProstT5 embedding and a predicted 3Di sequence for all
228,888 proteins. `homology.py --tool foldseek` then searches queries against the protocol's training
proteins with Foldseek on 3Di + amino acids (`--alignment-type 2`, e-value 1e-3, 300 hits). This
gives structure-aware homology without any 3D coordinates. AlphaFold DB was not used because the
Price set has RefSeq IDs with no AFDB models; predicting 3Di from sequence treats every set the
same way.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMseqs2, id50 protocol (entry 3) | 0.893 | 0.618 | 0.393 | 0.740 | 0.684 | 0.602 | 0.506 | 0.591 |
| Foldseek (ProstT5 3Di), id50 protocol | 0.884 | 0.615 | 0.423 | 0.752 | 0.701 | 0.617 | 0.530 | 0.606 |
| MMseqs2, full protocol | – | 0.622 | 0.385 | 0.754 | 0.684 | 0.610 | 0.514 | 0.595 |
| **Foldseek, full protocol** | – | 0.623 | **0.424** | 0.755 | 0.696 | **0.619** | **0.529** | **0.608** |

Hit coverage rises as well: Foldseek finds a hit for 96 to 100% of the queries in every set.
Combined with the five-head ensemble of entry 11 (`ensemble.py --homology mmseqs foldseek`, which
averages (`mean`) or unions (`max`) the two tools' EC calls and gates on the higher top-hit identity):

| hybrid (5 feature heads + …) | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMseqs2 (hyb_ens5, id50 protocol) | 0.915 | 0.628 | 0.446 | 0.778 | 0.742 | 0.647 | 0.569 | 0.635 |
| Foldseek | 0.906 | 0.625 | 0.468 | 0.777 | 0.731 | 0.650 | 0.575 | 0.638 |
| MMseqs2 + Foldseek, mean | 0.913 | 0.629 | 0.452 | 0.781 | 0.759 | 0.651 | 0.574 | 0.641 |
| MMseqs2 + Foldseek, max | 0.915 | 0.626 | 0.460 | 0.778 | 0.739 | 0.650 | 0.573 | 0.638 |
| *full protocol:* hyb_ens4 + MMseqs2 (entry 10) | – | 0.633 | 0.409 | 0.775 | 0.733 | 0.637 | 0.554 | 0.624 |
| hyb_ens5 + MMseqs2 | – | 0.628 | 0.421 | 0.780 | 0.740 | 0.641 | 0.560 | 0.628 |
| **hyb_ens5 + MMseqs2 + Foldseek, mean** | – | 0.632 | 0.437 | **0.782** | **0.749** | **0.647** | **0.567** | **0.636** |
| hyb_ens5 + MMseqs2 + Foldseek, max | – | 0.629 | 0.447 | 0.779 | 0.730 | 0.646 | 0.566 | 0.633 |
| ESM2 MLP + Foldseek | – | 0.629 | 0.434 | 0.764 | 0.707 | 0.631 | 0.548 | 0.619 |

- Alone, structural homology beats sequence homology on the remote and out-of-distribution sets:
  Price +0.039, T-50 +0.015, T-70 +0.009 (full protocol). On id50 it is slightly worse (−0.009).
  id50 is a near-homolog fold, where amino-acid identity is the sharper signal.
- The two tools are complementary. Averaging their calls in the gate gives the best full-protocol
  result so far, a test mean of 0.636 (+0.012 over hyb_ens4), with the best U-all, U-multi, T-70
  and T-50 to date. On id50 the mean and max variants tie (0.913 vs 0.915, within seed noise),
  so id50 cannot separate them. `mean` is used from here on, with the caveat that its test-set lead
  over `max` (0.636 vs 0.633) was seen after the fact and is not an id50 selection.

### 14. ESM2-3B and ProstT5 embeddings; larger ensembles; stacker with structural evidence

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM2-650M MLP (entry 1) | 0.879 | 0.607 | 0.376 | 0.734 | 0.660 | 0.590 | 0.502 | 0.578 |
| ESM2-3B last-layer MLP | 0.890 | 0.612 | 0.423 | 0.754 | 0.685 | 0.619 | 0.535 | 0.604 |
| ESM2-3B layer-24 MLP | 0.881 | 0.573 | 0.296 | 0.725 | 0.651 | 0.580 | 0.505 | 0.555 |
| ProstT5 MLP | 0.890 | 0.584 | 0.366 | 0.733 | 0.663 | 0.583 | 0.494 | 0.571 |
| ProstT5 ⊕ ESM2-650M | 0.892 | 0.591 | 0.395 | 0.738 | 0.670 | 0.587 | 0.496 | 0.579 |
| ProstT5 ⊕ ESMC-last ⊕ SAE-max | 0.903 | 0.593 | 0.424 | 0.762 | 0.667 | 0.620 | 0.541 | 0.601 |
| ESM2-3B ⊕ ESMC-last ⊕ SAE-max | 0.904 | 0.587 | 0.434 | 0.756 | 0.671 | 0.613 | 0.527 | 0.598 |
| *full protocol:* ESM2-3B last | – | 0.629 | 0.418 | 0.758 | 0.719 | 0.624 | 0.540 | **0.615** |
| ProstT5 | – | 0.602 | 0.399 | 0.733 | 0.651 | 0.590 | 0.497 | 0.579 |
| ProstT5 ⊕ ESMC-last ⊕ SAE-max | – | 0.596 | 0.433 | 0.770 | 0.689 | 0.626 | 0.547 | 0.610 |

- ESM2-3B's final layer is the best single-embedding head so far: full-protocol test mean 0.615,
  against 0.604 for ESMC-600M SAE-max and 0.588 for ESM2-650M.
- The ProstT5 embedding is a strong head on id50 (0.890, versus 0.870 for ProtT5, its unfinetuned
  parent). On the test sets it is ordinary, so its structure-awareness transfers better through
  3Di + Foldseek (entry 13) than through the pooled embedding.

Ensembles (id50 protocol; `msfs` = gate on MMseqs2 + Foldseek, mean):

| ensemble | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ens7 = ens5 + ESM2-3B + ProstT5 (no homology) | 0.902 | 0.608 | 0.445 | 0.765 | 0.707 | 0.628 | 0.547 | 0.617 |
| ens8 = ens7 + (ProstT5 ⊕ ESMC-last ⊕ SAE-max) (no homology) | 0.903 | 0.608 | 0.456 | 0.767 | 0.712 | 0.632 | 0.551 | 0.621 |
| div6 = 6 diverse heads, all PLMs (no homology) | **0.905** | 0.607 | 0.447 | 0.767 | 0.717 | 0.630 | 0.549 | 0.620 |
| hyb_ens5_msfs (entry 13) | 0.913 | 0.629 | 0.452 | 0.781 | 0.759 | 0.651 | 0.574 | 0.641 |
| hyb_ens7_msfs | 0.914 | 0.629 | 0.425 | 0.781 | 0.759 | 0.648 | 0.569 | 0.635 |
| hyb_ens8_msfs | 0.914 | 0.630 | 0.438 | 0.780 | 0.751 | 0.649 | 0.571 | 0.637 |
| hyb_div6_msfs | 0.915 | 0.626 | 0.438 | 0.781 | 0.746 | 0.648 | 0.569 | 0.635 |
| *full protocol:* ens8 (no homology) | – | 0.609 | 0.460 | 0.770 | 0.719 | 0.631 | 0.548 | 0.623 |
| hyb_ens7_msfs | – | 0.637 | 0.427 | 0.779 | 0.738 | 0.645 | 0.562 | 0.631 |
| hyb_ens8_msfs | – | 0.634 | 0.441 | 0.779 | 0.738 | 0.646 | 0.563 | 0.634 |
| hyb_ens5_msfs (entry 13) | – | 0.632 | 0.437 | 0.782 | 0.749 | 0.647 | 0.567 | 0.636 |

Stacker (`stack.py`, now taking several homology runs) over ESM2-650M, ESMC-last, SAE-max,
ESM2-3B and ProstT5 heads:

| stacker | id50 (OOF) | full: New | Price | U-all | U-multi | T-70 | T-50 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| + MMseqs2 evidence | 0.923 | 0.638 | 0.386 | 0.772 | 0.703 | 0.633 | 0.544 | 0.613 |
| + MMseqs2 + Foldseek evidence | 0.918 | 0.638 | 0.366 | 0.766 | 0.693 | 0.627 | 0.541 | 0.605 |

- New embeddings lift the model-only ensembles: the full-protocol test mean reaches 0.623 without
  homology, up from 0.616 for ens4. After the homology gate, though, everything lands at 0.631 to
  0.636 (full protocol), within seed noise. Homology already recovers most of what the extra heads
  add, so the hybrid is on a plateau.
- The stacker gets the highest id50 score but transfers worse: it learns the id50 near-homolog
  regime (identity ≥ 50% to training). Adding Foldseek evidence makes this worse. The
  three-parameter gate stays the combiner.

### 15. ESMC-6B features (final, layer-60, layer-60 SAE) and ensembles with them

`extract_plm.py` on ESMC-6B: `last` (final post-norm hidden state, 2560-d), `mid` (layer 60, the SAE
layer), and `sae_mean` / `sae_max` (16,384 TopK-64 features of `ESMC-6B-sae-layer60-k64-codebook16384`).
Each is mean-pooled over residues (max-pooled for `sae_max`). The head uses the default recipe, with
log1p on the SAE blocks.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESMC-6B last | 0.874 | 0.606 | 0.338 | 0.703 | 0.623 | 0.562 | 0.460 | 0.549 |
| ESMC-6B layer 60 | 0.879 | 0.546 | 0.256 | 0.734 | 0.713 | 0.574 | 0.503 | 0.554 |
| ESMC-6B SAE-mean | 0.889 | 0.554 | 0.348 | 0.732 | 0.629 | 0.581 | 0.507 | 0.558 |
| **ESMC-6B SAE-max** | **0.890** | 0.617 | 0.361 | 0.744 | 0.696 | 0.605 | 0.530 | 0.592 |
| *full protocol:* ESMC-6B last | – | 0.592 | 0.347 | 0.714 | 0.632 | 0.561 | 0.457 | 0.550 |
| ESMC-6B SAE-max | – | 0.624 | 0.402 | 0.758 | 0.690 | 0.618 | 0.543 | 0.606 |
| ESMC-600M SAE-max (entry 10) | – | 0.600 | 0.425 | 0.762 | 0.682 | 0.619 | 0.537 | 0.604 |
| ESM2-3B last (entry 14) | – | 0.629 | 0.418 | 0.758 | 0.719 | 0.624 | 0.540 | 0.615 |
| **ESM2-3B ⊕ ESMC-600M last ⊕ SAE-max** | – | 0.604 | **0.464** | **0.772** | 0.689 | **0.635** | **0.553** | **0.619** |

- Scaling ESMC from 600M to 6B does not help pooled hidden states. The 6B final-layer head (0.549
  full test mean) is below ESMC-600M last (0.579) and ESM2-650M (0.588). The stored features have no
  overflow or non-finite values, so the weakness is not a float16 artifact. Mean-pooling a 2,560-d
  residual stream with a few huge outlier dimensions (layer 60 has one dimension averaging ~530)
  seems to lose EC information that the SAE recovers.
- The 6B SAE-max features match the 600M ones: 0.606 versus 0.604 on the full protocol, with gains on
  New and losses on Price. Sparse dictionary features are again the best way to read an ESMC model.
- ESM2-3B ⊕ ESMC-600M last ⊕ SAE-max, retrained on the full protocol, is the best single head so far
  (0.619). It scores only 0.598 on the id50-protocol test columns, so the full retrain gains more
  for this concatenation than for single blocks.

Ensembles (`msfs` gate, grid as in entry 13, chosen on id50; `c3` = the 3-block concatenation above):

| ensemble | id50 | full: New | Price | U-all | U-multi | T-70 | T-50 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyb_ens5_msfs (entry 13) | 0.913 | 0.632 | 0.437 | 0.782 | 0.749 | 0.647 | 0.567 | 0.636 |
| **hyb_ens6_6b** = ens5 + ESMC-6B SAE-max | 0.914 | 0.635 | 0.441 | 0.783 | 0.742 | 0.650 | 0.572 | **0.637** |
| hyb_ens6_c3 = ens5 + c3 | 0.914 | 0.632 | 0.437 | 0.782 | 0.738 | 0.648 | 0.567 | 0.634 |
| hyb_ens7_6b_c3 = ens5 + both | 0.914 | **0.640** | 0.441 | 0.781 | 0.733 | 0.650 | 0.571 | 0.636 |
| hyb_div4 = c3 + 6B SAE-max + ESM2-3B + 600M SAE-max | 0.913 | 0.635 | 0.441 | 0.781 | 0.729 | 0.648 | 0.567 | 0.633 |

Paired bootstrap (2,000 resamples; `features/logs/bootstrap_autoresearch2.txt`), per-enzyme F1:

| | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---|---|---|---|---|---|
| hyb_ens6_6b | .635 [.589,.680] | .441 [.365,.518] | .783 [.760,.808] | .742 [.681,.806] | .650 [.622,.677] | .572 [.534,.607] |
| hyb_ens6_6b − ESM2-BC (retrained) | **+.051*** | **+.074*** | **+.049*** | **+.075*** | **+.060*** | **+.070*** |
| hyb_ens6_6b − released MIL-BC+ | **+.076*** | +.044 | **+.070*** | +.058 | **+.072*** | **+.072*** |
| hyb_ens6_6b − EnzymeHunter | +.015 | +.025 | **+.127*** | **+.146*** | **+.111*** | **+.118*** |
| hyb_ens6_6b − hyb_ens4 (entry 10) | +.003 | **+.031*** | **+.008*** | +.009 | **+.013*** | **+.017*** |
| hyb_ens6_6b − hyb_ens5_msfs | +.003 | +.003 | +.001 | −.007 | +.003 | +.005 |
| c3 head alone − ESM2-BC (retrained) | +.019 | **+.097*** | **+.037*** | +.022 | **+.044*** | **+.051*** |

- Adding ESMC-6B heads changes nothing measurable: every hybrid scores 0.913–0.914 on id50 and
  0.633–0.637 on the full test sets. hyb_ens6_6b is nominally best but indistinguishable from
  hyb_ens5_msfs (all paired intervals include 0).
- Compared with hyb_ens4, the entry-10 headline, the MMseqs2 + Foldseek hybrids are significantly
  better on Price, U-all, T-70 and T-50. That gain comes from adding Foldseek (entry 13), not from
  the larger PLMs.
- Against the retrained ESM2-BC baseline, hyb_ens6_6b is now significant on all six test sets.
  Against released MIL-BC+ it is significant on four (Price and U-multi have wide intervals).
- The c3 head alone, with no homology and a single network, beats retrained ESM2-BC on four sets.

### 16. Real AlphaFold structures: AFDB 3Di instead of ProstT5-predicted 3Di for Foldseek

`foldseek databases Alphafold/Swiss-Prot` (Foldseek mirror; AFDB model v6, 590k structures, 2.5 GB)
gives 3Di strings computed from actual AlphaFold2 models. `scripts/afdb_3di.py` swaps a protein's
ProstT5 3Di for the AFDB one when its accession is in AFDB and the sequence is identical. Coverage
is 98.7% of training proteins, 98.1% of the id50 fold, 96–99% of New, U-all and U-multi, 85–87% of
T-70 and T-50, and 0% of Price (no UniProt accessions), which keeps ProstT5 3Di. ProstT5 and AFDB
3Di agree on only 68% of residues. `homology.py --tool foldseek --states features/plm/afdb/3di.fasta`
runs the search, with the same settings as entry 13.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Foldseek, ProstT5 3Di (entry 13; full) | 0.884 | 0.623 | 0.424 | 0.755 | 0.696 | 0.619 | 0.529 | 0.608 |
| Foldseek, AFDB 3Di (full) | **0.890** | 0.630 | 0.409 | 0.761 | 0.685 | 0.624 | 0.534 | 0.607 |
| hyb_ens6_6b, MMseqs2 + Foldseek-ProstT5 (entry 15) | 0.914 | 0.635 | 0.441 | 0.783 | 0.742 | 0.650 | 0.572 | 0.637 |
| hyb_ens6_6b, Foldseek-AFDB only | 0.904 | 0.630 | 0.453 | 0.782 | 0.726 | 0.648 | 0.572 | 0.635 |
| **hyb_ens6_6b_msaf**, MMseqs2 + Foldseek-AFDB | 0.914 | **0.640** | 0.425 | **0.787** | **0.751** | 0.653 | 0.574 | 0.638 |
| hyb_ens6_6b, MMseqs2 + both Foldseeks | 0.911 | 0.636 | 0.453 | 0.784 | 0.748 | 0.653 | **0.576** | **0.642** |

Gate chosen on id50 in each case. Paired bootstrap differences (`features/logs/bootstrap_afdb.txt`):

| A − B | New | Price | U-all | U-multi | T-70 | T-50 |
|---|---|---|---|---|---|---|
| Foldseek AFDB − Foldseek ProstT5 | +.008 | −.015 | +.007 | −.011 | +.006 | +.004 |
| msaf − hyb_ens6_6b | **+.004*** | −.016 | **+.003*** | +.009 | +.002 | +.003 |
| MMseqs2 + both − hyb_ens6_6b | +.001 | +.012 | +.001 | +.006 | +.003 | +.004 |

- Real structures beat ProstT5's sequence-predicted 3Di on id50 (0.890 vs 0.884) and on four of the
  five test sets that have AFDB coverage. All of those test-set gains are within noise.
- Price loses: its queries keep ProstT5 3Di but now search AFDB 3Di targets, a mismatch that costs
  0.015. A predicted-3Di model of the query is better matched to a predicted-3Di database.
- In the hybrid, AFDB structure gives tiny, sometimes significant gains of +0.003 to +0.004 on New
  and U-all. hyb_ens6_6b_msaf is the id50-selected hybrid (it ties 0.914), with a full-protocol test
  mean of 0.638. The three-tool gate scores higher on the test sets (0.642) but lower on id50
  (0.911), so the protocol does not select it.
- Conclusion: homology transfer is saturated. MMseqs2, ProstT5-3Di Foldseek and AFDB-3Di Foldseek
  find largely the same neighbours for these sets.

### 17. Seed replicates: how big is the noise, and does averaging seeds help?

The six hyb_ens6_6b members were retrained with seeds 1 and 2 on both protocols (24 runs; recipe
unchanged).

| member, full-protocol test mean | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| ESM2-650M | 0.588 | 0.592 | 0.594 |
| ESMC-600M last | 0.579 | 0.585 | 0.589 |
| ESMC-600M SAE-max | 0.604 | 0.604 | 0.606 |
| ESMC-600M mid ⊕ ESM2 | 0.600 | 0.606 | 0.608 |
| ESMC-600M last ⊕ SAE-max ⊕ ESM2 | 0.605 | 0.610 | 0.609 |
| ESMC-6B SAE-max | 0.606 | 0.604 | 0.594 |

| ensemble (full protocol) | id50 | New | Price | U-all | U-multi | T-70 | T-50 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyb_ens6_6b, seed-0 members (entry 15) | 0.914 | 0.635 | 0.441 | 0.783 | 0.742 | 0.650 | 0.572 | 0.637 |
| same gate, seed-1 members | – | 0.631 | 0.456 | 0.777 | 0.738 | 0.645 | 0.565 | 0.635 |
| same gate, seed-2 members | – | 0.637 | 0.459 | 0.780 | 0.733 | 0.650 | 0.570 | 0.638 |
| ens6_6b, no homology, 1 seed | 0.900 | 0.613 | 0.437 | 0.772 | 0.728 | 0.632 | 0.552 | 0.622 |
| ens6_6b, no homology, 3 seeds averaged | 0.900 | 0.612 | 0.455 | 0.769 | 0.715 | 0.633 | 0.553 | 0.623 |
| hyb, MMseqs2 + Foldseek, 3 seeds | 0.913 | 0.637 | 0.452 | 0.782 | 0.733 | 0.651 | 0.573 | 0.638 |
| hyb, MMseqs2 + Foldseek-AFDB, 3 seeds | 0.914 | 0.638 | 0.438 | 0.786 | 0.742 | 0.653 | 0.575 | **0.639** |
| hyb, MMseqs2 + Foldseek-AFDB, 1 seed (entry 16) | 0.914 | **0.640** | 0.425 | **0.787** | **0.751** | 0.653 | 0.574 | 0.638 |

- Seed noise: single heads move by up to 0.01 on the full test mean, and the hybrid by about 0.003
  (0.635 to 0.638). Per set, Price swings by 0.02 between seeds and U-multi by 0.01. Every
  difference between hybrids since entry 13 (0.633 to 0.642) is at or below this noise level.
- Averaging three seeds per member changes nothing measurable (id50 0.913–0.914; full 0.638–0.639).
  With six different feature heads, the ensemble already averages out seed variance.

### 18. Partial fine-tuning of ESMC-600M (`finetune_plm.py`)

Blocks 25–36 of ESMC-600M plus the final norm are trained; blocks 1–24 stay frozen. That is 198M
trainable parameters with the head. The head is warm-started from the frozen-feature
`esmc600_last` head, with the same pooling and standardization. Two epochs: AdamW, backbone lr
2e-5, head lr 1e-4, cosine schedule with 2% warm-up, length-bucketed batches of 12k tokens (6,868
steps per epoch), bf16 autocast, and activation checkpointing. About 70 minutes per epoch on one
RTX A6000 (no flash-attention).

| model (id50 protocol) | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESMC-600M last, frozen head (start point) | 0.897 | 0.571 | 0.379 | 0.729 | 0.713 | 0.581 | 0.483 | 0.576 |
| fine-tuned, after epoch 1 | 0.895 | 0.534 | 0.393 | 0.725 | 0.716 | 0.577 | 0.484 | 0.572 |
| **fine-tuned, after epoch 2** | **0.908** | 0.568 | 0.426 | 0.734 | 0.732 | 0.591 | 0.494 | **0.591** |
| ens6_6b + fine-tuned (no homology) | 0.905 | 0.612 | 0.461 | 0.768 | 0.719 | 0.635 | 0.555 | 0.624 |
| hyb ens6_6b + fine-tuned, MMseqs2 + Foldseek-AFDB | **0.915** | 0.636 | 0.452 | 0.779 | 0.751 | 0.653 | 0.577 | 0.641 |
| hyb ens6_6b with frozen head swapped for fine-tuned | 0.914 | 0.630 | 0.450 | 0.781 | 0.748 | 0.651 | 0.575 | 0.639 |

- Fine-tuning works once the learning rate has decayed. Mid-schedule (epoch 1) the model is
  slightly worse than its start point; at the end of epoch 2 it is the best single model on id50
  (0.908, versus 0.904 for the best concatenated head and 0.890 for any other single embedding). It
  also beats its frozen start point on 5 of 6 test columns.
- In the ensembles it gives the highest id50 so far (0.915 with the gate; 0.905 without homology)
  but only by +0.001, within seed noise (entry 17). Homology still covers most of what a better
  model adds.

Full protocol (fine-tuned from `full_esmc600_last`, same settings, 2.4 h):

| model (full protocol) | New | Price | U-all | U-multi | T-70 | T-50 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESMC-600M last, frozen head | 0.561 | 0.405 | 0.739 | 0.692 | 0.585 | 0.493 | 0.579 |
| ESMC-600M fine-tuned | 0.575 | 0.425 | 0.740 | 0.694 | 0.591 | 0.494 | 0.586 |
| ens6_6b, no homology (entry 17) | 0.613 | 0.437 | 0.772 | 0.728 | 0.632 | 0.552 | 0.622 |
| ens7_ft = ens6_6b + fine-tuned, no homology | 0.612 | 0.438 | 0.772 | 0.728 | 0.634 | 0.553 | 0.623 |
| hyb_ens6_6b_msaf (entry 16) | 0.640 | 0.425 | 0.787 | 0.751 | 0.653 | 0.574 | 0.638 |
| hyb_ens7_ft_msaf | **0.644** | 0.424 | 0.783 | 0.742 | 0.651 | 0.571 | 0.636 |

- On the full protocol the gain over the frozen start is smaller: +0.007 test mean, mostly New and
  Price. The fine-tuned model is still below the frozen SAE-max heads (0.604–0.606) and the
  ESM2-3B concatenation (0.619). Fine-tuning helps ESMC-600M's own final layer but does not beat
  better frozen features.
- Ensembles do not move (0.623 without homology, 0.636 with the gate). The fine-tuned head is not
  kept as an ensemble member. 

Longer schedule (`id50_ft_esmc600_e4`, 4 epochs, otherwise identical; 4.7 h):

| epoch (id50 protocol) | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.895 | 0.555 | 0.444 | 0.730 | 0.674 | 0.592 | 0.501 | 0.583 |
| 2 | 0.907 | 0.572 | 0.417 | 0.729 | 0.644 | 0.591 | 0.507 | 0.577 |
| 3 | 0.910 | 0.581 | 0.426 | 0.741 | 0.665 | 0.599 | 0.511 | 0.587 |
| **4** | **0.912** | 0.578 | 0.428 | 0.744 | 0.665 | 0.601 | 0.515 | 0.589 |
| 2-epoch run, final | 0.908 | 0.568 | 0.426 | 0.734 | 0.732 | 0.591 | 0.494 | 0.591 |

- With twice the schedule, id50 keeps rising (0.912, the best single model on id50) and T-70/T-50
  improve. U-multi falls and the test mean is flat (0.589 vs 0.591). In the hybrid it gives the same
  0.915 on id50 as the 2-epoch head. Its full-protocol retrain (about 5 h) is not run: on the id50
  protocol the test columns are unchanged and the hybrid is unchanged.

### 19. Where the remaining errors are (`scripts/stratify.py`)

The best model's per-enzyme F1 is split by each query's top-hit identity to training. Identity is
the maximum over MMseqs2 and Foldseek-AFDB, and "no hit" means neither tool finds a hit at e-value
1e-3. Each row compares five runs: the hybrid `hyb_ens6_6b_msaf`, its six-head ensemble without
homology (`ens6_6b`), the best single head (`c3` = ESM2-3B ⊕ ESMC-600M last ⊕ SAE-max), and each
homology tool alone. Tests use the full protocol, id50 the id50 protocol. Full output:
`features/logs/stratify.txt`.

| pooled test queries (3,394) | n | share | hybrid | models | c3 head | MMseqs2 | Foldseek-AFDB | share of hybrid error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no hit | 95 | 0.028 | 0.404 | 0.404 | 0.372 | 0 | 0 | 0.050 |
| < 30% | 608 | 0.179 | 0.483 | 0.465 | 0.450 | 0.381 | 0.440 | 0.277 |
| 30–50% | 1,519 | 0.448 | 0.612 | 0.591 | 0.601 | 0.588 | 0.594 | 0.518 |
| 50–70% | 707 | 0.208 | 0.847 | 0.824 | 0.827 | 0.845 | 0.847 | 0.095 |
| ≥ 70% | 465 | 0.137 | 0.854 | 0.846 | 0.835 | 0.854 | 0.847 | 0.060 |
| *id50 fold, 30–50%* | 9,319 | 0.894 | 0.939 | 0.936 | 0.940 | 0.926 | 0.927 | – |

The same hybrid is split by how many training examples the query's rarest true EC has
(`features/logs/stratify_freq.txt`):

| rarest true EC: training examples | id50: share | id50: F1 | tests: share | tests: F1 | tests: share of error |
|---|---:|---:|---:|---:|---:|
| 1–9 | 0.120 | 0.723 | 0.328 | 0.505 | 0.485 |
| 10–49 | 0.170 | 0.892 | 0.312 | 0.674 | 0.304 |
| 50–199 | 0.227 | 0.917 | 0.212 | 0.723 | 0.175 |
| ≥ 200 | 0.483 | 0.968 | 0.149 | 0.918 | 0.037 |

- **85% of the remaining test error is on queries below 50% identity**, or with no hit at all.
  Above 50% identity, homology alone is as good as the hybrid. Learned models can only add value in
  the remote-homology regime.
- **The headroom is in rare ECs.** A third of the test queries carry an EC with fewer than 10
  training examples, against 12% of id50. Those queries account for half of the test error. Common
  ECs (≥ 200 examples) are essentially solved (0.918).
- **Why id50 selection transfers poorly.** At the same 30–50% identity, id50 queries score 0.94 and
  test queries 0.61. The id50 fold is drawn from the training distribution, where common ECs
  dominate, so it rewards fitting common classes. The test sets are enriched in rare and recently
  annotated ECs. This explains why the stacker (entry 8) and the late hybrids win on id50 without
  winning on the tests. It also means a gain for rare classes may hardly show on id50.
- **Model vs homology, per bin.** Below 30% identity, the model ensemble beats both homology tools
  (0.465 vs 0.381 / 0.440), and the gate adds a little more (0.483). On Price the model alone is
  better than the hybrid (c3 head 0.464 vs 0.425): its homologs mislead.
- **New ≥ 70% identity is only 0.709.** Even close homologs often carry a different EC, which looks
  like annotation changes or EC refinements in newly annotated proteins. Neither models nor
  homology can recover these from sequence similarity.

### 20. Head architecture sweep on the c3 features (`train_feat.py --arch/--act`)

**Question.** Does a deeper or more elaborate head help, given the best feature set? The features
are c3 (ESM2-3B last ⊕ ESMC-600M last ⊕ ESMC-600M SAE-max). Each block is projected to 512-d and
LayerNorm'd before the head.

**Note on the baseline.** The baseline `mlp` head is already concat → GELU → dropout →
Linear(1536, 1024) → LayerNorm → GELU → dropout(0.3) → output. So LayerNorm, GELU and dropout were
in place from the start.

**New options.**
- `--arch resmlp`: pre-norm residual blocks, each LN → Linear(×2) → act → dropout → Linear → dropout
  plus a skip connection.
- `--arch blockattn`: the three projected blocks become tokens, with a learned block embedding and
  a [CLS] token. These go through a pre-norm Transformer encoder with 8 heads, and the CLS output
  feeds the classifier.
- `--act gelu|relu|silu`.

**Setup.** id50 protocol, 100 epochs, everything else at its default (lr 1e-3, wd 0.05, batch
512). Selection is on id50. The "test mean" column comes from the same id50-protocol models
(trained without the id50 fold) on the six tests. It is not the full-protocol number, and it is
shown only as a secondary check.

| head | params | seeds | id50 | test mean (id50 models) |
|---|---:|---:|---:|---:|
| **mlp, 1×1024, GELU, dropout 0.3 (baseline)** | 17.2M | 3 | 0.904 / 0.904 / 0.901 | 0.598 / 0.606 / 0.607 |
| mlp, dropout 0.1 | 17.2M | 1 | 0.901 | 0.597 |
| mlp, dropout 0.5 | 17.2M | 1 | 0.905 | 0.603 |
| mlp, 2 hidden layers | – | 1 | 0.901 | 0.604 |
| mlp, 3 hidden layers | – | 1 | 0.903 | 0.605 |
| mlp, SiLU | 17.2M | 1 | 0.902 | 0.610 |
| mlp, ReLU | 17.2M | 1 | 0.904 | 0.600 |
| resmlp, 2 blocks × 1024 | – | 2 | 0.904 / 0.898 | 0.611 / 0.599 |
| resmlp, 4 blocks × 1024 | 34.0M | 3 | 0.901 / 0.901 / 0.901 | 0.614 / 0.597 / 0.602 |
| resmlp, 4 blocks × 2048 | 91.3M | 1 | 0.902 | 0.604 |
| gated block fusion | 16.2M | 1 | 0.894 | 0.601 |
| block attention, 2 layers, FFN 2048 | – | 1 | 0.886 | 0.581 |
| block attention, 4 layers, FFN 2048 | 25.6M | 1 | 0.891 | 0.582 |

**Findings.**
- **Most heads land within seed noise of the baseline.** That covers depth (1–3 MLP layers;
  residual stacks of 2–4 blocks at up to 91M parameters), activation (GELU, SiLU, ReLU) and dropout
  (0.1–0.5).
  - On id50 these heads score 0.898–0.905, against a baseline seed range of 0.901–0.904.
  - On the tests they score 0.597–0.614, against a baseline seed range of 0.598–0.607.
  - resmlp-4 first scored 0.614 on the tests. Its two seed repeats scored 0.597 and 0.602, so that
    gain was noise.
- **Block attention is clearly worse:** about −0.015 on id50 and −0.02 on the tests. With only
  three tokens (one per pooled embedding) there is nothing for attention to work on. It just adds
  optimisation difficulty compared with concatenation.
- **The gated fusion is also below the baseline on id50** (0.894).
- **Why the head doesn't matter.** These are pooled, frozen embeddings, and the classes are mostly
  separable by a shallow head. The binding constraint is the rare EC classes and remote homologs
  found in entry 19, not head capacity. Adding capacity mainly helps fit the common classes, which
  are already solved.
- **No full-protocol retrain was run:** no head survived id50 selection.

### 21. Rebalancing for rare ECs (`train_feat.py --balance`, `scripts/rare_eval.py --adjust`)

**Motivation.** Entry 19 found that half of the test error is on queries whose EC has fewer than
10 training examples. This entry tests two standard long-tail fixes on the c3 head (id50 protocol,
100 epochs).

- **Post-hoc logit adjustment.** Add `tau * max(0, log(n0 / n_c))` to the logit of every class
  with fewer than `n0` training proteins (Menon et al. 2021, restricted to the tail). It costs
  nothing to train.
- **Loss reweighting** (`--balance beta`). Each training protein's BCE is weighted by
  `n^-beta` (mean 1), where `n` is the training count of its rarest EC.

`scripts/rare_eval.py` reports per-enzyme F1 by the training count of the query's rarest true EC.
"tests" pools the 3,394 queries of the six tests, and the counts come from the id50-protocol
training set.

| model (id50 protocol) | id50 | id50, 1–9 | tests, 1–9 (1,129) | tests, 10–49 | tests, 50–199 | tests, ≥ 200 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| c3 baseline (seed 1) | 0.904 | 0.641 | 0.458 | 0.653 | 0.718 | 0.907 | 0.606 |
| + adjust tau 0.5, n0 200 | 0.903 | 0.650 | 0.488 | 0.652 | 0.712 | 0.894 | 0.614 |
| + adjust tau 1, n0 50 | 0.902 | 0.653 | 0.487 | 0.647 | 0.700 | 0.891 | 0.610 |
| + adjust tau 2, n0 50 | 0.887 | 0.614 | 0.447 | 0.612 | 0.654 | 0.873 | 0.570 |
| balance beta 0.3 | 0.902 | 0.636 | 0.454 | 0.661 | 0.696 | 0.902 | 0.607 |
| balance beta 0.5 | 0.901 | 0.623 | 0.449 | 0.657 | 0.706 | 0.915 | 0.603 |
| balance beta 1.0 | 0.896 | 0.640 | 0.454 | 0.639 | 0.694 | 0.891 | 0.592 |

**Logit adjustment does not replicate.**
- On seed 1, adjustment with tau 0.5 and n0 200 lifts rare-EC test queries by +0.03, costs about
  0.01 on common ones, and gains +0.008 on the test mean. It costs −0.001 on id50.
- The same setting on three other c3-family models changes the test mean by −0.005 (seed 2),
  +0.005 (seed 0) and −0.002 (resmlp-4), so the overall change is noise.
- In effect the adjustment trades common-class precision for rare-class recall at roughly break-even.

**Loss reweighting does not help, not even the rare bin** (0.449–0.454 vs 0.458).
- Higher beta mostly hurts. Beta 1.0 costs 0.008 on id50 and 0.014 on the tests.
- So the rare-class gap is not a class-prior or bias problem that reweighting can fix. Rare ECs
  are ranked low because their few training proteins don't place remote test members near them in
  embedding space.

### 22. Prototype (class-mean) and cosine output layers (`train_feat.py --out proto|cosine`)

**Idea.** Score a protein against the average embedding of each EC instead of a learned weight
vector per EC. A class with one training protein then gets a prototype as good as that protein's
embedding, and there is no per-class bias encoding the prior.

**Design.**
- `proto`: the score is `s * cos(h - m, mu_c) + b`.
  - `mu_c` is the mean of the centered, L2-normalized hidden vectors of EC `c`'s training
    proteins. It is recomputed from the current network every 50 steps and before evaluation.
  - `s` and `b` are a single learned scale and bias shared by all classes.
  - Recomputing once per epoch diverged: the prototypes went stale, and from a random start every
    class mean points the same way. Refreshing every 50 steps and centering on the training mean
    fixed this.
- `cosine`: the same scoring, but with learned per-class vectors (a normalized classifier).
- Setup: c3 features, MLP body, id50 protocol, 100 epochs.

| output layer | id50 | calls per protein (id50) | tests, 1–9 | tests, ≥ 200 | test mean |
|---|---:|---:|---:|---:|---:|
| linear (baseline, seed 1) | 0.904 | 1.04 | 0.458 | 0.907 | 0.606 |
| cosine | 0.902 | 1.04 | 0.460 | 0.912 | 0.610 |
| proto, decision rule as is | 0.749 | 1.83 | – | – | 0.484 |
| proto, logits shifted by −8 | 0.818 | 1.09 | 0.489 | 0.780 | 0.529 |

**Findings.**
- **The cosine classifier equals the linear head**, within seed noise.
- **The prototype head over-calls and ranks worse.** Neighbouring ECs have similar means, so
  several pass the threshold. Even with a global shift that brings it to one call per protein, it
  reaches only 0.818 on id50.
- **It does help rare ECs a little** (tests 0.489 vs 0.458), as intended. But it collapses on
  common ones (0.780 vs 0.907).
- **Why common ECs fail.** Large EC classes are multi-modal: unrelated folds and convergent
  families share an EC number, so one mean vector represents them poorly. The kNN methods (entries
  2 and 11) sit at the other extreme, with one prototype per protein, and also lost to the linear
  head.
- **Blending doesn't rescue it.** Averaging the baseline's probabilities with the prototype
  head's (weight 0.3 or 0.5) gives 0.592–0.607 on the tests and ≤ 0.902 on id50. Averaging two
  seeds of the baseline gives 0.611–0.613, so the prototype head adds nothing an extra seed
  doesn't.

### 23. SaProt-650M-AF2 structure-aware embeddings (`extract_plm.py --model saprot-650m`)

**Model.** SaProt (Su et al. 2024) is ESM-2 650M retrained on structure-aware tokens. Each residue
is one of 21 amino acids × 21 Foldseek 3Di states (`#` = masked), giving 446 tokens with the
specials.
- Weights: `models/SaProt_650M_AF2.pt`, supplied by the user since huggingface.co is blocked here.
- Loading: fair-esm in `.venv`, with SaProt's vocabulary order (`<cls> <pad> <eos> <unk> <mask>`,
  then amino acid × 3Di).
- **Vocabulary check.** With 1/7 of the tokens masked in 200 proteins, the model recovers 61% of
  the amino acids and 77% of the 3Di states, against about 5% by chance. So the token order is
  right.
- **Input.** The 3Di comes from `features/plm/afdb/3di.fasta`: AlphaFold-DB where available (98.7%
  of training, 85–99% of the tests), ProstT5-predicted otherwise. Price is entirely
  ProstT5-predicted.
- **Features.** Mean-pooled final layer (33) and layer 24, 1,280-d. Extraction of 228,888
  proteins took 75 min on one A6000 in fp16.

**Deviation from SaProt's recipe.** SaProt masks 3Di to `#` wherever AF2 pLDDT is below 70. Our
Foldseek AFDB database has no pLDDT, and the per-residue confidence files return 403 from this
server. The AFDB API only gives whole-protein summaries, and the bulk Swiss-Prot archive (28.6 GB)
downloads at about 250 KB/s (about 32 h). So the 3Di here is unmasked.

**Results** (MLP head, 100 epochs; test columns are full protocol):

| features | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM2-650M last (same architecture, sequence only) | 0.879 | 0.595 | 0.414 | 0.738 | 0.682 | 0.594 | 0.505 | 0.588 |
| ProstT5 last | 0.890 | 0.602 | 0.399 | 0.733 | 0.651 | 0.590 | 0.497 | 0.579 |
| SaProt last (2 seeds on id50) | 0.867 / 0.871 | 0.598 | 0.362 | 0.729 | 0.661 | 0.580 | 0.486 | 0.569 |
| SaProt layer 24 | 0.875 | – | – | – | – | – | – | – |
| SaProt last, all 3Di masked (sequence-only control) | 0.776 | 0.554 | 0.223 | 0.645 | 0.522 | 0.473 | 0.370 | 0.464 |
| c3 (ESM2-3B ⊕ ESMC-600M ⊕ SAE-max) | 0.904 | 0.604 | 0.464 | 0.772 | 0.689 | 0.635 | 0.553 | 0.619 |
| c3 ⊕ SaProt last | 0.904 | 0.595 | 0.454 | 0.772 | 0.719 | 0.630 | 0.549 | 0.620 |

Pooled test queries by top-hit identity (full protocol, single heads):

| bin (n) | no hit (95) | < 30% (608) | 30–50% (1,519) | 50–70% (707) | ≥ 70% (465) |
|---|---:|---:|---:|---:|---:|
| ESM2-650M | 0.372 | 0.406 | 0.552 | 0.796 | 0.837 |
| ProstT5 | 0.277 | 0.371 | 0.558 | 0.805 | 0.835 |
| SaProt | 0.298 | 0.395 | 0.531 | 0.802 | 0.836 |
| SaProt, 3Di masked | 0.214 | 0.281 | 0.412 | 0.708 | 0.818 |

In the id50 hybrid (six heads plus MMseqs2/Foldseek-AFDB gate, chosen 0/1/0.9):

| ensemble | id50 | test mean (id50 models) |
|---|---:|---:|
| ens6_6b_msaf (current best) | 0.914 | 0.639 |
| + SaProt last (7 heads) | 0.914 | 0.638 |
| SaProt last in place of ESM2-650M | 0.914 | 0.640 |
| + SaProt layer 24 (7 heads) | 0.913 | 0.637 |

**Findings.**
- **SaProt relies on its structure tokens, but its pooled embedding is no better than ESM2-650M's.**
  - With 3Di masked, it drops to 0.776 on id50 and 0.464 on the tests.
  - With 3Di, it reaches 0.867–0.871 on id50, still about 0.01 below ESM2-650M (0.879), and 0.569
    vs 0.588 on the tests.
  - It does not help where structure should matter most. Below 30% identity it scores 0.395 vs
    0.406, and with no hit 0.298 vs 0.372.
- **Price shows the train/test 3Di mismatch.** Price has only ProstT5-predicted 3Di while training
  used AFDB 3Di, and SaProt drops to 0.362 there vs 0.414 for ESM2-650M. T-70 and T-50 also have
  13–15% predicted 3Di.
- **Nothing new is added.** Added to c3 or to the hybrid, SaProt ties: c3 0.620 vs 0.619; hybrid
  0.913–0.914 on id50. The structural signal is already covered by Foldseek-AFDB homology and the
  3B-scale sequence models.
- **Caveats.**
  - Missing pLDDT masking puts disordered-region 3Di, which is unreliable, into the input, so this
    is a lower bound on SaProt. The paper's EC gains also come from fine-tuning, not frozen
    embeddings.
  - A fair rerun needs per-residue pLDDT, from `swissprot_pdb_v6.tar` (B-factor column) or the
    AFDB per-protein confidence files. Then mask the AFDB 3Di below pLDDT 70, and use `#` for
    proteins without an AFDB model.

### 24. SaProt-650M-AF2 with pLDDT-masked 3Di (rerun of entry 23 with SaProt's recipe)

**Change from entry 23.** The user downloaded `swissprot_pdb_v6.tar` (AFDB Swiss-Prot, 28.6 GB).
`scripts/afdb_plddt.py` streams it without extracting and reads each model's CA B-factor column,
which AFDB fills with pLDDT. It does this for every protein whose AFDB 3Di we already use (same
exact-sequence rule as `afdb_3di.py`).
- **Coverage:** 225,786 proteins, all but one of the 225,787 that use AFDB 3Di.
- **Masking:** 3Di is replaced by `#` (SaProt's mask state) where pLDDT < 70. That is 5.9% of AFDB
  residues: about 5% of training residues and 9–11% in the test sets. Price is unaffected, since it
  has no AFDB models.
- **Output:** `features/plm/afdb/plddt.npz` (per-residue pLDDT) and
  `features/plm/afdb/3di_plddt70.fasta`.
- **Proteins without an AFDB model** keep their ProstT5-predicted 3Di, as in entry 23, so masking
  is the only change. Entry 23 suggested `#` for these instead. But fully masked input is a much
  weaker embedding (0.776 id50, entry 23), so that would only hurt Price.
- **Extraction:** `extract_plm.py --model saprot-650m --states features/plm/afdb/3di_plddt70.fasta`
  into `features/plm/saprot-650m-plddt70`. It ran split across both A6000s with the new `--shard i/n`
  option, in 35 min instead of 75. A 200-protein sharded vs single-GPU check agreed to 5e-4 (fp16).

**Results** (same MLP head and settings as entry 23; test columns are full protocol):

| features | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM2-650M last (sequence only) | 0.879 | 0.595 | 0.414 | 0.738 | 0.682 | 0.594 | 0.505 | 0.588 |
| SaProt last, unmasked (entry 23; 2 seeds on id50) | 0.867 / 0.871 | 0.598 | 0.362 | 0.729 | 0.661 | 0.580 | 0.486 | 0.569 |
| SaProt last, pLDDT-masked (2 seeds on id50) | 0.863 / 0.879 | 0.575 | 0.355 | 0.742 | 0.663 | 0.584 | 0.494 | 0.569 |
| SaProt layer 24, unmasked (entry 23) | 0.875 | – | – | – | – | – | – | – |
| SaProt layer 24, pLDDT-masked | 0.880 | – | – | – | – | – | – | – |
| c3 (ESM2-3B ⊕ ESMC-600M ⊕ SAE-max) | 0.904 | 0.604 | 0.464 | 0.772 | 0.689 | 0.635 | 0.553 | 0.619 |
| c3 ⊕ SaProt last, unmasked (entry 23) | 0.904 | 0.595 | 0.454 | 0.772 | 0.719 | 0.630 | 0.549 | 0.620 |
| c3 ⊕ SaProt last, pLDDT-masked | 0.906 | 0.595 | 0.459 | 0.771 | 0.693 | 0.629 | 0.551 | 0.616 |

Pooled test queries by top-hit identity (full protocol, single heads):

| bin (n) | no hit (95) | < 30% (608) | 30–50% (1,519) | 50–70% (707) | ≥ 70% (465) |
|---|---:|---:|---:|---:|---:|
| ESM2-650M | 0.372 | 0.406 | 0.552 | 0.796 | 0.837 |
| SaProt, unmasked | 0.298 | 0.395 | 0.531 | 0.802 | 0.836 |
| SaProt, pLDDT-masked | 0.330 | 0.387 | 0.538 | 0.805 | 0.832 |

Hybrid (homology gate fixed at the chosen 0/1/0.9):

| ensemble | id50 | test mean, id50 models | full-protocol test mean |
|---|---:|---:|---:|
| ens6_6b_msaf (current best) | 0.914 | 0.639 | 0.638 |
| + masked SaProt last (7 heads) | 0.914 | 0.641 | 0.637 (New 0.642, Price 0.428, U-all 0.784, U-multi 0.742, T-70 0.652, T-50 0.572) |
| masked SaProt last in place of ESM2-650M | 0.914 | 0.639 | – |

**Findings.**
- **pLDDT masking changes nothing measurable.**
  - Single head: full test mean 0.569 both ways. The two id50 seeds span 0.863–0.879 against
    0.867–0.871 unmasked, i.e. within seed noise.
  - Layer 24: 0.880 vs 0.875.
  - Identity bins: within ±0.03, and the largest bin moves only 0.007.
- **Still below ESM2-650M.** On sequence alone, ESM2-650M scores 0.588 on the test mean; SaProt
  scores 0.569 with or without masking.
- **No gain in combination.** Masked SaProt ties when added to c3 (0.616 vs 0.619) and to the
  hybrid (0.637 vs 0.638 full; 0.641 vs 0.639 on id50 models).
- **Why masking matters little here.** Only about 5% of training residues fall below pLDDT 70, and
  they are mostly in disordered termini. A mean-pooled embedding is dominated by the confident
  core, and the confident core's 3Di was already right.
- **Conclusion.** Frozen SaProt embeddings do not help EC prediction on this benchmark, with or
  without its pLDDT recipe. Structure already enters the best model through Foldseek-AFDB
  homology. What's left to try for SaProt is fine-tuning (the setting of the paper's EC results),
  not more input fixes.

### 25. Best single network + homology (`hyb_c3_msaf`)

The c3 head (ESM2-3B last ⊕ ESMC-600M last ⊕ ESMC-600M SAE-max, one MLP) is blended with MMseqs2 +
Foldseek-AFDB homology through the same identity gate as the best model. The gate is grid-searched
on id50 (lo 0–0.5, hi 0.6–1, amax 0.5–1). It picks lo 0, hi 1, amax 0.9, the same gate as
`hyb_ens6_6b_msaf`.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean | mean without U-multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c3 single head | 0.904 | 0.604 | 0.464 | 0.772 | 0.689 | 0.635 | 0.553 | 0.619 | 0.606 |
| MMseqs2 alone | – | 0.622 | 0.385 | 0.754 | 0.684 | 0.610 | 0.514 | 0.595 | 0.577 |
| Foldseek-AFDB alone | – | 0.630 | 0.409 | 0.761 | 0.685 | 0.624 | 0.534 | 0.607 | 0.592 |
| **hyb_c3_msaf** (c3 + MMseqs2 + Foldseek-AFDB) | **0.916** | 0.636 | **0.440** | 0.783 | 0.716 | 0.650 | 0.570 | 0.632 | **0.616** |
| hyb_ens6_6b_msaf (6 heads + same homology; current best) | 0.914 | **0.640** | 0.425 | **0.787** | **0.751** | **0.653** | **0.574** | **0.638** | **0.616** |

Test columns are full protocol (c3 head retrained on all of split100).

**Findings.**
- **Homology adds +0.013 to c3** (0.619 → 0.632). id50 rises from 0.904 to 0.916, the highest id50
  of any model, though only 0.002 above the six-head hybrid (seed noise about 0.003).
- **One head with homology matches six heads with homology, except on U-multi.**
  - The 0.006 gap in test mean comes almost entirely from U-multi: 77 proteins, each with several
    ECs (0.716 vs 0.751).
  - Without U-multi the five-set means are identical (0.616).
  - c3's hybrid is better on Price (0.440 vs 0.425) and 0.003–0.004 lower on the others.
- **Practical trade-off.** `hyb_c3_msaf` needs three embeddings (ESM2-3B, ESMC-600M, ESMC-600M
  SAE) and one head. The six-head ensemble also needs ESMC-6B and five more heads, and buys mainly
  better multi-EC recall. When multi-function enzymes are not the focus, `hyb_c3_msaf` is the
  simpler model at the same accuracy.

### 26. Best model on a single embedding

The question: which single feature source (one PLM layer, or one SAE feature set) works best, alone
and with homology. Two more seeds of ESM2-3B were trained so that the top single embeddings all
have 3 seeds. Each single embedding was also blended with MMseqs2 + Foldseek-AFDB, with the gate
grid-searched on id50 as in entry 25.

Single MLP heads, mean over 3 seeds (per-seed values in brackets):

| embedding | id50 | full test mean |
|---|---:|---:|
| **ESM2-3B last** | 0.888 (0.890 / 0.884 / 0.890) | **0.617** (0.615 / 0.613 / 0.624) |
| ESMC-600M SAE-max | **0.891** (0.893 / 0.890 / 0.891) | 0.605 (0.604 / 0.604 / 0.606) |
| ESMC-6B SAE-max | 0.890 (0.890 / 0.890 / 0.891) | 0.601 (0.606 / 0.604 / 0.594) |
| ESM2-650M last | 0.878 (0.879 / 0.876 / 0.878) | 0.591 (0.588 / 0.592 / 0.594) |
| ESMC-600M last | 0.894 (0.897 / 0.890 / 0.894) | 0.584 (0.579 / 0.585 / 0.589) |
| ESMC-600M fine-tuned (entry 18) | 0.908 | 0.586 |
| *c3 combination, for reference* | *0.904* | *0.619* |

With homology (MMseqs2 + Foldseek-AFDB, id50-chosen gate):

| model | gate (lo, hi, amax) | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM2-3B last + homology | 0, 0.8, 0.75 | 0.906 | 0.639 | 0.430 | 0.769 | 0.725 | 0.637 | 0.551 | 0.625 |
| ESMC-600M SAE-max + homology | 0, 1, 0.9 | 0.911 | 0.624 | 0.433 | 0.781 | 0.720 | 0.643 | 0.560 | 0.627 |
| *c3 + homology (entry 25)* | 0, 1, 0.9 | *0.916* | 0.636 | 0.440 | 0.783 | 0.716 | 0.650 | 0.570 | *0.632* |

**Findings.**
- **Best single embedding on the tests: ESM2-3B last.** Its 3-seed mean is 0.617, ahead of
  ESMC-600M SAE-max (0.605) and ESMC-6B SAE-max (0.601). That margin is larger than the seed
  spread, and it nearly matches the c3 combination (0.619).
  - On id50 it is slightly lower (0.888 vs 0.891). id50 under-represents remote homologs, and that
    is where ESM2-3B gains.
  - ESMC-600M last and fine-tuned ESMC-600M lead on id50 but are among the weakest on the tests.
- **With homology, ESM2-3B and ESMC-600M SAE-max tie** (0.625 vs 0.627 test mean). id50 would
  pick ESMC-600M SAE-max (0.911 vs 0.906). ESM2-3B keeps an edge on New; the SAE features do
  better on U-all, T-70 and T-50.
- **Combining embeddings adds little alone (+0.002 for c3) but more under homology (+0.005–0.007).**
  The best single-embedding hybrid (0.627) trails the best model (0.638) by 0.011.

### 27. Homology-only combinations, and every category tested against homology alone

**Why.** The best hybrid gates on both MMseqs2 and Foldseek-AFDB, so a fair homology-only baseline
should use both tools too. `scripts/homology_combo.py` combines two homology runs with no learned
model and no parameters:
- `best`: per query, the calls of the tool whose top hit has the higher identity;
- `union`: every EC that either tool transfers.

| model | id50 | New | Price | U-all | U-multi | T-70 | T-50 | test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMseqs2 (entry 3) | 0.893 | 0.622 | 0.385 | 0.754 | 0.684 | 0.610 | 0.514 | 0.595 |
| Foldseek-AFDB (entry 16) | 0.890 | 0.630 | 0.409 | 0.761 | 0.685 | 0.624 | 0.534 | 0.607 |
| **hom_best** (MMseqs2 + Foldseek-AFDB) | **0.897** | 0.633 | 0.386 | 0.760 | 0.696 | 0.620 | 0.528 | 0.604 |
| hom_union | 0.894 | 0.634 | 0.407 | 0.759 | 0.692 | 0.624 | 0.534 | 0.608 |

`hom_best` has the highest id50 of any homology-only method, so the protocol selects it as the
baseline. Homology alone reaches 0.595–0.608 on the test mean.

**Paired bootstrap against hom_best** (`scripts/category_bootstrap.py`, 2,000 resamples, seed-0 full
runs; test-mean difference with 95% interval):

| model | Δ test mean | 95% CI | sets with significant gain / loss |
|---|---:|---|---|
| ESM2-3B last | +0.011 | [−0.004, 0.027] | 0 / 0 |
| ESMC-600M SAE-max | +0.000 | [−0.015, 0.015] | 0 / 1 (New) |
| ProstT5 | −0.025 | [−0.041, −0.009] | 0 / 4 |
| SaProt, pLDDT-masked | −0.035 | [−0.053, −0.017] | 0 / 4 |
| ESMC-600M fine-tuned | −0.017 | [−0.033, −0.002] | 0 / 4 |
| c3 head | +0.015 | [0.001, 0.028] | 2 / 1 |
| ens6_6b (no homology) | +0.018 | [0.005, 0.031] | 1 / 0 |
| hyb_c3_msaf | +0.029 | [0.017, 0.040] | 4 / 0 |
| **hyb_ens6_6b_msaf** | **+0.034** | [0.025, 0.043] | 5 / 0 (all but New) |

**Where each source wins** (`scripts/identity_bins.py`, `scripts/rare_eval.py`; pooled tests,
seed-0 full runs):
- **By identity:** at ≥50% identity, homology alone (0.846–0.854) matches the hybrid and beats every
  PLM head. At 30–50%, homology, ESM2-3B and the ensemble are level (0.580–0.591). Below 30% and
  with no hit, PLM heads lead.
- **By EC frequency:** PLM heads lead on common ECs (≥200 training examples: 0.90–0.92 vs 0.84).
  Homology leads on rare ECs (1–9 examples: 0.497 vs 0.429–0.483). The hybrid is best or tied in
  every bin.

**Conclusion.** No single PLM embedding, SAE or structure-aware PLM beats sensitive homology alone.
Structure-aware and fine-tuned heads are significantly worse. PLM ensembles give a small gain; the
identity-gated hybrid a clear one. Data for the manuscript (`reports/manuscript/data/`) come from
`scripts/manuscript_data.py` and the three scripts above.

### 28. Published methods against homology alone, with significance and error breakdown

**What.** `scripts/bootstrap_ci.py` gained `--out` and `--dump-predictions`, so our runs can be compared
with the published models (released MIL-BC+/MIL-CL+, retrained ESM2-BC, EnzymeHunter) on the same 2,000
paired resamples:
- Results: `features/logs/bootstrap_published.{json,txt}`.
- Every model's calls: `features/predictions_published/`.
- Breakdown by identity and EC rarity: `scripts/bins_from_predictions.py`.

**Per-enzyme F1, sets significantly better / worse (95% CI):**

| A − B | EnzymeHunter | MIL-CL+ | MIL-BC+ (released) | ESM2-BC (retrained) |
|---|---|---|---|---|
| homology only (hom_best) | 4 / 0 | 5 / 0 | 4 / 0 | 4 / 0 |
| MMseqs2 | 4 / 0 | 5 / 0 | 3 / 0 | 2 / 0 |
| ESM2-3B head (seed 0) | 4 / 0 | 5 / 0 | 4 / 0 | 5 / 0 |
| six-head ensemble | 4 / 0 | 5 / 0 | 4 / 0 | 6 / 0 |
| best hybrid (hyb_ens6_6b_msaf) | 4 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |

- **Micro F1:** homology alone ties ESM2-BC on every set.
- **Support-weighted F1:** released MIL-BC+ (test mean 0.614) ranks above homology (0.602) and every
  single model of ours except the hybrid (0.620).

**Where the published methods lose** (pooled tests; per-enzyme F1):
- **Close homologues:** at ≥50% identity they score 0.74–0.82, against 0.846–0.854 for homology.
- **Rare ECs:** with 1–9 training examples, 0.40–0.46 against 0.497 for homology.
- **Common ECs, by method type:** with ≥200 training examples, the classifiers ESM2-BC (0.895) and
  MIL-BC+ (0.878) beat homology (0.838). The centroid methods do not: MIL-CL+ 0.785, EnzymeHunter 0.622.
  This matches the prototype-layer failure in entry 22.

### 29. CLEAN, EnzHier and CLEAN-Contact re-run under the same protocol

**What.** Three more published contrastive predictors were run from their released split100 models on the six test
sets. All three call ECs by max-separation over EC centroids. Drivers are in `scripts/baselines/` (see its README);
predictions are in `features/external/{clean,enzhier,cleancontact}_tests.csv`.
- **CLEAN:** released `split100.pth` with its precomputed training embeddings, on ESM-1b query embeddings.
- **EnzHier:** released `split100_triplet_withEC_7000.pth` with its released ESM-1b training embeddings. Our query
  embeddings match those to within 4e-6.
- **CLEAN-Contact:** released `addition` model, trained on split100_reduced (224,741 enzymes with an AFDB
  structure, 5,197 ECs), with the authors' training representations. Each query uses ESM2-3B (2,560-d) plus a
  ResNet-50 embedding (2,048-d) of its Cβ<8 Å contact map.
  - Structures come from the local AFDB Swiss-Prot v6 tar (1,332 of 1,528 proteins), because direct AFDB downloads
    return 403. The other 196 (121 RefSeq Price proteins and 75 Swiss-Prot entries missing from AFDB) use ESMFold v1.
  - ESMFold ran through the transformers port loaded with Meta's original weights, without openfold. A control
    protein reproduces its AFDB contacts at Jaccard 0.91. Mean pLDDT over the 196 was 0.79.
  - Against the authors' released New-392 representations, ours match: sequence part cos > 0.9999, structure
    part median cos 0.996 (0.990 for the ESMFold ones).
  - A first attempt with ESM-2-predicted contact maps instead of structures was unfair to the method (Price-149 at
    0.23) and was discarded.

**Faithfulness.** The support-weighted F1 of CLEAN and EnzHier matches the MIL-BC paper (Bioinformatics 2026,
btag804) to within 0.001 on all six sets. CLEAN-Contact matches it exactly on New, U-all and U-multi and to within
0.003 on Test-70 and Test-50. On Price-149 it scores 0.452 against the paper's 0.478 (different predicted structures).

**Per-enzyme F1** (paired bootstrap, `features/logs/bootstrap_published.{json,txt}`; earlier numbers unchanged):

| model | New | Price | U-all | U-multi | T-70 | T-50 | mean | calls |
|---|---|---|---|---|---|---|---|---|
| CLEAN | .535 | .424 | .679 | .580 | .542 | .458 | .536 | 1.23 |
| EnzHier | .522 | .417 | .635 | .533 | .500 | .409 | .503 | 1.25 |
| CLEAN-Contact | .619 | .422 | .688 | .595 | .583 | .499 | .568 | 1.20 |
| homology only | .633 | .386 | .760 | .696 | .620 | .528 | .604 | 1.39 |

Sets significantly better / significantly worse (95% CI):

| A − B | CLEAN | EnzHier | CLEAN-Contact |
|---|---|---|---|
| homology only | 5 / 0 | 5 / 0 | 3 / 0 |
| ESM2-3B head | 5 / 0 | 5 / 0 | 4 / 0 |
| best hybrid | 5 / 0 | 5 / 0 | 4 / 0 |

- **Hierarchical loss:** EnzHier is below CLEAN, significantly on 3 sets.
- **Structure and a larger PLM:** CLEAN-Contact is above CLEAN (+0.032), significantly on 3 sets, and is the best
  centroid method. It is still below a plain MLP head on the same ESM2-3B embedding (0.615).
- **Bins (pooled tests):**
  - At ≥50% identity, published methods now span 0.70–0.82.
  - On rare ECs (1–9 examples), 0.40–0.46.
  - With no hit, 0.28–0.41.
  - On ECs with ≥200 examples, all five centroid methods (0.62–0.79) fall below homology (0.838), while both
    classifiers are above it (0.88–0.90).

**Manuscript.** Updated the abstract, introduction, §2.7, Table 3, Fig. 2b, Fig. 5, the metric section,
Discussion, Limitations, Conclusions, Methods and Table S1. `scripts/manuscript_data.py` now takes the external
tools' scores from the bootstrap JSON and writes `data/bootstrap_published.json` itself.
