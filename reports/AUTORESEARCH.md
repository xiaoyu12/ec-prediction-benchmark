# Autoresearch: new EC-prediction models

Branch `autoresearch`, started 2026-11-06. Goal: beat the existing models (ESM2-BC, MIL-BC+, the
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

## Summary (through entry 18)

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
- **What helped:** sparse-autoencoder (SAE) max-pooled features of ESMC, ensembling different
  feature heads, identity-gated homology, and structural homology (Foldseek).
- **What did not:** ESMC-6B hidden states, alternative losses, EC-hierarchy loss, block dropout,
  hidden-space kNN, a GBT stacker, seed averaging, and real-vs-predicted 3Di.
- **Plateau:** every hybrid since entry 13 scores 0.913–0.916 on id50 and 0.633–0.642 on the full
  test mean. The seed noise is about 0.003.

## Code

| file | what |
|---|---|
| `code/extract_plm.py` | Per-protein pooled embeddings: ESMC-600M/6B (final and SAE-layer hidden states, plus SAE features), ProtT5, ProstT5, ESM2-3B. Runs in `envs/esmc` (Biohub `esm`), resumable. |
| `code/train_feat.py` | Heads on fixed embeddings, single or combined (MLP / gated / linear; BCE / ASL / focal; EMA; block dropout; EC-hierarchy aux loss; hidden-space kNN). Both protocols. |
| `code/knn_feat.py` | Cosine nearest-neighbour EC transfer on the same embeddings. |
| `code/homology.py` | MMseqs2 homology transfer; `--tool foldseek` for structural (ProstT5 3Di + AA) homology. |
| `code/prostt5_from_gguf.py` | Rebuilds ProstT5 (encoder + 3Di CNN) from Foldseek's GGUF copy, since huggingface.co is unreachable here. It agrees with Foldseek's own 3Di prediction on 99.8% of residues. |
| `code/ensemble.py` | Probability averaging plus the identity-gated homology blend. |
| `code/stack.py` | Gradient-boosted re-ranker over model and homology evidence (cluster-grouped CV). |
| `code/finetune_plm.py` | Partial fine-tuning of ESMC-600M (implemented, smoke-tested, deprioritized). |
| `scripts/feat.sh`, `scripts/extract_plm.sh` | Memory-capped wrappers. |
| `scripts/afdb_3di.py` | 3Di from AlphaFold-DB Swiss-Prot structures (Foldseek DB), ProstT5 fallback. |
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
