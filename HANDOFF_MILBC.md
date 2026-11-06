# Handoff: why does the retrained MIL-BC score below the released one?

Written 2026-11-05 at the end of a Claude Code session on the WSL2 machine (RTX 3090). Read this
first; README.md's "Train" section documents the tools.

## The problem

`train.py --model milbc` with the paper's settings (80 epochs, AdamW, lr 2e-4, weight decay 0.01,
batch 512, BCE, `call_ec` decision rule) gives clearly lower F1 than the released
`checkpoints/milbc/model_v1.pth`. Everything else reproduces:

| F1                  | New   | Price | Uniprot-all | Uniprot-multi | Test-70 | Test-50 |
| ------------------- | ----- | ----- | ----------- | ------------- | ------- | ------- |
| released MIL-BC     | 0.518 | 0.509 | 0.696       | 0.611         | 0.553   | 0.499   |
| retrained MIL-BC    | 0.436 | 0.303 | 0.644       | 0.481         | 0.484   | 0.424   |
| released MIL-BC+    | 0.554 | 0.530 | 0.747       | 0.689         | 0.611   | 0.555   |
| retrained MIL-BC+   | 0.524 | 0.454 | 0.727       | 0.583         | 0.577   | 0.509   |
| paper ESM2-BC       | 0.504 | 0.468 | 0.712       | 0.589         | 0.564   | 0.481   |
| retrained ESM2-BC   | 0.500 | 0.394 | 0.724       | 0.595         | 0.572   | 0.502   |
| released MIL-CL+    | 0.541 | 0.486 | 0.698       | 0.511         | 0.570   | 0.511   |
| retrained MIL-CL+   | 0.528 | 0.442 | 0.704       | 0.544         | 0.562   | 0.490   |

The gap is largest for the model that relies only on the domain tokens (MIL-BC), smaller with ESM2
added (MIL-BC+), and absent for ESM2 alone (ESM2-BC).

## Ruled out

- **Features, labels, metric, decision rule.** The released checkpoints scored on the rebuilt test
  features reproduce the paper's tables to ±0.001, and the released MIL-BC gets 0.998 top-1 on our
  split100 training features (`features/debug/trainloss.py`), so the training features and the
  label order (`checkpoints/classes.npy`) are compatible with it.
- **Train/test feature mismatch.** Token counts and value distributions of training and test
  domain tokens look alike (about 3 vs 3.6 tokens per protein; |x| mean 0.49, std 0.22 in both).
- **Seed noise.** Seeds 0, 1, 2 give Test-70 F1 0.484, 0.501, 0.481 (`runs/variants/milbc_seed*`).
- **Too few epochs.** 200 epochs: Test-70 ~0.50, flat after epoch 100 (`runs/variants/milbc_e200`).
- **The ~15-epoch plateau at the start.** All 5,242 sigmoids start at 0.5; initialising the output
  bias to -log(5242) (`features/debug/prior_bias.py`) removes the plateau but converges to the same
  Test-70 0.509 (`runs/variants/milbc_prior`).
- **Underfitting.** Retrained training BCE 1.57e-5, top-1 0.993 vs released 1.69e-5, 0.998.
- **Architecture.** State-dict keys and shapes are identical; `infer.py` loads both the same way.

## Best lead (not yet tested)

Weight norms differ systematically. Every Linear layer of the released model is about 1.6× larger
(e.g. `instance_encoder.0.weight` 29.4 vs 18.5, `attention_V.0.weight` 26.4 vs 15.3,
`classifier.8.weight` 207.8 vs 128.4); the LayerNorm gains and biases are similar. Most of those
Linear layers feed a LayerNorm, so their scale doesn't change the function, but it records how
much the optimiser moved them: larger means more or bigger updates. That points to a higher
learning rate, a smaller batch (more steps per epoch), weaker weight decay, or a learning-rate
schedule. Suggested sweep, one ~5 min run each:

1. `--lr 5e-4` and `--lr 1e-3` (at 2e-3 an early smoke test sat on the plateau a long time).
2. `--batch-size 128` (4× the steps).
3. `--weight-decay 0` vs 0.01 (does weight decay actually bite here?).
4. Then compare the resulting weight norms with the released ones as well as F1.

Other ideas: dropout off/on during the released training; fp32 vs fp16 storage of the domain
tokens (the `.compact.npz` caches hold fp16; `--no-cache` trains on float64 archives but needs the
28 GB originals); check whether the upstream repo (github.com/omics-ai/enzyme-mil) has published
training code since.

**Don't select epochs or hyperparameters on the test sets** — use `--num-folds 5 --fold 0` for a
held-out validation split, and report test numbers only at the end.

## Running things

Commands run from `code/`. On WSL2 always `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
first, or the GPU fails with a spurious "CUDA error: out of memory".

```bash
F=../features
EVAL=""; for n in new price uniprot-all uniprot-multi test70 test50; do
  EVAL="$EVAL --eval-set $F/tests/$n.npz $F/tests/$n.csv"; done
python train.py --model milbc --npzdata $F/shards/split100_[0-9][0-9].npz \
    --csvdata $F/split100_min5.csv $EVAL --eval-every 10 --lr 5e-4 \
    --output-dir ../runs/variants/milbc_lr5e-4
```

`train.py` reads the shards through their `.compact.npz` caches; when only a cache is present it
is used directly, so the 28 GB `split100_NN.npz` originals are not needed. Data loading takes
about 1 min; an 80-epoch MIL-BC run takes 3–5 min on a 3090 (~7 GB VRAM, ~28 GB RAM).
`python ../features/score.py label=milbc:<checkpoint>` re-scores a checkpoint through `infer.py`.

## Files

- `features/shards/split100_NN.compact.npz` (16): training features, 227,360 enzymes.
- `features/split100_min5.csv`: their labels (split100 without sequences shorter than 5 aa).
- `features/tests/<name>.{npz,compact.npz,csv}`: the six test sets.
- `features/logs/train_*.log`, `features/logs/variant_*.log`: every run's epoch-by-epoch F1.
- `runs/<model>/`, `runs/variants/*/`: checkpoints, `config.json`, `history.json`.
- `features/debug/`: `trainloss.py` (training-set BCE/top-1 of released vs retrained) and
  `prior_bias.py` (train.py wrapper with the prior-initialised output bias).

---

# Session 2 — 2026-11-05, Linux box (2× RTX A6000, `CUDA_VISIBLE_DEVICES=1`)

Environment: `uv sync` (pyproject.toml now holds requirements.txt; torch 2.15 + CUDA 13.2). No
`PYTORCH_CUDA_ALLOC_CONF` workaround needed here. Only the `.compact.npz` caches are present, so
pass shard paths explicitly (`features/sweep_milbc.sh` / `full_milbc.sh` do this). Upstream
github.com/omics-ai/enzyme-mil still has no training code or hyperparameters (checked today).

## Newly ruled out

- **Features, again, against the paper.** Training domain counts are 50,651 / 61,663 / 115,046
  for 1 / 2 / >2 domains vs the paper's Table 2 50,653 / 61,661 / 115,046.
- **Optimiser (the "best lead" above).** On fold 0 (80 % train), every variant ends at fold-0 F1
  0.946–0.949 and Test-70 0.466–0.471: base, `--lr 5e-4`, `--lr 1e-3`, `--batch-size 128`,
  `--weight-decay 0` (`features/logs/sweep_f0_*.log`, `runs/sweep/`). Higher lr / more steps only
  shorten the initial plateau. The 1.6× weight-norm difference is a symptom, not the cause.
- **Softmax CE instead of BCE.** (The paper writes the head as `softmax` with loss `-Σ Y log Ŷ`.)
  Released logits are BCE-shaped (negatives ≈ -27…-30, MIL-BC+ output bias drifted -0.021 like ours).
- **Different training labels.** Released MIL-BC and MIL-BC+ don't share "wrong" training
  predictions any more than released/retrained pairs do (`features/debug/agree.py`).

## Found: the released models were trained WITHOUT the padding mask

`features/debug/maskprobe.py` scores with and without the attention mask:

| Test-70 F1          | mask  | no mask |
| ------------------- | ----- | ------- |
| released MIL-BC     | 0.553 | 0.551   |
| retrained MIL-BC    | 0.484 | 0.448   |
| released MIL-BC+    | 0.611 | 0.607   |
| retrained MIL-BC+   | 0.577 | 0.391   |
| no-mask-trained BC+ | 0.568 | 0.568   |

The released models give padding slots low attention scores (-7.5 vs -4.2 for real domains;
-14.1 vs -9.3 for BC+). They can only learn that if padding was in the softmax during training.
`train.py --train-mask off` (new flag, training forward pass only; evaluation still masks)
reproduces that invariance exactly. **But it does not close the gap:** fold 0 0.947 (vs 0.948),
full-data MIL-BC Test-70 0.476, MIL-BC+ 0.568 (`runs/variants/milbc{,_plus}_nomask`). Use it anyway
for fidelity.

## Observations worth chasing next

- **Fold-0 F1 can't tell the variants apart** (all 0.946–0.949). The gap only shows on the
  out-of-distribution test sets, so a validation proxy closer to them is needed, e.g. a
  sequence-identity-clustered held-out fold (MMseqs2 at 50 %, as Test-50 was built).
- **Rare-class fit** (`features/debug/freqdiag.py`, training top-1 by EC class size 1 / 2 / 3–5):
  released MIL-BC 0.94 / 0.95 / 0.97 vs retrained 0.76 / 0.85 / 0.92. Its output-row norms are also
  flatter across class sizes. Candidates: class-balanced / oversampled training, or many more
  epochs at the end. Inconsistency: released MIL-BC+ fits rare classes *worse* than the retrained one
  (0.83 vs 0.90) yet generalises better, so the two released models may not share one recipe.
- The gap is largest on Price-149 (rare ECs) for every retrained model.

Diagnostics in `features/debug/` (`maskprobe.py`, `freqdiag.py`, `agree.py`, `logitstats.py`) use
relative paths: run them from `code/`.

## Identity-clustered validation fold (`features/folds/id50_*`)

`features/make_cluster_fold.py` (MMseqs2 18.8cc5c, installed at `~/.local/opt/mmseqs2`):
easy-cluster split100 at 50 % identity, `-c 0.8` → 32,291 clusters; whole clusters go to
validation in random order (seed 0) until 10 %, skipping clusters whose removal would leave an EC
untrainable; then `easy-search` validation vs training (≥50 % id, ≥50 % coverage, `-s 7.5`) and
drop every validation enzyme with a hit. 12,380 of 22,802 picked enzymes leaked past clustering and
are excluded from both sides. Result: 10,422 validation enzymes, 1,109 ECs, 399 multi-label;
204,558 training enzymes. Use `train.py --val-entries features/folds/id50_val.txt
--exclude-entries features/folds/id50_exclude.txt` (or `features/id50_train.sh <name> [args]`).

Proxy check, paper settings, epoch 80 (`runs/id50/`, `features/logs/id50_*.log`):

| model             | id50 F1 | fold-0 F1 | Test-70 F1 (same run) |
| ----------------- | ------- | --------- | --------------------- |
| MIL-BC            | 0.824   | 0.948     | 0.485                 |
| MIL-BC, no mask   | 0.809   | 0.947     | 0.474                 |
| MIL-BC+           | 0.867   | 0.967     | 0.554                 |
| ESM2-BC           | 0.880   | —         | 0.569                 |

It orders the models as the test sets do, with ~2.5× the spread of fold 0, so use it for selection
from now on. Caveats: still much easier than the test sets (all labels learnable by construction,
skewed to common ECs: 209 val enzymes from singleton classes); epoch-to-epoch noise is ~±0.01;
seed noise not yet measured. Note ESM2-BC ≥ MIL-BC+ here and on the tests, whereas the paper has
MIL-BC+ well ahead — the same reproduction gap.

## id50 sweep — seeds, sampling, long training (`runs/id50/`, `features/logs/id50_*.log`)

MIL-BC, paper settings unless noted, epoch 80:

| variant                                 | id50 F1                  | Test-70             |
| --------------------------------------- | ------------------------ | ------------------- |
| baseline seeds 0 / 1 / 2                | 0.824 / 0.846 / 0.832    | 0.485 / 0.496 / 0.477 |
| `--sampling sqrt` (EC count^-0.5)       | 0.832                    | 0.495               |
| `--sampling balanced` (EC count^-1)     | 0.799                    | 0.448               |
| `--train-mask off --epochs 200`         | 0.832 (flat after ~100)  | 0.486               |

Seed noise ≈ ±0.011 on id50: treat differences under ~0.02 as noise. Class-balanced sampling and
longer training are ruled out; sqrt sampling is within noise.

## Initialisation (`--init`, new flag) and a test-leakage probe

Released hidden-layer norms match **Kaiming-normal** init almost exactly: an id50 MIL-BC run with
`--init kaiming_normal` ends at 32.4 / 26.1 / 26.1 / 30.1 / 25.8 vs released 29.4 / 26.4 / 24.9 /
27.5 / 23.3 (default init: 17.9 / 14.9 / …). So the released models were very likely
Kaiming-normal initialised (and, from above, trained without the padding mask). Neither changes F1:

| id50 F1, epoch 80               | MIL-BC | MIL-BC+ |
| ------------------------------- | ------ | ------- |
| default init (seed mean)        | 0.834  | 0.867   |
| `--init kaiming_normal`         | 0.829  | 0.865   |
| `--init xavier_normal`          | 0.837  |         |
| kaiming_normal + mask off       | 0.840  |         |

The 1.6× weight-norm "lead" is fully explained by init and is not the cause of the gap.

Leakage probe (`features/debug/memprobe.py`): share of test enzymes predicted correctly with a
train-like top logit (>10). Released MIL-BC+ is *not* more train-confident than retrained MIL-BC+
(e.g. Uniprot-all 18.2 % vs 25.2 %), so no sign the released models saw test enzymes. Released
MIL-BC does sit at higher logits overall (mean top logit on Price 3.5 vs ~1.1 for every retrained
model); `call_ec`'s multi-label threshold depends on logit level, so calibration is worth a look.

### Status
Ruled out so far: features, labels, metric, optimiser (lr / batch / wd / epochs), class-balanced
sampling, padding mask, initialisation, test leakage. Untested ideas: instance-level augmentation
(random domain dropout from the bag), label smoothing, SWA/EMA or checkpoint averaging, input
normalisation of the DCT tokens, and decomposing the F1 gap into top-1 accuracy vs the multi-label
threshold (calibration).

## Step 1 result: the gap is mostly calibration × the metric (`features/debug/{gapsplit,threshold}.py`)

Ranking vs calling, Test-70 (all sets in the script output):

| model             | top-1 acc | F1 oracle-k | F1 call_ec | calls / enzyme |
| ----------------- | --------- | ----------- | ---------- | -------------- |
| released MIL-BC   | 0.528     | 0.524       | 0.553      | 1.27           |
| retrained MIL-BC  | 0.500     | 0.494       | 0.484      | 1.11           |
| released MIL-BC+  | 0.588     | 0.571       | 0.611      | 1.40           |
| retrained MIL-BC+ | 0.591     | 0.576       | 0.577      | 1.09           |

- MIL-BC+ ranks identically; the whole gap is in how many ECs `call_ec` emits.
- CLEAN's support-weighted F1 gives zero weight to ECs absent from a test set's truth, so a wrong
  call on such an EC is free. 39–51 % of extra (non-top-1) calls are free this way, and F1 rises
  monotonically as the logit threshold drops, for every model.
- Retrained MIL-BC+ at threshold −2 (1.38 calls/enzyme ≈ released 1.40): 0.561 / 0.493 / 0.745 /
  0.641 / 0.609 / 0.538 vs released 0.554 / 0.530 / 0.747 / 0.689 / 0.611 / 0.555. Retrained ESM2-BC
  at −2 beats the paper's ESM2-BC on every set. (Thresholds moved on test sets as a diagnostic only.)
- MIL-BC keeps a real ranking gap of ~0.03 (top-1 0.528 vs 0.500, oracle-k 0.524 vs 0.494).

Open: why released logits sit higher (earlier checkpoint? loss detail?); pick thresholds on the
id50 fold for a clean number; re-score everything with a metric that charges false positives
(micro / per-sample F1) to check the paper's ordering; the residual MIL-BC ranking gap.

## Why do released logits sit higher? (`features/debug/calib.py`, Test-70)

Released MIL-BC / MIL-BC+: 1.27 / 1.40 calls per enzyme, 0.80 / 0.85 of enzymes with any logit ≥ 0,
2nd-best logit −2.9 / −2.0, top logit 4.5 / 4.2. **All 30+ retrained checkpoints** sit at
1.05–1.14 calls, 0.56–0.72 with a call, 2nd logit −2.8…−5.3, top logit 1.2–3.8 — regardless of
lr, batch, wd, epochs, sampling, mask, seed, and the knobs below:

| id50 MIL-BC run (Test-70 calibration) | calls | any≥0 | 2nd | top | id50 F1 |
| ------------------------------------- | ----- | ----- | ----- | ---- | ------- |
| `--init kaiming_normal`               | 1.11  | 0.66  | −3.57 | 2.62 | 0.829   |
| `--bce-reduction sum` (paper's Σ_c)   | 1.12  | 0.67  | −3.49 | 2.62 | 0.830   |
| `--adam-eps 1e-12`                    | 1.12  | 0.70  | −3.56 | 3.03 | 0.836   |
| sum + kaiming + mask off              | 1.08  | 0.64  | −4.52 | 2.28 | 0.838   |
| `--dropout 0`                         | 1.05  | 0.56  | −4.68 | 1.21 | 0.815   |
| `--dropout 0.1`                       | 1.09  | 0.65  | −3.75 | 2.35 | 0.824   |
| `--dropout 0.4`                       | 1.14  | 0.71  | −2.82 | 2.32 | 0.822   |

MIL-BC+: `--bce-reduction sum` 1.08 calls (id50 F1 0.871), `--dropout 0` 0.870. So initialisation,
loss scale / Adam eps, and dropout are ruled out as the source of the calibration difference.
Dropout raises the 2nd logit monotonically (0.4 ≈ released) but not the top logit, and F1 does not
improve. New flags: `--bce-reduction`, `--adam-eps`, `--dropout`.

Still untested: label smoothing / soft targets (would *raise* negatives, but released train-set
negatives are *lower*, −30 vs −24, so unlikely); SWA / EMA / checkpoint averaging; per-sample
domain dropout augmentation; a larger or different training set than split100 (vocabulary is
split100's 5,242 ECs, but extra sequences would raise OOD confidence).

## Logit shift for the retrained models

`infer.call_ec(logits, shift)` calls ECs with `sigmoid(logit + shift) >= 0.5` (fallback to top-1
and cap of 5 unchanged; `shift=0` is the paper's rule). `infer.py --logit-shift S`, or a
`calibration.json` (`{"logit_shift": S}`) beside the checkpoint, applies it; released checkpoints
have no sidecar and are unchanged. `train.py --select-shift` (needs `--val-entries`/`--num-folds`)
and `code/calibrate.py` (for an existing checkpoint; `--apply-to DIR` copies the file) choose S
on the validation set only: max support-weighted F1 over 0…8 in steps of 0.25, ties to smaller.

Chosen on `runs/id50/<model>` and copied to `checkpoints/retrained/<model>/calibration.json`:
MIL-BC +3.5, MIL-BC+ +5.0, ESM2-BC +5.0 (id50 calls/enzyme 1.05 → 1.39–1.45). Full-data retrained
checkpoints through `features/score.py` (i.e. infer.py with the sidecar):

| F1                     | New   | Price | U-all | U-multi | Test-70 | Test-50 |
| ---------------------- | ----- | ----- | ----- | ------- | ------- | ------- |
| released MIL-BC        | 0.518 | 0.509 | 0.696 | 0.611   | 0.553   | 0.499   |
| retrained MIL-BC +3.5  | 0.512 | 0.533 | 0.708 | 0.670   | 0.564   | 0.514   |
| released MIL-BC+       | 0.554 | 0.530 | 0.747 | 0.689   | 0.611   | 0.555   |
| retrained MIL-BC+ +5   | 0.599 | 0.549 | 0.754 | 0.712   | 0.634   | 0.583   |
| retrained ESM2-BC +5   | 0.573 | 0.572 | 0.747 | 0.739   | 0.627   | 0.572   |

With a validation-chosen shift the retrained models match or beat the released ones. Note that a
shifted ESM2-BC is level with MIL-BC+ — the paper's MIL-vs-ESM2 margin is largely a calibration
effect under this metric (see "Step 1 result").

## Metrics that charge false positives (`features/compare_metrics.py` → `logs/compare_metrics.json`)

New in `train.py`: `micro_prf` (pooled over all enzyme–EC decisions) and `sample_prf` (per-enzyme
P/R/F1 averaged over enzymes, protein-centric as in CAFA). Fmax = best per-enzyme F1 over shifts
0…8 *on the test set* (calibration-free; ranks models, not a deployable number).

Per-enzyme F1 (shift 0 = paper's rule):

| model                      | New   | Price | U-all | U-multi | T-70  | T-50  |
| -------------------------- | ----- | ----- | ----- | ------- | ----- | ----- |
| released MIL-BC            | 0.515 | 0.396 | 0.676 | 0.632   | 0.531 | 0.451 |
| released MIL-BC+           | 0.560 | 0.397 | 0.713 | 0.684   | 0.578 | 0.500 |
| released MIL-CL+ (maxsep)  | 0.575 | 0.402 | 0.695 | 0.591   | 0.562 | 0.480 |
| retrained MIL-BC           | 0.480 | 0.266 | 0.645 | 0.523   | 0.493 | 0.411 |
| retrained MIL-BC+          | 0.570 | 0.400 | 0.727 | 0.625   | 0.583 | 0.495 |
| retrained ESM2-BC          | 0.585 | 0.367 | 0.734 | 0.667   | 0.590 | 0.502 |
| retrained MIL-CL+ (maxsep) | 0.579 | 0.390 | 0.707 | 0.602   | 0.566 | 0.474 |
| retrained MIL-BC+, +5      | 0.509 | 0.271 | 0.652 | 0.605   | 0.519 | 0.448 |

- Chosen on id50 by per-enzyme F1, the best shift is **0** for all three retrained models; the
  weighted-F1 shifts (+3.5/+5) give 2–3 calls/enzyme on the test sets (vs ~1.4 on id50) and lose
  0.05–0.12 per-enzyme F1. The `calibration.json` sidecars in `checkpoints/retrained/*` only help
  under the weighted metric.
- Under per-enzyme/micro F1 and Fmax, retrained MIL-BC+ ≈ released MIL-BC+ (except U-multi, 77
  enzymes), so MIL-BC+ is reproduced; retrained MIL-BC stays ~0.04 behind released MIL-BC.
- Retrained ESM2-BC is level with or ahead of both MIL-BC+ models on most sets (Fmax T-70: 0.590 vs
  0.585 / 0.578). The paper's MIL-over-sequence margin does not survive a metric that charges false
  positives. Test sets are small (Price 149, U-multi 77): differences < ~0.03 need bootstrap CIs.

## Sidecars removed; `--shift-criterion`; paired bootstrap CIs

- Deleted every `calibration.json` (checkpoints/retrained/{milbc,milbc_plus,esm2_bc}, runs/id50/*);
  infer.py is back on the paper's rule for all checkpoints. Selection logs remain in
  `features/logs/calibrate_*.log`.
- `train.py --select-shift` / `calibrate.py` take `--shift-criterion {sample,micro,weighted}`,
  default `sample` (per-enzyme F1). On id50 that picks shift 0 for all three retrained models.
- `features/bootstrap_ci.py` → `logs/bootstrap_ci.json`: paired bootstrap over enzymes (2,000
  resamples, seed 0), every model scored on the same draws; 95 % percentile CIs for each score and
  for paired differences. Note Test-70/Test-50 are drawn from the union of the other three sets,
  so the six sets are not independent.

Paired differences, per-enzyme F1 (* = 95 % CI excludes 0):

| A − B                         | New     | Price   | U-all   | U-multi | T-70    | T-50    |
| ----------------------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| retrained − released MIL-BC+  | +0.010  | +0.004  | +0.014* | −0.059* | +0.006  | −0.005  |
| retrained − released MIL-BC   | −0.035* | −0.129* | −0.031* | −0.109* | −0.039* | −0.039* |
| retrained − released MIL-CL+  | +0.004  | −0.012  | +0.012  | +0.011  | +0.003  | −0.006  |
| released MIL-BC+ − ESM2-BC    | −0.025  | +0.030  | −0.021* | +0.017  | −0.012  | −0.002  |
| retrained MIL-BC+ − ESM2-BC   | −0.015  | +0.034  | −0.007  | −0.042  | −0.007  | −0.007  |
| released MIL-BC − ESM2-BC     | −0.070* | +0.029  | −0.058* | −0.034  | −0.059* | −0.051* |
| released MIL-BC+ − MIL-BC     | +0.045* | +0.001  | +0.037* | +0.052  | +0.047* | +0.049* |

Under the weighted F1 the same comparisons flip: released MIL-BC+ beats ESM2-BC on all six
(Test-70 +0.039*) and beats retrained MIL-BC+ on all six. Conclusions:
1. MIL-BC+ and MIL-CL+ are reproduced (no consistent significant difference per-enzyme / micro).
   MIL-BC is the only real reproduction gap (released better on all six sets, every metric).
2. With a metric that charges false positives, no MIL model beats the ESM2-BC baseline anywhere;
   ESM2-BC beats domain-only MIL-BC on 4/6 and MIL-BC+ on U-all. The paper's MIL > ESM2 result
   depends on the support-weighted F1.
3. Adding ESM2 to the domain bag (MIL-BC → MIL-BC+) helps significantly on 4/6 sets.
