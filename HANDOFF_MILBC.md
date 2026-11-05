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
