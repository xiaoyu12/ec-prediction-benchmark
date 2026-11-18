# CLEAN, EnzHier and CLEAN-Contact re-runs

Drivers for re-running three published contrastive EC predictors on the six test sets (AUTORESEARCH entry 29).
They expect a work directory `~/workspace/baselines` holding the upstream repositories and downloads:

| path | source |
|---|---|
| `CLEAN/` + `CLEAN/app/data/pretrained/` | github.com/tttianhao/CLEAN; released weights (Google Drive, `pretrained.zip`) |
| `EnzHier/` | github.com/TangTZscut/EnzHier (model and ESM-1b training embeddings in the repo) |
| `CLEAN-Contact/` + `download/cleancontact/esm_data.zip` | github.com/PNNL-CompBio/CLEAN-Contact; training representations (Google Drive) |
| `download/esmfold/esmfold_3B_v1.pt` | dl.fbaipublicfiles.com/fair-esm/models/ |
| `.venv` | venv on the `EnzymeHunter` conda env (torch 2.7, fair-esm 2.0, torchvision, transformers 4.51) + gdown, omegaconf |

Steps, run from `~/workspace/baselines` with the scripts in this directory (they find the test CSVs in this repository's `dataset/`):

1. `esm1b_embed.py esm1b_data <six test CSVs>`: ESM-1b query embeddings (CLEAN format).
2. `run_clean_like.py clean preds/clean` and `run_clean_like.py enzhier preds/enzhier`.
3. `extract_afdb.py <swissprot_pdb_v6.tar> test_ids.txt pdbs`: AFDB structures (1,332 of 1,528 proteins).
4. `esmfold_predict.py fallback_ids.txt esmfold_pdbs`: ESMFold v1 (transformers port, original weights) for the other 196.
5. `cleancontact_embed.py test_ids.txt pdbs cc_query --esmfold esmfold_pdbs`; `check_cc_query.py` compares with the
   released New-392 representations.
6. `run_cleancontact.py cc_query preds/cleancontact`.
7. `merge_score.py preds/<tool> features/external/<tool>_tests.csv`, then `scripts/bootstrap_ci.py --external ...`.
