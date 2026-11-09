# Kronos × EWZ — project status

Updated 2026-11-09. The v5 pilot is trained and evaluated. Historical model weights, datasets,
numerical results, and explorer scenarios are preserved; explanatory reports were revised.

## Frozen direct predictor and forward evaluation

The direct frozen-representation ridge experiment is completed in
`results/direct_probe.json`. Terminal MAE was 0.01370856 versus 0.01374903 for
zero return. Its paired interval spans zero. The frozen deployed candidate is
`finetuned/direct_frozen_v2`, with safe NPZ coefficients/scaler and a JSON manifest
recording freeze time, training prior, fixed penalty, and input/code/model hashes.
A replay on the saved CUDA device reproduced the prior experiment's MAE exactly. Version 1 is an earlier
setup snapshot. Version 2 is the final artifact used by the evaluator.

`scripts/forward_ewz.py update` verifies hashes, validates supplied continuation
against seven historical overlap bars, excludes incomplete hourly bars, and writes
an atomic locked ledger. Forecasts retain their original issue time. Outcomes mature
after seven observed complete target bars. Timing flags identify forecasts whose
first target had already begun at issuance. They must be excluded when assessing
fully prospective performance. Revision checks reject changes to forecast context
or previously scored terminal outcomes.

Dukascopy collection was retried on **2026-11-05 19:12 UTC** and returned **HTTP 429**
before overlap verification. The candidate URL symbol and price divisor therefore remain
unverified. No continuation data was accepted, and no new forecast was issued from stale
October data. This is a historical collection record, saved in
`results/forward_collection_status.json`. The runtime forward ledger is generated locally
by `scripts/forward_ewz.py update`, is not shipped, and starts empty for a new freeze. The
collector accepts data only after overlap agreement, and performs bounded requests without
automatic retries.

Run from this directory:

```bash
.venv/bin/python scripts/collect_ewz_forward.py
# After collection status reports accepted=true
HF_HUB_OFFLINE=1 .venv/bin/python scripts/forward_ewz.py update --continuation data/forward/continuation.csv
```

A supplied CSV can also be used with `--continuation`. It must include matching
historical overlap and the original seven columns. The collector's candidate endpoint
will require correction if later overlap verification fails. Outcome matching uses
observed bars, so it requires a complete provider continuation. No future session
timestamps are synthesized. No automatic scheduling was configured.

**Verification** completed with 41 regression tests, exact frozen-model replay,
and two idempotent update runs. Prospective scores remain dependent on successful
data collection and later target-bar maturity.

## Latest finding

The corrected v5 recipe reduces path MAE **8.6% versus v3** on the same 244 retrospective
7-bar windows. The paired circular-block bootstrap interval for v5 minus v3 MAE is
**[−0.001402, −0.000304]**. It remains **6.4% worse than unchanged price** on path MAE.
Direction is **51.6%, exactly the always-down baseline**. This is an improvement over the
previous recipe, with no demonstrated predictive advantage over the simple baseline.

| Forecast | Path MAE | Terminal MAE | Direction | Raw P(up) AUC | Raw Brier |
|---|---:|---:|---:|---:|---:|
| Unchanged price | 0.008212 | 0.013749 | — | — | 0.250585 (training prior) |
| Original small, 128-bar context | 0.008928 | 0.014487 | 50.0% | 0.517 | 0.279045 |
| Legacy v3, 512-bar context | 0.009554 | 0.015140 | 51.2% | 0.514 | 0.277421 |
| **v5 frozen tokenizer, 128-bar context** | **0.008734** | **0.014171** | **51.6%** | **0.532** | **0.270196** |

The path-MAE difference versus unchanged price has a block-bootstrap 95% interval of
[+0.000233, +0.000839], so v5 is still worse under this metric. The terminal-MAE difference
interval includes zero. Raw nominal-90% pointwise coverage is **71.1%**, versus v3's 77.9%:
the new recipe did not improve every measure, and uncertainty remains under-dispersed.
No v5 probability or interval calibration was fitted in this pilot.

The new v5 and original-small runs use **T=1.0, top_p=0.9, seed 42, 64 paths**, fixed before
looking at test results. Legacy v3 retains its previous validation-selected settings. This is
a comparison of complete recipes; it cannot attribute improvement to any single change.
All comparisons use identical realized targets, checked in `scripts/compare_v5_pilot.py`.

## Training and checkpoint

- Model: original `pretrained/Kronos-small`; original `pretrained/Kronos-Tokenizer-base` frozen.
- Data: 17,120 hourly EWZ CFD BID bars, 2017-01-26 through 2026-10-30; no dataset edits.
- Lookback 128 bars, forecast 7 bars, max context 512; price normalization uses past bars only.
- Forecast-only token loss; s2 teacher forcing; deterministic, causal validation; token weighting.
- Learning rate 3e-6; batch 32; max 8 epochs; early-stopping patience 3; CUDA device 1 only.
- Completed 8 epochs in about **4.7 minutes**. Best checkpoint: **epoch 8**, validation token
  loss **2.519506**. Token loss is not a financial forecast metric and is not comparable to
  old checkpoints with different tokenizers, losses, or validation attention.
- Checkpoint: `finetuned/ewz_h1_small_v5_7bar_frozen/basemodel/best_model`.
- Metadata, resolved config, split provenance, and history are saved beside the checkpoint.

## Code corrections

1. Removed the extra training-window bar. Forecast loss now includes the first forecast bar
   and covers exactly `[lookback, lookback + predict)`.
2. Restored `loss_mask_predict_only=False` as the implicit training default; v3/v4's historical
   explicit configs were preserved. The v5 config explicitly opts into forecast-only loss.
3. Full-sequence dependency attention is causal in evaluation as well as training. One-query
   decoding still attends to all available history, with the existing RoPE position fix.
4. Tokenizer is frozen and in evaluation mode during predictor training. Teacher forcing
   gives deterministic conditional s2 validation likelihood rather than a sampled s1 token.
5. Added early stopping, correct accumulation/partial-group scaling and scheduler step counts,
   token-weighted validation, run overwrite guards, and training/checkpoint metadata.
6. Validation always scores only forecast labels, even with full-sequence training loss;
   training-period context is never scored as validation data.
7. Forward calls pass through the training wrapper; loss-head access unwraps it. The wrapper
   path has a regression test; an actual multi-GPU distributed run was not exercised.
8. The training launcher resolves relative config paths before changing directories.

The post-pilot review changes to validation masking and wrapped calls do not alter this
single-GPU v5 run: it already used forecast-only masks and an unwrapped model.

## Walk-forward baseline experiment

`scripts/walk_forward_baselines.py` runs five expanding chronological folds at 7 and 24 bars.
Features are past-only; labels must mature before each boundary; transforms fit training
only; penalties and shrinkage are chosen on validation only. Final coefficients refit on
train+validation with settings frozen. Later folds may train on earlier folds' matured test
blocks, as in ordinary expanding-window evaluation.

| Horizon | Test windows | Unchanged path MAE | Logistic path MAE | Ridge path MAE | Logistic direction |
|---|---:|---:|---:|---:|---:|
| 7 bars | 726 | 0.008333 | 0.008336 | 0.008350 | 50.3% |
| 24 bars | 212 | 0.015314 | 0.015307 | 0.015894 | 44.3% |

No clear improvement over unchanged price emerged. For 7-bar logistic terminal MAE, the
bootstrap difference interval includes zero. Trailing-volatility, EWMA, and validation-
residual bands are scored by terminal coverage, width, and proper interval score. Brier
skill uses a deployable historical prior, not the test base rate. Confidence intervals use
moving blocks to account partly for serial dependence.

These folds cover roughly December 2023 to October 2026, unlike the pilot's November 2025
to October 2026 test interval. Do not compare their aggregate MAE numbers directly.

## Verification and preservation

- **41 regression tests passed**: dataset boundaries, target masks, future perturbations,
  attention causality, one-query and padding behavior, training wrappers, accumulation,
  model selection, and no dependence of predictions on current-fold future test labels.
- Independent attention check: perturbing future keys changed earlier eval output before
  the fix; after the fix the earlier output difference was exactly zero in the same test.
- Both GPU evaluation runs completed: 244 windows × 64 paths each.
- SHA256 audit of 124 pre-existing data/model/result/scenario files: **123 unchanged**;
  the only changed file is the intentionally revised `results/kronos-ewz-report.html`.
- No deployment, push, or package installation. No training/evaluation job remains running.
- Cumulative vendored patch and metadata are in `patches/`, based on commit `67b630e`.

## Interpretation and next experiments

This remains retrospective research: the old test period has repeatedly informed design.
One asset, one sampling seed, a heuristic bootstrap block length, and unknown overlap with
Kronos pretraining limit the conclusions. The 64 paths are model draws, not 64 additional
market observations; Wilson intervals quantify Monte Carlo uncertainty, not market calibration.
A coverage improvement after interval scaling does not prove conditional volatility skill.
The drift experiment identifies the effect of that correction, not the complete cause of failure.

The next distinct model experiment is a **direct return/direction head on frozen Kronos
representations**, compared with the same head on ordinary lagged features. That would test
whether the representation is useful when trained for the actual downstream target. Related-
market inputs and independently verified event/gap data are another direction. Larger models
or more epochs alone are not established solutions. Freeze the protocol before collecting a
new confirmation period. The direct return-head experiment and freeze are now implemented as described above. Additional related-market inputs remain a proposed experiment.

The static explorer continues to replay the preserved historical v3/v4 scenarios; it does not
silently substitute v5 forecasts. Its case study now includes the measured v5 update.

## Artifact map and reproduction

- `results/v5_pilot_comparison.json`: paired pilot/reference/legacy scores and uncertainty.
- `results/eval_v5_test_7bar_raw.json` and `.windows.csv`: new checkpoint evaluation.
- `results/eval_v5_reference_zeroshot_test_7bar_raw.json`: matched original-model reference.
- `results/walk_forward_baselines_v5.json` and `_windows.csv`: five-fold baseline results.
- `configs/config_ewz_h1_small_v5_7bar_frozen.yaml`: bounded training recipe.
- `scripts/compare_v5_pilot.py`: reproducible identical-target comparisons.
- `scripts/run_v5_pilot_gpu1.sh`: optional calibration workflow; evaluation-only mode skips
  training. It uses the resolved `.local-configs/` copy created by `scripts/setup_local.py`.
  It saves rerun outputs under `results/local_v5_pilot/` by default. Set `V5_OUT_DIR`
  to select a different output directory.
- `logs/train_h1_small_v5_7bar_frozen.log` and `logs/eval_v5*.log`: run logs.

Run from this project directory:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/compare_v5_pilot.py --out /tmp/v5-pilot-comparison.json
.venv/bin/python scripts/walk_forward_baselines.py \
  --out /tmp/ewz-walk-forward.json --windows-csv /tmp/ewz-walk-forward.csv
```

To reproduce training, run `scripts/setup_local.py`, copy the resolved v5 YAML from
`.local-configs/` to a **new experiment name/output directory** rather than replacing this
checkpoint, then pass its absolute config path to `scripts/run_finetune.sh`. The bounded
launcher also uses the resolved local config by default:

```bash
scripts/run_v5_pilot_gpu1.sh --run
```

The optional full pilot runner performs a validation-only calibration grid; that grid was
not part of the fixed-sampling pilot reported above.
