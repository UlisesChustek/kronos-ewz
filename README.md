# Kronos × EWZ
## Can a pretrained financial model improve forecasts for a Brazilian equity ETF?

[Interactive explorer](https://uliseschustek.github.io/kronos-ewz/) ·
[Research report](https://uliseschustek.github.io/kronos-ewz/case-study.html)

This project investigates whether Kronos can use historical EWZ prices and volume to
forecast what happens next. EWZ is an ETF tracking Brazilian equities. Kronos is a
pretrained transformer that processes financial price bars, with a design inspired by
language models.

The practical question is simple: **does the model make better forecasts than assuming
the price stays where it is?**

The project uses 17,120 hourly bars from January 2017 through October 2026. The source is
Dukascopy's EWZ CFD bid feed, including its feed-specific volume.

## What was built and tested

- A data preparation, fine-tuning, calibration, and evaluation pipeline.
- Several training recipes, including seven-bar and 24-bar forecast horizons.
- Corrections to training-target alignment and validation attention that could otherwise
  distort model selection.
- Comparisons against unchanged price, historical drift, and simple price/volume models.
- A second approach that freezes Kronos and trains a small linear model to predict the
  seven-bar return directly from its internal representation.
- An interactive forecast explorer, a research report, and a timestamped workflow for
  evaluating the frozen predictor on new data.

Seven hourly bars are approximately one trading session. Twenty-four bars span several
sessions. The evaluation accounts for the gaps between trading sessions.

## What the results show

Improving the training recipe reduced average forecast-path error **8.6% versus the
previous fine-tuned version**. The improved version still had **6.4% more error than
unchanged price**, and its **51.6% directional accuracy matched always predicting down**.

The direct predictor came closer to the simple baseline. These are errors at the end
of the same 244 seven-bar test windows, measured as fractions of the starting price.

| Forecast | Terminal mean absolute error ↓ |
|---|---:|
| Unchanged price | 0.013749 |
| Linear model using ordinary price/volume features | 0.013753 |
| Linear model using frozen Kronos representations | **0.013709** |
| Improved fine-tuned Kronos, v5 | 0.014171 |

The direct Kronos predictor's advantage over unchanged price was about **0.4 basis points**.
Its paired 95% uncertainty interval ranges from roughly 1.4 basis points better to
0.6 basis points worse. That interval includes zero.

**The experiments demonstrate that the revised recipe improved the previous model.
They have yet to establish a reliable forecasting advantage over unchanged price.**
The direct predictor's small observed advantage needs confirmation on new data.

Several recipe changes were applied together, so their individual contributions remain
unresolved. The test period was inspected repeatedly during development, making these
results retrospective. The conclusions apply to the tested EWZ data and settings.
Overlap with Kronos's original pretraining data is unknown.

## Current status

The direct predictor was frozen with its coefficients, scaler, and settings saved.
A local replay of the frozen predictor reproduced the measured result exactly.

The latest saved Dukascopy collection attempt returned HTTP 429, so new continuation data
has not passed source verification. That record is historical. The forward ledger is a
runtime artifact written locally by `scripts/forward_ewz.py update`; it is not shipped.
No forecast has been issued in this release, and a new freeze starts with an empty ledger.
Prospective evaluation is pending.

The project includes **41 passing regression tests**, covering data boundaries, future
information leakage, attention behavior, and forward-scoring rules.

## Explore the work

The explorer is a historical, precomputed artifact built from the v3/v4 scenarios. It is
not a live service and does not substitute v5 forecasts.

- [Interactive explorer](demo/index.html) — historical v3/v4 sampled forecasts.
- [Research report](demo/case-study.html) — experiment results and interpretation.
- [Detailed project status](PROJECT_STATUS.md) — methods, limitations, and next steps.
- [V5 comparison](results/v5_pilot_comparison.json) — error comparisons and uncertainty.
- [Direct-predictor experiment](results/direct_probe.json) — frozen-representation results.
- [Forward collection status](results/forward_collection_status.json) — historical data attempt.

## Reproduce the checks

Run from a clean checkout with Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_local.py
```

`setup_local.py` writes resolved, machine-specific copies of every config to
`.local-configs/`. Project shell and Python launchers use those files by default, so run
setup before training or evaluation. Use a CUDA-enabled PyTorch installation for GPU
experiments; the checked-in configs target CUDA device 1, which can be changed in the
generated config. The following checks can run on CPU:

```bash
.venv/bin/python -m unittest discover -s tests -v

# With the original EWZ CSV supplied, save a fresh copy of the comparison
.venv/bin/python scripts/compare_v5_pilot.py --out /tmp/v5-comparison.json

# Retry continuation collection, then inspect its acceptance status
.venv/bin/python scripts/collect_ewz_forward.py
```

Further training, inference, and continuation commands are documented in
[PROJECT_STATUS.md](PROJECT_STATUS.md). The bounded v5 pilot can be launched with:

```bash
# Requires the original CSV, downloaded model weights, and CUDA
scripts/run_v5_pilot_gpu1.sh --run
```

## Attribution

The pretrained models and base architecture come from
[ShiYu's Kronos project](https://github.com/shiyu-coder/Kronos).
The vendored Kronos code retains its upstream MIT license
([third_party/Kronos/LICENSE](third_party/Kronos/LICENSE)); this repository does not
include a separate project-level license. This project contributes the EWZ adaptation
experiments, evaluation, diagnostics, local corrections, and reporting. Vendored changes
are recorded in [patches](patches/README.md).

## Research inputs

The repository includes code, saved numerical results, and precomputed explorer scenarios.
Full source data and downloaded model weights are separate research inputs; they are not
included. Exact reproduction of the historical evaluations requires the original EWZ CSV.

Place a verified input CSV at data/processed/ewz_h1.csv with columns
timestamps, open, high, low, close, volume, amount. Timestamps use UTC.
The acquisition status and limitations are recorded in PROJECT_STATUS.md.

Download the original public models when needed:

```bash
.venv/bin/python scripts/setup_local.py --download-models
```

The setup command writes machine-specific configurations under .local-configs/.
Project launchers use those files by default. Pass an absolute path to one of them to
scripts/run_finetune.sh when overriding the default. Historical result JSONs describe the
original runs. Frozen predictor artifacts and their machine-specific manifests are
generated locally by scripts/forward_ewz.py freeze; the runtime ledger is generated by
scripts/forward_ewz.py update and is not committed.

The explorer runs without Python or a GPU. Open demo/index.html, or serve demo/ with
python3 -m http.server 8080 --directory demo.
