> **v5 update (2026-11-05):** the separately trained 7-bar pilot reduced path MAE 8.6%
> versus v3, but remains 6.4% worse than unchanged price; direction 51.6% equals always down.
> The scenario files in this explorer remain historical v3/v4 data. See
> [project status](../PROJECT_STATUS.md) and [case-study update](case-study.html#v5-update).

# Kronos × EWZ — static interactive demo

A public, backend-free explorer for a Kronos financial foundation model fine-tuned on
9.7 years of EWZ hourly data. Forecasts are **precomputed locally**; the browser only
replays them.

## Result (read this first)

This is a **diagnostic negative result**, not a predictor demo.

> **Retrospective warning.** The same EWZ test period was inspected repeatedly while the
> variants below were developed. Treat every test metric here as retrospective, not as fresh
> out-of-sample confirmation. Overlap with the original Kronos pretraining corpus is unknown.

- **No point edge.** v4 at 24 bars: MAE **0.0194** vs random walk **0.0180**; it beats the
  random walk in **45.1%** of windows.
- **No direction edge.** Accuracy is **45.1%** — exactly the always-down baseline — with
  retrospective `AUC(p_up) = 0.46`.
- **What changed:** fine-tuning moved EWZ's marginal scale. After validation-only calibration,
  90% intervals cover **92.4% of realized closes on average pointwise**; terminal-bar coverage
  is **91.6% calibrated** vs **64.8% raw**. That is interval calibration, not evidence that the
  model learned a conditional volatility mechanism.
- **Why it looks bearish:** a **−0.40% mean path drift** over 24 bars while the test period
  drifted **+0.16%**. Removing that drift moves Brier skill from −0.32 to −0.03, but leaves
  MAE 2.7% worse than a random walk and AUC at 0.43.
- **What the baselines add:** no 24-bar signal in momentum, reversal, logistic or ridge models.
  An exploratory 7-bar logistic fit reached AUC ≈ 0.59, but that number is retrospective and was
  obtained after repeated inspection of the same test period. The later expanding-window v5
  benchmark (`results/walk_forward_baselines_v5.json`) found no material gain: direction ≈
  50.3%, Brier skill ≈ +0.001, and no MAE improvement over unchanged at 7 bars.
- **Scope:** one asset, one 12-month test window, 71 non-overlapping 5-day windows. This is
  not a verdict on Kronos or financial foundation models.

Full diagnosis: [`case-study.html#why`](case-study.html#why). Leakage-safe baseline results:
[`../results/walk_forward_baselines_v5.json`](../results/walk_forward_baselines_v5.json).

## Preview locally

The page works from `file://`, but a tiny server is closer to production:

```bash
cd demo
python3 -m http.server 8080
# open http://localhost:8080
```

Or just open `demo/index.html` in a browser.

## What's in here

| Path | Purpose |
|---|---|
| `index.html` | Explorer page |
| `case-study.html` | Self-contained evaluation write-up (print → PDF) |
| `assets/style.css` | Dark responsive theme, no external fonts |
| `assets/app.js` | Model/date/context selectors, metrics, Plotly chart |
| `assets/plotly.min.js` | Vendored Plotly 2.35.2 (offline) |
| `data/demo.js` | `window.DEMO_DATA` — precomputed scenarios |
| `data/demo.json` | Same dataset as JSON, for inspection |

## Dataset

- **15 as-of dates** from Nov 2025 to Oct 30, 2026, including three mid-month snapshots.
- **3 models**: fine-tuned v4 (24-bar), fine-tuned v3 (7-bar), and zero-shot (24-bar).
- **45 scenarios total**, each with **64 sampled paths**, a 5–95% band, and terminal metrics.
- Latest live bar: **2026-10-30 19:00 UTC**; forecast timestamps are synthesized from the
  next weekday sessions.

## Regenerate the dataset

Inference runs on the local GPU and reuses the evaluation preprocessing:

```bash
cd ..
.venv/bin/python scripts/make_demo_data.py
```

That samples 64 paths for each `model × as-of date` pair and rewrites
`demo/data/demo.js` + `demo/data/demo.json`. The script uses `cuda:1` when available.

## Static hosting

`demo/` is a complete static site: no build step, server, or API keys. Serve it locally or
publish the folder with any static host. The repository's Pages workflow publishes `demo/`
from the `main` branch.

## Honest framing

Read the "Result" section above. In short: the fine-tuned model **does not beat a random walk**
on point error, its 45.1% directional accuracy equals always predicting down, and its apparent
bearish signal is a removable drift — not a forecast. Its calibrated intervals are useful as a
marginal uncertainty summary, but the 92.4% figure is average pointwise coverage under
retrospective evaluation, not a conditional guarantee. The demo is a probabilistic scenario and
diagnostic artifact, not a trading signal. Keep that framing and the retrospective warning if
you reuse the page.
