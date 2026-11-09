#!/usr/bin/env python3
"""How much of the v4-24 error is drift vs path shape?

Re-runs the exact v4/24 test-window inference (71 windows, 64 paths) and
decomposes the forecast error into:

  raw                 model paths as sampled
  common_drift_removed  remove the cross-path mean log drift per window
                        (keeps relative path shape and dispersion)
  per_path_detrended    remove each path's own linear log trend
                        (keeps curvature / path shape only)
  oracle_shift          raw paths shifted by the test-set mean error
                        (an upper bound, not a model result)

For each variant: direction accuracy, MAE, MAE vs random walk, AUC(p_up),
Brier skill vs base rate, and 90% interval coverage.

Only forward passes; no training. Uses GPU1 by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Kronos"))

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps
from kronos_paths import forecast_paths

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
CLOSE = FEATURES.index("close")
CLOSE_COL = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / ".local-configs/config_ewz_h1_small_v4_24_masked.yaml"))
    p.add_argument("--split", default="test")
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default=str(ROOT / "results/drift_neutral_v4.json"))
    p.add_argument("--paths-out", default=str(ROOT / "results/drift_neutral_v4_paths.npz"))
    p.add_argument("--reuse-paths", action="store_true")
    return p.parse_args()


def auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks = pd.Series(score).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_ci(values: np.ndarray, stat, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        out[i] = stat(values[idx])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize(tag: str, close_paths: np.ndarray, close0: np.ndarray,
              actual: np.ndarray, base_up: float) -> dict:
    """close_paths: (n_windows, n_paths, pred_len) absolute prices."""
    n, _, T = close_paths.shape
    rT = close_paths[:, :, -1] / close0[:, None] - 1.0
    med_path = np.median(close_paths, axis=1)
    med_r = np.median(rT, axis=1)
    mae = float(np.mean(np.abs(med_path - actual) / close0[:, None]))
    rw_mae = float(np.mean(np.abs(close0[:, None] - actual) / close0[:, None]))
    rw_worse = float(np.mean(
        np.abs(med_path - actual).mean(axis=1) / close0
        > np.abs(close0[:, None] - actual).mean(axis=1) / close0
    ))
    p_up = (rT > 0).mean(axis=1)
    y_up = (actual[:, -1] > close0).astype(int)
    p5 = np.percentile(close_paths[:, :, -1], 5, axis=1)
    p95 = np.percentile(close_paths[:, :, -1], 95, axis=1)
    coverage = float(np.mean((actual[:, -1] >= p5) & (actual[:, -1] <= p95)))
    brier = float(np.mean((p_up - y_up) ** 2))
    brier_base = float(np.mean((base_up - y_up) ** 2))
    direction = float(np.mean(np.sign(med_r) == np.sign(actual[:, -1] / close0 - 1)))
    return {
        "variant": tag,
        "n": int(n),
        "mean_terminal_r": float(np.mean(rT)),
        "mean_median_r": float(np.mean(med_r)),
        "dir_acc": direction,
        "dir_ci": boot_ci((np.sign(med_r) == np.sign(actual[:, -1] / close0 - 1)).astype(float), np.mean),
        "auc_p_up": auc(p_up, y_up),
        "auc_ci": boot_ci(np.arange(len(p_up)), lambda idx: auc(p_up[idx], y_up[idx])),
        "mae": mae,
        "rw_mae": rw_mae,
        "mae_ratio": mae / rw_mae,
        "frac_worse_than_rw": rw_worse,
        "beat_rw": 1.0 - rw_worse,
        "coverage_90": coverage,
        "brier": brier,
        "brier_skill_vs_base": 1.0 - brier / brier_base if brier_base > 0 else float("nan"),
        "p_up_mean": float(p_up.mean()),
        "p_up_std": float(p_up.std()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    df = pd.read_csv(d["data_path"])
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    model_dir = Path(cfg["model_paths"]["finetuned_tokenizer"]).parent.parent / "basemodel/best_model"
    # v4 config points at the v2 tokenizer; model itself lives under exp_name
    tok_dir = Path(cfg["model_paths"]["finetuned_tokenizer"])
    model_dir = Path(cfg["model_paths"]["base_path"]) / cfg["model_paths"]["exp_name"] / "basemodel/best_model"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = KronosTokenizer.from_pretrained(str(tok_dir)).to(device).eval()
    model = Kronos.from_pretrained(str(model_dir)).to(device).eval()
    print(f"model={model_dir}\ntokenizer={tok_dir}\ndevice={device}")

    lookback = int(d["lookback_window"])
    pred_len = args.pred_len
    n = len(df)
    val_end = int(n * (float(d["train_ratio"]) + float(d["val_ratio"])))
    split_end = n if args.split == "test" else val_end
    split_start = val_end
    anchors = list(range(max(split_start, lookback), split_end - pred_len + 1, pred_len))

    all_paths = []
    if args.reuse_paths and Path(args.paths_out).exists():
        z = np.load(args.paths_out)
        paths = z["paths"]
        close0 = z["close0"]
        actual = z["actual"]
        anchors = z["anchors"].tolist()
        print(f"reused {len(anchors)} windows from {args.paths_out}")
    else:
        for anchor in anchors:
            x_df = df.iloc[anchor - lookback:anchor][FEATURES].reset_index(drop=True)
            y_df = df.iloc[anchor:anchor + pred_len][FEATURES].reset_index(drop=True)
            x_ts = df["timestamps"].iloc[anchor - lookback:anchor].reset_index(drop=True)
            y_ts = df["timestamps"].iloc[anchor:anchor + pred_len].reset_index(drop=True)

            x = x_df.to_numpy(dtype=np.float32)
            mean, std = x.mean(axis=0), x.std(axis=0)
            x_norm = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)

            x_t = torch.from_numpy(x_norm[None]).to(device)
            xs_t = torch.from_numpy(calc_time_stamps(x_ts).to_numpy(dtype=np.float32)[None]).to(device)
            ys_t = torch.from_numpy(calc_time_stamps(y_ts).to_numpy(dtype=np.float32)[None]).to(device)

            paths_norm = forecast_paths(
                tokenizer, model, x_t, xs_t, ys_t,
                max_context=int(d["max_context"]), pred_len=pred_len,
                clip=float(d["clip"]), T=args.temperature, top_p=args.top_p,
                sample_count=args.samples, verbose=False,
            )
            paths = paths_norm * (std + 1e-5) + mean
            all_paths.append(paths[:, :, CLOSE].astype(np.float32))
            if (len(all_paths)) % 10 == 0:
                print(f"  {len(all_paths)}/{len(anchors)} windows", flush=True)

        paths = np.stack(all_paths)  # (W, S, T)
        close0 = np.array([df["close"].iloc[a - 1] for a in anchors], dtype=float)
        actual = np.stack([df["close"].iloc[a:a + pred_len].to_numpy(dtype=float) for a in anchors])

        np.savez_compressed(args.paths_out, paths=paths, close0=close0, actual=actual,
                            anchors=np.array(anchors), pred_len=pred_len)

    # ---- variants ----------------------------------------------------------
    W, S, T = paths.shape
    logp = np.log(paths / close0[:, None, None])
    t = np.arange(1, T + 1, dtype=float)

    # 1. raw
    raw = paths.copy()

    # 2. common drift removed: subtract cross-path mean terminal log return
    #    as a linear drift over the path (preserves all relative path shapes)
    mu = logp[:, :, -1].mean(axis=1)  # (W,)
    common = close0[:, None, None] * np.exp(logp - mu[:, None, None] * (t / t[-1])[None, None, :])

    # 3. per-path detrend: remove each path's own OLS slope in log space
    tbar = t.mean()
    denom = np.sum((t - tbar) ** 2)
    slopes = ((logp - logp.mean(axis=2, keepdims=True)) * (t - tbar)[None, None, :]).sum(axis=2) / denom
    lp_detr = logp - slopes[:, :, None] * t[None, None, :]
    detr = close0[:, None, None] * np.exp(lp_detr)

    # 4. oracle constant shift on the raw median terminal return
    y_term = actual[:, -1] / close0 - 1.0
    med_term = np.median(paths[:, :, -1], axis=1) / close0 - 1.0
    err = y_term - med_term
    shift = float(np.mean(err))  # in-sample oracle
    oracle = paths * (1.0 + shift)

    base_up = float((actual[:, -1] > close0).mean())
    variants = {
        "raw": summarize("raw", raw, close0, actual, base_up),
        "common_drift_removed": summarize("common_drift_removed", common, close0, actual, base_up),
        "per_path_detrended": summarize("per_path_detrended", detr, close0, actual, base_up),
        "oracle_shift": summarize("oracle_shift", oracle, close0, actual, base_up),
    }

    out = {
        "config": str(args.config),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tok_dir),
        "split": args.split,
        "n_windows": int(W),
        "samples_per_window": int(S),
        "pred_len": pred_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "base_up": base_up,
        "test_drift": float(np.mean(y_term)),
        "mean_path_drift_log": float(np.mean(mu)),
        "drift_std_log": float(np.std(mu)),
        "oracle_shift": shift,
        "variants": variants,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    print(f"\nwrote {args.out}\nwrote {args.paths_out}")


if __name__ == "__main__":
    main()
