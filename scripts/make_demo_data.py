#!/usr/bin/env python3
"""Precompute the static EWZ forecast-explorer dataset.

Runs the three demo models over a grid of as-of dates, samples 64 paths per
scenario, and writes demo/data/demo.json + demo/data/demo.js.

Inference only. Reuses the same preprocessing as scripts/evaluate.py so the
demo is consistent with the measured evaluation.
"""
from __future__ import annotations

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
HIGH = FEATURES.index("high")
LOOKBACK = 512
SAMPLES = 64
SEED = 42

MODELS = {
    "v4-24": {
        "label": "EWZ v4",
        "sublabel": "24 bars · masked loss",
        "horizon": 24,
        "temperature": 1.0,
        "top_p": 0.85,
        "model_dir": ROOT / "finetuned/ewz_h1_small_v4_24_masked/basemodel/best_model",
        "tokenizer_dir": ROOT / "finetuned/ewz_h1_small_v2/tokenizer/best_model",
        "accent": "mint",
    },
    "v3-7": {
        "label": "EWZ v3",
        "sublabel": "7 bars · 1 trading day",
        "horizon": 7,
        "temperature": 1.2,
        "top_p": 0.90,
        "model_dir": ROOT / "finetuned/ewz_h1_small_v3_7bar/basemodel/best_model",
        "tokenizer_dir": ROOT / "finetuned/ewz_h1_small_v2/tokenizer/best_model",
        "accent": "blue",
    },
    "zero-24": {
        "label": "Zero-shot",
        "sublabel": "24 bars · pretrained",
        "horizon": 24,
        "temperature": 1.0,
        "top_p": 0.90,
        "model_dir": ROOT / "pretrained/Kronos-small",
        "tokenizer_dir": ROOT / "pretrained/Kronos-Tokenizer-base",
        "accent": "slate",
    },
}


def iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def rnd(x, n=4):
    return round(float(x), n)


def future_session_timestamps(last_ts: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """Next weekday session bars after the last bar (14:00-20:00 UTC, winter)."""
    out = []
    day = pd.Timestamp(last_ts).normalize() + pd.Timedelta(days=1)
    while len(out) < count:
        if day.weekday() < 5:
            for hour in range(14, 21):
                out.append(day + pd.Timedelta(hours=hour))
                if len(out) >= count:
                    break
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(out)


def build_date_grid(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    test_start = int(n * 0.9)
    anchors: dict[int, str] = {}

    # Last bar of each month in the test period.
    sub = df.iloc[test_start:].copy()
    sub["ym"] = sub["timestamps"].dt.to_period("M")
    for _, group in sub.groupby("ym"):
        idx = int(group.index[-1])
        anchors[idx + 1] = df["timestamps"].iloc[idx].strftime("%b %Y")

    # A few mid-month snapshots for variety.
    for day in ("2026-02-17", "2026-05-18", "2026-08-17"):
        idx = int(df.index[df["timestamps"] <= pd.Timestamp(day)].max())
        anchors[idx + 1] = df["timestamps"].iloc[idx].strftime("%b %d, %Y")

    # Latest available bar has no actual continuation.
    anchors[n] = "Latest"

    dates = []
    for anchor, label in sorted(anchors.items()):
        if anchor < LOOKBACK:
            continue
        if anchor < n and anchor + 24 > n:
            continue
        latest = anchor == n
        hist = df.iloc[anchor - LOOKBACK:anchor]
        last_ts = hist["timestamps"].iloc[-1]
        if latest:
            label = f"Latest · {last_ts.strftime('%b %d, %Y')}"
        dates.append({"id": str(anchor), "anchor": anchor, "label": label, "latest": latest})

    # Newest first in the UI.
    dates.sort(key=lambda d: d["anchor"], reverse=True)
    return dates


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_path = ROOT / "data/processed/ewz_h1.csv"
    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    print(f"data: {len(df)} bars | {df.timestamps.iloc[0]} -> {df.timestamps.iloc[-1]}")
    dates = build_date_grid(df)
    print(f"dates: {len(dates)} -> {[d['label'] for d in dates]}")

    if torch.cuda.is_available():
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    else:
        device = "cpu"
    print(f"device: {device}")

    # Load the three model/tokenizer pairs once.
    loaded = {}
    tok_ft = KronosTokenizer.from_pretrained(str(MODELS["v3-7"]["tokenizer_dir"])).to(device).eval()
    tok_pre = KronosTokenizer.from_pretrained(str(MODELS["zero-24"]["tokenizer_dir"])).to(device).eval()
    for key, spec in MODELS.items():
        print(f"loading {key}: {spec['model_dir']}")
        model = Kronos.from_pretrained(str(spec["model_dir"])).to(device).eval()
        tok = tok_ft if key != "zero-24" else tok_pre
        loaded[key] = {"model": model, "tokenizer": tok}

    out = {
        "meta": {
            "asset": "EWZ · iShares MSCI Brazil ETF",
            "source": "Dukascopy · EWZ.US-USD · 1h · BID",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "lookback": LOOKBACK,
            "samples": SAMPLES,
            "disclaimer": "Research demo, not investment advice. Probabilistic scenarios, not predictions.",
        },
        "models": {
            k: {
                "label": v["label"],
                "sublabel": v["sublabel"],
                "horizon": v["horizon"],
                "temperature": v["temperature"],
                "top_p": v["top_p"],
                "accent": v["accent"],
            }
            for k, v in MODELS.items()
        },
        "dates": [],
        "scenarios": {},
    }

    for di, d in enumerate(dates):
        anchor = d["anchor"]
        hist = df.iloc[anchor - LOOKBACK:anchor].reset_index(drop=True)
        last_ts = hist["timestamps"].iloc[-1]
        is_latest = d["latest"]

        if is_latest:
            future_ts = future_session_timestamps(last_ts, 24)
            actual_df = None
        else:
            future_ts = df["timestamps"].iloc[anchor:anchor + 24].reset_index(drop=True)
            actual_df = df.iloc[anchor:anchor + 24].reset_index(drop=True)

        history = {
            "t": [iso(t) for t in hist["timestamps"]],
            "o": [rnd(v) for v in hist["open"]],
            "h": [rnd(v) for v in hist["high"]],
            "l": [rnd(v) for v in hist["low"]],
            "c": [rnd(v) for v in hist["close"]],
            "v": [rnd(v, 3) for v in hist["volume"]],
        }
        actual = None
        if actual_df is not None:
            actual = {
                "t": [iso(t) for t in actual_df["timestamps"]],
                "c": [rnd(v) for v in actual_df["close"]],
                "h": [rnd(v) for v in actual_df["high"]],
            }

        out["dates"].append({
            "id": d["id"],
            "label": d["label"],
            "latest": is_latest,
            "history": history,
            "actual": actual,
        })

        x = hist[FEATURES].to_numpy(dtype=np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)
        x_stamp = calc_time_stamps(hist["timestamps"]).to_numpy(dtype=np.float32)

        for key, spec in MODELS.items():
            horizon = spec["horizon"]
            print(f"[{di+1}/{len(dates)}] {d['label']} | {key} | h={horizon}")
            y_ts = future_ts[:horizon]
            y_stamp = calc_time_stamps(pd.Series(y_ts)).to_numpy(dtype=np.float32)

            torch.manual_seed(SEED + di)
            paths_norm = forecast_paths(
                loaded[key]["tokenizer"], loaded[key]["model"],
                torch.from_numpy(x_norm[None]).to(device),
                torch.from_numpy(x_stamp[None]).to(device),
                torch.from_numpy(y_stamp[None]).to(device),
                max_context=512, pred_len=horizon, clip=5.0,
                T=spec["temperature"], top_p=spec["top_p"],
                sample_count=SAMPLES, verbose=False,
            )
            paths = paths_norm * (std + 1e-5) + mean
            pred_close = paths[:, :, CLOSE]
            pred_high = paths[:, :, HIGH]
            close0 = float(hist["close"].iloc[-1])

            r_paths = pred_close[:, -1] / close0 - 1.0
            median_path = np.median(pred_close, axis=0)
            p5, p25, p50, p75, p95 = np.percentile(pred_close, [5, 25, 50, 75, 95], axis=0)

            vol_paths = np.std(np.diff(np.log(pred_close), axis=1), axis=1)
            trail_vol = float(np.std(np.diff(np.log(hist["close"].iloc[-25:].to_numpy()))))
            vol_pred = float(np.median(vol_paths))

            metrics = {
                "p_up": rnd((r_paths > 0).mean()),
                "p_up_2": rnd((r_paths >= 0.02).mean()),
                "p_up_5": rnd((r_paths >= 0.05).mean()),
                "p_down_2": rnd((r_paths <= -0.02).mean()),
                "median_r": rnd(np.median(r_paths), 5),
                "mean_r": rnd(r_paths.mean(), 5),
                "close0": rnd(close0),
                "vol_pred": rnd(vol_pred, 6),
                "trail_vol": rnd(trail_vol, 6),
                "amp": rnd(vol_pred / (trail_vol + 1e-12), 4),
                "p_vol_expand": rnd((vol_paths > trail_vol).mean()),
                "p5_r": rnd(p5[-1] / close0 - 1.0, 5),
                "p95_r": rnd(p95[-1] / close0 - 1.0, 5),
                "touch_2": rnd((pred_high.max(axis=1) / close0 - 1.0 >= 0.02).mean()),
            }

            if actual_df is not None:
                actual_close = actual_df["close"].to_numpy()[:horizon]
                r_actual = float(actual_close[-1] / close0 - 1.0)
                mae = float(np.mean(np.abs(median_path - actual_close)) / close0)
                rw_mae = float(np.mean(np.abs(close0 - actual_close)) / close0)
                metrics.update({
                    "r_actual": rnd(r_actual, 5),
                    "mae": rnd(mae, 5),
                    "rw_mae": rnd(rw_mae, 5),
                    "dir_correct": int(np.sign(np.median(r_paths)) == np.sign(r_actual)),
                    "coverage_90": rnd(np.mean((actual_close >= p5) & (actual_close <= p95))),
                })

            sample_idx = np.linspace(0, SAMPLES - 1, 8).astype(int)
            out["scenarios"][f"{key}|{d['id']}"] = {
                "model": key,
                "date": d["id"],
                "horizon": horizon,
                "forecast": {
                    "t": [iso(t) for t in y_ts],
                    "p5": [rnd(v) for v in p5],
                    "p25": [rnd(v) for v in p25],
                    "p50": [rnd(v) for v in p50],
                    "p75": [rnd(v) for v in p75],
                    "p95": [rnd(v) for v in p95],
                    "samples": [[rnd(v) for v in pred_close[i]] for i in sample_idx],
                },
                "metrics": metrics,
            }

    out_dir = ROOT / "demo/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = "window.DEMO_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"
    (out_dir / "demo.js").write_text(js)
    (out_dir / "demo.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {out_dir/'demo.js'} ({len(js)/1024:.0f} KB)")
    print(f"wrote {out_dir/'demo.json'}")


if __name__ == "__main__":
    main()
