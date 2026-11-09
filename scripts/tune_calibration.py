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
sys.path.insert(0, str(ROOT / "scripts"))

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps
from kronos_paths import forecast_paths
from evaluate import FEATURES, metric_row, aggregate, resolve_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / ".local-configs" / "config_ewz_h1_small_v2.yaml"))
    p.add_argument("--data", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--temperatures", default="0.8,1.0,1.2")
    p.add_argument("--top-ps", default="0.85,0.9,0.95")
    p.add_argument("--path-scales", default="0.8,1.0,1.25,1.5,1.75,2.0,2.5,3.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(ROOT / "results" / "calibration.json"))
    return p.parse_args()


def isotonic_knots(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]

    starts, ends, vals, wts = [], [], [], []
    for i in range(len(xs)):
        starts.append(i)
        ends.append(i)
        vals.append(ys[i])
        wts.append(1.0)
        while len(vals) >= 2 and vals[-2] > vals[-1] + 1e-12:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            ends[-2] = ends[-1]
            vals[-2] = v
            wts[-2] = w
            starts.pop()
            ends.pop()
            vals.pop()
            wts.pop()

    fitted = np.empty(len(xs), dtype=float)
    for s, e, v in zip(starts, ends, vals):
        fitted[s:e + 1] = v

    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.zeros(len(ux), dtype=float)
    cnt = np.zeros(len(ux), dtype=float)
    np.add.at(uy, inv, fitted)
    np.add.at(cnt, inv, 1.0)
    uy /= cnt
    uy = np.maximum.accumulate(uy)
    return ux, uy


def fit_maps(rows):
    maps = {}
    targets = {
        "p_up": ("p_up_raw", lambda r: float(r["r_actual"] > 0)),
        "p_up_2pct": ("p_up_2pct_raw", lambda r: float(r["r_actual"] >= 0.02)),
        "p_up_5pct": ("p_up_5pct_raw", lambda r: float(r["r_actual"] >= 0.05)),
        "touch_pred_2pct": ("touch_pred_2pct_raw", lambda r: float(r["touch_actual_2pct"] > 0)),
    }
    for name, (col, fn) in targets.items():
        xs = [r[col] for r in rows]
        ys = [fn(r) for r in rows]
        ux, uy = isotonic_knots(xs, ys)
        maps[name] = {"x": ux.tolist(), "y": uy.tolist()}
    return maps


def apply_map(p, spec):
    return float(np.clip(np.interp(p, spec["x"], spec["y"], left=spec["y"][0], right=spec["y"][-1]), 0.0, 1.0))


def brier(rows, col="p_up_raw", target=lambda r: float(r["r_actual"] > 0)):
    return float(np.mean([(r[col] - target(r)) ** 2 for r in rows]))


def coverage_curve(paths, x_df, y_df, alphas):
    from evaluate import CLOSE
    close0 = float(x_df["close"].iloc[-1])
    pred_close = paths[:, :, CLOSE]
    median = np.median(pred_close, axis=0)
    actual = y_df["close"].to_numpy()
    out = []
    for a in alphas:
        adj = median + a * (pred_close - median)
        p5 = np.percentile(adj, 5, axis=0)
        p95 = np.percentile(adj, 95, axis=0)
        out.append(float(np.mean((actual >= p5) & (actual <= p95))))
    return out


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    data_path = Path(args.data) if args.data else Path(cfg["data"]["data_path"])
    model_dir, tokenizer_dir = resolve_paths(args, cfg)

    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(device).eval()
    model = Kronos.from_pretrained(str(model_dir)).to(device).eval()
    print(f"model={model_dir} | tokenizer={tokenizer_dir} | device={device}")

    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    d = cfg["data"]
    lookback = int(d["lookback_window"])
    pred_len = int(args.pred_len)
    n = len(df)
    train_end = int(n * float(d["train_ratio"]))
    val_end = int(n * (float(d["train_ratio"]) + float(d["val_ratio"])))
    split_start, split_end = (lookback, train_end) if args.split == "train" else (train_end, val_end)

    anchors = list(range(max(split_start, lookback), split_end - pred_len + 1, args.stride))
    if args.max_windows:
        anchors = anchors[-int(args.max_windows):]
    print(f"split={args.split} | anchors={len(anchors)} | stride={args.stride} | samples={args.samples}")

    windows = []
    for anchor in anchors:
        x_df = df.iloc[anchor - lookback:anchor][FEATURES].reset_index(drop=True)
        y_df = df.iloc[anchor:anchor + pred_len][FEATURES].reset_index(drop=True)
        x_ts = df["timestamps"].iloc[anchor - lookback:anchor].reset_index(drop=True)
        y_ts = df["timestamps"].iloc[anchor:anchor + pred_len].reset_index(drop=True)
        x = x_df.to_numpy(dtype=np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)
        x_stamp = calc_time_stamps(x_ts).to_numpy(dtype=np.float32)
        y_stamp = calc_time_stamps(y_ts).to_numpy(dtype=np.float32)
        windows.append((x_df, y_df, x_norm, x_stamp, y_stamp, mean, std))

    temperatures = [float(v) for v in args.temperatures.split(",")]
    top_ps = [float(v) for v in args.top_ps.split(",")]
    alphas = [float(v) for v in args.path_scales.split(",")]
    grid = []
    best = None
    for T in temperatures:
        for top_p in top_ps:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            rows = []
            cov_curves = []
            for x_df, y_df, x_norm, x_stamp, y_stamp, mean, std in windows:
                x_t = torch.from_numpy(x_norm[None]).to(device)
                xs_t = torch.from_numpy(x_stamp[None]).to(device)
                ys_t = torch.from_numpy(y_stamp[None]).to(device)
                paths_norm = forecast_paths(
                    tokenizer, model, x_t, xs_t, ys_t,
                    max_context=int(d["max_context"]), pred_len=pred_len,
                    clip=float(d["clip"]), T=T, top_p=top_p,
                    sample_count=args.samples, verbose=False,
                )
                paths = paths_norm * (std + 1e-5) + mean
                rows.append(metric_row(paths, x_df, y_df, pred_len))
                cov_curves.append(coverage_curve(paths, x_df, y_df, alphas))

            m = aggregate(rows)
            rec = {
                "temperature": T,
                "top_p": top_p,
                "n_windows": len(rows),
                "brier_up": float(m["brier_up_raw"]["mean"]),
                "brier_skill_vs_base": float(m["brier_skill_vs_base_raw"]),
                "mae_norm": float(m["mae_norm"]["mean"]),
                "rmse_norm": float(m["rmse_norm"]["mean"]),
                "dir_correct": float(m["dir_correct"]["mean"]),
                "p_up_mean": float(m["p_up_raw"]["mean"]),
                "base_rate_up": float(m["base_rate_up"]),
                "vol_pred": float(m["vol_pred_raw"]["mean"]),
                "vol_actual": float(m["vol_actual"]["mean"]),
                "amp_pred": float(m["amp_pred"]["mean"]),
                "amp_actual": float(m["amp_actual"]["mean"]),
            }
            grid.append(rec)
            print(f"T={T:.2f} top_p={top_p:.2f} | brier={rec['brier_up']:.4f} skill={rec['brier_skill_vs_base']:+.3f} "
                  f"mae={rec['mae_norm']:.4f} dir={rec['dir_correct']:.3f} p_up={rec['p_up_mean']:.3f}")
            if best is None or (rec["brier_up"], rec["mae_norm"]) < (best["rec"]["brier_up"], best["rec"]["mae_norm"]):
                best = {"rec": rec, "rows": rows, "cov_curves": cov_curves}

    rows = best["rows"]
    cut = max(1, int(len(rows) * 0.7))
    fit_rows, hold_rows = rows[:cut], rows[cut:]
    maps = fit_maps(fit_rows)
    vol_scale = float(np.median([r["vol_actual"] / (r["vol_pred_raw"] + 1e-12) for r in fit_rows]))
    vol_scale = float(np.clip(vol_scale, 0.5, 3.0))

    cov_fit = np.mean(np.array(best["cov_curves"][:cut]), axis=0)
    path_scale = float(alphas[int(np.argmin(np.abs(cov_fit - 0.9)))])
    cov_hold = np.mean(np.array(best["cov_curves"][cut:]), axis=0)
    cov_hold_raw = float(cov_hold[int(np.argmin(np.abs(np.array(alphas) - 1.0)))])
    cov_hold_cal = float(cov_hold[int(np.argmin(np.abs(np.array(alphas) - path_scale)))])

    hold_raw = brier(hold_rows, "p_up_raw")
    hold_cal = float(np.mean([(apply_map(r["p_up_raw"], maps["p_up"]) - float(r["r_actual"] > 0)) ** 2 for r in hold_rows]))
    hold_vol_raw = float(np.mean([(np.log(r["vol_pred_raw"] ** 2 + 1e-12) + (r["vol_actual"] ** 2 + 1e-12) / (r["vol_pred_raw"] ** 2 + 1e-12)) for r in hold_rows]))
    hold_vol_cal = float(np.mean([(np.log((r["vol_pred_raw"] * vol_scale) ** 2 + 1e-12) + (r["vol_actual"] ** 2 + 1e-12) / ((r["vol_pred_raw"] * vol_scale) ** 2 + 1e-12)) for r in hold_rows]))

    final_maps = fit_maps(rows)
    final_vol_scale = float(np.clip(np.median([r["vol_actual"] / (r["vol_pred_raw"] + 1e-12) for r in rows]), 0.5, 3.0))

    out = {
        "config": str(args.config),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "split": args.split,
        "pred_len": pred_len,
        "stride": args.stride,
        "samples_per_window": args.samples,
        "n_windows": len(rows),
        "temperature": best["rec"]["temperature"],
        "top_p": best["rec"]["top_p"],
        "vol_scale": final_vol_scale,
        "path_scale": path_scale,
        "maps": final_maps,
        "val_metrics_raw": aggregate(rows),
        "holdout": {
            "n_windows": len(hold_rows),
            "brier_raw": hold_raw,
            "brier_calibrated": hold_cal,
            "qlike_raw": hold_vol_raw,
            "qlike_calibrated": hold_vol_cal,
            "coverage_raw": cov_hold_raw,
            "coverage_calibrated": cov_hold_cal,
            "vol_scale_fit": vol_scale,
            "path_scale_fit": path_scale,
        },
        "grid": grid,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nbest T={best['rec']['temperature']} top_p={best['rec']['top_p']} | vol_scale={final_vol_scale:.3f} path_scale={path_scale:.3f}")
    print(f"holdout brier raw={hold_raw:.4f} calibrated={hold_cal:.4f} | qlike raw={hold_vol_raw:.4f} calibrated={hold_vol_cal:.4f}")
    print(f"holdout coverage raw={cov_hold_raw:.3f} calibrated={cov_hold_cal:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
