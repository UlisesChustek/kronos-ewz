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
HIGH = FEATURES.index("high")
LOW = FEATURES.index("low")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / ".local-configs" / "config_ewz_h1_small.yaml"))
    p.add_argument("--data", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    return p.parse_args()


def resolve_paths(args, cfg):
    mp = cfg["model_paths"]
    base = Path(mp["base_path"]) / mp["exp_name"]
    model_dir = Path(args.model_dir) if args.model_dir else base / "basemodel" / "best_model"
    if args.tokenizer_dir:
        tokenizer_dir = Path(args.tokenizer_dir)
    elif mp.get("finetuned_tokenizer"):
        tokenizer_dir = Path(mp["finetuned_tokenizer"])
    else:
        tokenizer_dir = base / "tokenizer" / "best_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"model not found: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_dir}")
    return model_dir, tokenizer_dir


def _apply_map(p, spec):
    if not spec:
        return p
    return float(np.clip(np.interp(p, spec["x"], spec["y"], left=spec["y"][0], right=spec["y"][-1]), 0.0, 1.0))


def metric_row(paths, x_df, y_df, pred_len, calibration=None):
    calibration = calibration or {}
    maps = calibration.get("maps", {})
    close0 = float(x_df["close"].iloc[-1])
    pred_close = paths[:, :, CLOSE]
    pred_high = paths[:, :, HIGH]
    actual_close = y_df["close"].to_numpy()
    actual_high = y_df["high"].to_numpy()

    r_paths = pred_close[:, -1] / close0 - 1.0
    r_actual = float(actual_close[-1] / close0 - 1.0)

    p_up_raw = float((r_paths > 0).mean())
    p_up2_raw = float((r_paths >= 0.02).mean())
    p_up5_raw = float((r_paths >= 0.05).mean())
    p_up = _apply_map(p_up_raw, maps.get("p_up"))
    p_up2 = _apply_map(p_up2_raw, maps.get("p_up_2pct"))
    p_up5 = _apply_map(p_up5_raw, maps.get("p_up_5pct"))
    brier = (p_up - float(r_actual > 0)) ** 2
    brier_raw = (p_up_raw - float(r_actual > 0)) ** 2

    median_path = np.median(pred_close, axis=0)
    mae = float(np.mean(np.abs(median_path - actual_close)) / close0)
    rmse = float(np.sqrt(np.mean((median_path - actual_close) ** 2)) / close0)
    rw_mae = float(np.mean(np.abs(close0 - actual_close)) / close0)
    rw_rmse = float(np.sqrt(np.mean((close0 - actual_close) ** 2)) / close0)

    path_scale = float(calibration.get("path_scale", 1.0))
    pred_close_adj = median_path + path_scale * (pred_close - median_path)
    p5 = np.percentile(pred_close_adj, 5, axis=0)
    p95 = np.percentile(pred_close_adj, 95, axis=0)
    coverage = float(np.mean((actual_close >= p5) & (actual_close <= p95)))
    p5r = np.percentile(pred_close, 5, axis=0)
    p95r = np.percentile(pred_close, 95, axis=0)
    coverage_raw = float(np.mean((actual_close >= p5r) & (actual_close <= p95r)))

    target = 0.02
    touch_pred_raw = float((pred_high.max(axis=1) / close0 - 1.0 >= target).mean())
    touch_pred = _apply_map(touch_pred_raw, maps.get("touch_pred_2pct"))
    touch_actual = float(actual_high.max() / close0 - 1.0 >= target)
    brier_touch = (touch_pred - touch_actual) ** 2
    brier_touch_raw = (touch_pred_raw - touch_actual) ** 2

    pred_vol_paths = np.std(np.diff(np.log(pred_close), axis=1), axis=1)
    vol_pred_raw = float(np.median(pred_vol_paths))
    vol_pred = vol_pred_raw * float(calibration.get("vol_scale", 1.0))
    vol_actual = float(np.std(np.diff(np.log(actual_close))))
    trail_vol = float(np.std(np.diff(np.log(x_df["close"].iloc[-25:].to_numpy()))))
    eps = 1e-12
    amp_pred = vol_pred / (trail_vol + eps)
    amp_actual = vol_actual / (trail_vol + eps)
    qlike = float(np.log(vol_pred ** 2 + eps) + (vol_actual ** 2 + eps) / (vol_pred ** 2 + eps))

    return {
        "close0": close0,
        "r_actual": r_actual,
        "p_up": p_up,
        "p_up_raw": p_up_raw,
        "p_up_2pct": p_up2,
        "p_up_2pct_raw": p_up2_raw,
        "p_up_5pct": p_up5,
        "p_up_5pct_raw": p_up5_raw,
        "median_r": float(np.median(r_paths)),
        "dir_correct": int(np.sign(np.median(r_paths)) == np.sign(r_actual)),
        "dir_correct_cal": int((p_up > 0.5) == (r_actual > 0)),
        "brier_up": brier,
        "brier_up_raw": brier_raw,
        "touch_pred_2pct": touch_pred,
        "touch_pred_2pct_raw": touch_pred_raw,
        "touch_actual_2pct": touch_actual,
        "brier_touch_2pct": brier_touch,
        "brier_touch_2pct_raw": brier_touch_raw,
        "mae_norm": mae,
        "rmse_norm": rmse,
        "rw_mae_norm": rw_mae,
        "rw_rmse_norm": rw_rmse,
        "coverage_90": coverage,
        "coverage_90_raw": coverage_raw,
        "vol_pred": vol_pred,
        "vol_pred_raw": vol_pred_raw,
        "vol_actual": vol_actual,
        "trail_vol": trail_vol,
        "amp_pred": amp_pred,
        "amp_actual": amp_actual,
        "qlike": qlike,
        "n_samples": int(paths.shape[0]),
        "pred_len": pred_len,
    }


def aggregate(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("n_samples", "pred_len")]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    base = float(np.mean([r["r_actual"] > 0 for r in rows]))
    out["base_rate_up"] = base
    brier = float(np.mean([r["brier_up"] for r in rows]))
    brier_raw = float(np.mean([r["brier_up_raw"] for r in rows]))
    brier_base = float(np.mean([(base - float(r["r_actual"] > 0)) ** 2 for r in rows]))
    out["brier_skill_vs_base"] = 1.0 - brier / brier_base if brier_base > 0 else float("nan")
    out["brier_skill_vs_base_raw"] = 1.0 - brier_raw / brier_base if brier_base > 0 else float("nan")
    return out


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    data_path = Path(args.data) if args.data else Path(cfg["data"]["data_path"])
    model_dir, tokenizer_dir = resolve_paths(args, cfg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(device).eval()
    model = Kronos.from_pretrained(str(model_dir)).to(device).eval()
    print(f"model={model_dir} | tokenizer={tokenizer_dir} | device={device}")

    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text())
        print(f"calibration={args.calibration} (T={calibration.get('temperature')}, top_p={calibration.get('top_p')}, vol_scale={calibration.get('vol_scale')})")

    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    d = cfg["data"]
    lookback = int(d["lookback_window"])
    pred_len = int(args.pred_len)
    stride = int(args.stride or pred_len)
    n = len(df)
    train_end = int(n * float(d["train_ratio"]))
    val_end = int(n * (float(d["train_ratio"]) + float(d["val_ratio"])))
    if args.split == "train":
        split_start, split_end = lookback, train_end
    elif args.split == "val":
        split_start, split_end = train_end, val_end
    else:
        split_start, split_end = val_end, n

    anchors = list(range(max(split_start, lookback), split_end - pred_len + 1, stride))
    print(f"rows={n} | split={args.split} [{split_start}:{split_end}] | windows={len(anchors)} | pred_len={pred_len} | samples={args.samples}")

    rows = []
    for anchor in anchors:
        x_df = df.iloc[anchor - lookback:anchor][FEATURES].reset_index(drop=True)
        y_df = df.iloc[anchor:anchor + pred_len].reset_index(drop=True)
        x_ts = df["timestamps"].iloc[anchor - lookback:anchor].reset_index(drop=True)
        y_ts = df["timestamps"].iloc[anchor:anchor + pred_len].reset_index(drop=True)

        x = x_df.to_numpy(dtype=np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)

        x_stamp = calc_time_stamps(x_ts).to_numpy(dtype=np.float32)
        y_stamp = calc_time_stamps(y_ts).to_numpy(dtype=np.float32)

        x_t = torch.from_numpy(x_norm[None]).to(device)
        xs_t = torch.from_numpy(x_stamp[None]).to(device)
        ys_t = torch.from_numpy(y_stamp[None]).to(device)

        paths_norm = forecast_paths(
            tokenizer, model, x_t, xs_t, ys_t,
            max_context=int(d["max_context"]), pred_len=pred_len,
            clip=float(d["clip"]), T=args.temperature, top_p=args.top_p,
            sample_count=args.samples, verbose=False,
        )
        paths = paths_norm * (std + 1e-5) + mean
        rows.append(metric_row(paths, x_df, y_df, pred_len, calibration))

    summary = {
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "data": str(data_path),
        "split": args.split,
        "n_windows": len(rows),
        "pred_len": pred_len,
        "samples_per_window": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "calibration": str(args.calibration) if args.calibration else None,
        "metrics": aggregate(rows),
    }

    out_path = Path(args.out) if args.out else ROOT / "results" / f"eval_{model_dir.parent.parent.name}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".windows.csv"), index=False)

    m = summary["metrics"]
    print(f"\nwrote {out_path}")
    print(f"windows={len(rows)} | base_rate_up={m['base_rate_up']:.3f}")
    for k in ["p_up", "p_up_raw", "median_r", "dir_correct", "dir_correct_cal", "brier_up", "brier_up_raw",
              "brier_skill_vs_base", "brier_skill_vs_base_raw", "touch_pred_2pct", "brier_touch_2pct",
              "mae_norm", "rw_mae_norm", "coverage_90", "vol_pred", "vol_pred_raw", "vol_actual",
              "amp_pred", "amp_actual", "qlike"]:
        v = m[k]
        if isinstance(v, dict):
            print(f"  {k:22s} {v['mean']:+.4f}  (std {v['std']:.4f})")
        else:
            print(f"  {k:22s} {v:+.4f}")


if __name__ == "__main__":
    main()
