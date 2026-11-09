import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def boot_ci(x, n_boot, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def main():
    args = parse_args()
    d = pd.read_csv(args.windows)
    n = len(d)

    mae = d["mae_norm"].to_numpy()
    rw_mae = d["rw_mae_norm"].to_numpy()
    rmse = d["rmse_norm"].to_numpy()
    rw_rmse = d["rw_rmse_norm"].to_numpy()
    diff_mae = mae - rw_mae
    diff_rmse = rmse - rw_rmse

    outcome = (d["r_actual"] > 0).to_numpy().astype(float)
    base = outcome.mean()
    brier = d["brier_up"].to_numpy()
    base_brier = (base - outcome) ** 2
    skill = 1.0 - brier.mean() / base_brier.mean()

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    skill_boot = []
    for row in idx:
        b = outcome[row].mean()
        bb = ((b - outcome[row]) ** 2).mean()
        skill_boot.append(1.0 - brier[row].mean() / bb if bb > 0 else np.nan)
    skill_ci = [float(np.nanpercentile(skill_boot, 2.5)), float(np.nanpercentile(skill_boot, 97.5))]

    dir_acc = float(d["dir_correct"].mean())
    dir_ci = wilson_ci(int(d["dir_correct"].sum()), n)

    out = {
        "windows": args.windows,
        "n_windows": n,
        "base_rate_up": float(base),
        "mae_model": float(mae.mean()),
        "mae_rw": float(rw_mae.mean()),
        "mae_diff": float(diff_mae.mean()),
        "mae_diff_ci95": boot_ci(diff_mae, args.n_boot, args.seed),
        "rmse_model": float(rmse.mean()),
        "rmse_rw": float(rw_rmse.mean()),
        "rmse_diff": float(diff_rmse.mean()),
        "rmse_diff_ci95": boot_ci(diff_rmse, args.n_boot, args.seed),
        "dir_acc": dir_acc,
        "dir_ci95": dir_ci,
        "brier": float(brier.mean()),
        "brier_base": float(base_brier.mean()),
        "brier_skill": float(skill),
        "brier_skill_ci95": skill_ci,
        "coverage_90": float(d["coverage_90"].mean()),
        "p_up_mean": float(d["p_up"].mean()),
    }
    if "p_up_raw" in d.columns:
        out["p_up_raw_mean"] = float(d["p_up_raw"].mean())
    if "brier_up_raw" in d.columns:
        out["brier_raw"] = float(d["brier_up_raw"].mean())
        raw_skill = 1.0 - d["brier_up_raw"].mean() / base_brier.mean()
        out["brier_skill_raw"] = float(raw_skill)
    if "qlike" in d.columns:
        out["qlike"] = float(d["qlike"].mean())

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
