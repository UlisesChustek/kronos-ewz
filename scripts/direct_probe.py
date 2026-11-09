#!/usr/bin/env python3
"""Direct 7-bar probe on frozen pretrained Kronos representations.

This is a small, deliberately leakage-aware experiment:

* Load the ORIGINAL pretrained ``Kronos-small`` and the matching ORIGINAL
  ``Kronos-Tokenizer-base`` from ``pretrained/`` (local files only), both in
  evaluation mode.
* At each anchor, take exactly the 128 bars strictly before the anchor,
  standardize each feature within that context window only, clip to [-5, 5],
  tokenize, run ``decode_s1``, and use the final context hidden vector.
* Compare two linear heads, each fit twice (representation features and
  ordinary past-only lagged features):
    - ridge regression on the 7-bar terminal return;
    - logistic regression on the 7-bar direction.
* Use the existing chronological 80/10/10 split.  Anchors are spaced 7 bars
  apart so targets do not overlap, and every target lies wholly inside its
  split.  The test anchors are exactly the 244 anchors used by the legacy v5
  7-bar evaluation.
* Standardization is fit on training rows only.  The L2 penalty is selected
  on the chronological validation split only.  Coefficients are fit on train
  only; validation is not reused for fitting.  Test labels are used once, for
  the final reported metrics.  There is no tuning after seeing test results.
* Report terminal MAE versus a zero return, Brier versus the training up-rate,
  direction accuracy, AUC, and paired circular moving-block bootstrap
  intervals.  The test period has already been inspected in earlier project
  work, so these numbers are retrospective, not a fresh confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Kronos"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import calc_time_stamps  # noqa: E402
from walk_forward_baselines import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_WARMUP,
    FeatureScaler,
    feature_row,
    fit_logistic,
    fit_ridge,
    logloss,
    moving_block_indices,
    predict_logistic,
    predict_ridge,
)

DATA_PATH = ROOT / "data" / "processed" / "ewz_h1.csv"
LEGACY_TEST_WINDOWS = ROOT / "results" / "eval_v5_test_7bar_raw.windows.csv"
PREDICTOR_DIR = ROOT / "pretrained" / "Kronos-small"
TOKENIZER_DIR = ROOT / "pretrained" / "Kronos-Tokenizer-base"

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK = 128
HORIZON = 7
STRIDE = 7
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# Strong L2 candidates.  Selection is validation-only; the grid is fixed before
# any test outcome is examined.
LAMBDA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)

SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--legacy-test-windows", type=Path, default=LEGACY_TEST_WINDOWS)
    p.add_argument("--predictor-dir", type=Path, default=PREDICTOR_DIR)
    p.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "direct_probe.json")
    p.add_argument("--device", default="cuda:1", help="cuda:1 by default; use cpu only if intended")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamps", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    if df["close"].isna().any():
        raise ValueError("close contains NaN values")
    return df


def split_boundaries(n_rows: int) -> tuple[int, int]:
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


def make_anchors(n_rows: int) -> dict[str, np.ndarray]:
    """Non-overlapping 7-bar targets wholly inside each chronological split.

    The test slice deliberately reproduces ``evaluate.py``'s legacy v5
    anchors: ``range(val_end, n_rows - 7 + 1, 7)``.
    """
    train_end, val_end = split_boundaries(n_rows)
    # feature_row needs FEATURE_WARMUP past bars; the representation needs 128.
    train_start = max(LOOKBACK, FEATURE_WARMUP)
    train = np.arange(train_start, train_end - HORIZON + 1, STRIDE, dtype=int)
    val = np.arange(train_end, val_end - HORIZON + 1, STRIDE, dtype=int)
    test = np.arange(val_end, n_rows - HORIZON + 1, STRIDE, dtype=int)
    splits = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, n_rows),
    }
    for name, anchors in (("train", train), ("validation", val), ("test", test)):
        start, end = splits[name]
        if len(anchors) == 0:
            raise ValueError(f"{name}: no anchors")
        if anchors[0] < LOOKBACK:
            raise ValueError(f"{name}: first anchor has no 128-bar lookback")
        if not np.all(anchors >= start):
            raise ValueError(f"{name}: anchor before split start")
        if not np.all(anchors + HORIZON - 1 < end):
            raise ValueError(f"{name}: target crosses split end")
        if not np.all(np.diff(anchors) == STRIDE):
            raise ValueError(f"{name}: anchors are not uniformly non-overlapping")
    return {"train": train, "validation": val, "test": test}


def target_returns(df: pd.DataFrame, anchors: np.ndarray) -> np.ndarray:
    c = df["close"].to_numpy(dtype=float)
    close0 = c[anchors - 1]
    terminal = c[anchors + HORIZON - 1]
    return terminal / close0 - 1.0


def extract_representations(
    df: pd.DataFrame,
    anchors: np.ndarray,
    tokenizer: KronosTokenizer,
    model: Kronos,
    device: torch.device,
    batch_size: int,
    label: str,
) -> np.ndarray:
    """Last decode_s1 hidden vector for each anchor, using context-only scaling."""
    d_model = int(model.d_model)
    reps = np.empty((len(anchors), d_model), dtype=np.float32)
    timestamps = df["timestamps"]
    values = df[FEATURES]

    started = time.time()
    for start in range(0, len(anchors), batch_size):
        batch_anchors = anchors[start:start + batch_size]
        xs = []
        stamps = []
        for anchor in batch_anchors:
            anchor = int(anchor)
            x = values.iloc[anchor - LOOKBACK:anchor].to_numpy(dtype=np.float32)
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            xs.append(np.clip((x - mean) / (std + 1e-5), -5.0, 5.0))
            ts = timestamps.iloc[anchor - LOOKBACK:anchor].reset_index(drop=True)
            stamps.append(calc_time_stamps(ts).to_numpy(dtype=np.float32))

        x_t = torch.from_numpy(np.stack(xs, axis=0)).to(device)
        stamp_t = torch.from_numpy(np.stack(stamps, axis=0)).to(device)
        with torch.inference_mode():
            s1_ids, s2_ids = tokenizer.encode(x_t, half=True)
            _, context = model.decode_s1(s1_ids, s2_ids, stamp_t)
            h = context[:, -1, :].float().cpu().numpy()

        end = start + len(batch_anchors)
        reps[start:end] = h
        elapsed = time.time() - started
        print(
            f"  [{label}] representations {end}/{len(anchors)} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    if not np.isfinite(reps).all():
        raise ValueError("non-finite representation values")
    return reps


def build_feature_matrix(df: pd.DataFrame, anchors: np.ndarray) -> np.ndarray:
    return np.vstack([feature_row(df, int(a)) for a in anchors])


def select_ridge(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray
) -> tuple[dict, list[dict]]:
    rows = []
    for lam in LAMBDA_GRID:
        beta = fit_ridge(Xtr, ytr, lam)
        pred = predict_ridge(Xva, beta)
        row = {
            "lambda": float(lam),
            "validation_mae": float(np.mean(np.abs(pred - yva))),
        }
        rows.append(row)
    # Argmin; on exact ties keep the stronger penalty (later in the fixed grid).
    best = min(rows, key=lambda r: (r["validation_mae"], -r["lambda"]))
    return best, rows


def select_logistic(
    Xtr: np.ndarray, ytr_up: np.ndarray, Xva: np.ndarray, yva_up: np.ndarray
) -> tuple[dict, list[dict]]:
    rows = []
    for lam in LAMBDA_GRID:
        beta = fit_logistic(Xtr, ytr_up, lam)
        p = predict_logistic(Xva, beta)
        row = {
            "lambda": float(lam),
            "validation_logloss": logloss(yva_up, p),
            "validation_brier": float(np.mean((p - yva_up) ** 2)),
            "validation_direction_acc": float(np.mean((p > 0.5) == (yva_up == 1))),
        }
        rows.append(row)
    # Argmin validation log loss; stronger penalty wins exact ties.
    best = min(rows, key=lambda r: (r["validation_logloss"], -r["lambda"]))
    return best, rows


def auc_score(score: np.ndarray, y_up: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    y_up = np.asarray(y_up, dtype=bool)
    n1 = int(y_up.sum())
    n0 = len(y_up) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y_up].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def direction_accuracy(pred_up: np.ndarray, y_up: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_up, dtype=bool) == np.asarray(y_up, dtype=bool)))


def bootstrap_metrics(
    n: int,
    block_len: int,
    n_boot: int,
    seed: int,
    r: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    y: np.ndarray,
    y_up: np.ndarray,
    train_prior: float,
) -> dict[str, np.ndarray]:
    """One shared circular moving-block resampling for all paired statistics."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        out.setdefault(name, []).append(value)

    for _ in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        yy = y[idx]
        yu = y_up[idx]
        abs_y = np.abs(yy)
        zero_mae = float(np.mean(abs_y))
        prior_brier = float(np.mean((train_prior - yu) ** 2))

        for name, rh in r.items():
            mae = float(np.mean(np.abs(rh[idx] - yy)))
            add(f"{name}_terminal_mae", mae)
            add(f"{name}_terminal_mae_minus_zero", mae - zero_mae)
            add(f"{name}_direction_acc", direction_accuracy(rh[idx] > 0, yu))
            add(f"{name}_auc", auc_score(rh[idx], yu))

        for name, ph in p.items():
            brier = float(np.mean((ph[idx] - yu) ** 2))
            add(f"{name}_brier", brier)
            add(f"{name}_brier_minus_prior", brier - prior_brier)
            add(f"{name}_brier_skill_vs_prior", 1.0 - brier / prior_brier)
            add(f"{name}_direction_acc", direction_accuracy(ph[idx] > 0.5, yu))
            add(f"{name}_auc", auc_score(ph[idx], yu))

        add("zero_terminal_mae", zero_mae)
        add("train_prior_brier", prior_brier)

        # Paired representation-minus-lagged-head differences.
        add(
            "ridge_repr_minus_feat_terminal_mae",
            float(np.mean(np.abs(r["repr_ridge"][idx] - yy)))
            - float(np.mean(np.abs(r["feat_ridge"][idx] - yy))),
        )
        add(
            "ridge_repr_minus_feat_direction_acc",
            direction_accuracy(r["repr_ridge"][idx] > 0, yu)
            - direction_accuracy(r["feat_ridge"][idx] > 0, yu),
        )
        add(
            "ridge_repr_minus_feat_auc",
            auc_score(r["repr_ridge"][idx], yu) - auc_score(r["feat_ridge"][idx], yu),
        )
        add(
            "logistic_repr_minus_feat_brier",
            float(np.mean((p["repr_logistic"][idx] - yu) ** 2))
            - float(np.mean((p["feat_logistic"][idx] - yu) ** 2)),
        )
        add(
            "logistic_repr_minus_feat_brier_skill_vs_prior",
            (1.0 - float(np.mean((p["repr_logistic"][idx] - yu) ** 2)) / prior_brier)
            - (1.0 - float(np.mean((p["feat_logistic"][idx] - yu) ** 2)) / prior_brier),
        )
        add(
            "logistic_repr_minus_feat_direction_acc",
            direction_accuracy(p["repr_logistic"][idx] > 0.5, yu)
            - direction_accuracy(p["feat_logistic"][idx] > 0.5, yu),
        )
        add(
            "logistic_repr_minus_feat_auc",
            auc_score(p["repr_logistic"][idx], yu) - auc_score(p["feat_logistic"][idx], yu),
        )

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def ci95(samples: np.ndarray) -> list[float] | None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return None
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def point_metrics(
    r: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    y: np.ndarray,
    y_up: np.ndarray,
    train_prior: float,
) -> dict:
    abs_y = np.abs(y)
    zero_mae = float(np.mean(abs_y))
    prior_brier = float(np.mean((train_prior - y_up) ** 2))
    models = {}
    for name, rh in r.items():
        mae = float(np.mean(np.abs(rh - y)))
        models[name] = {
            "head": "ridge_return",
            "features": "representation" if name.startswith("repr") else "lagged",
            "terminal_mae": mae,
            "terminal_mae_minus_zero": mae - zero_mae,
            "direction_acc": direction_accuracy(rh > 0, y_up),
            "auc": auc_score(rh, y_up),
        }
    for name, ph in p.items():
        brier = float(np.mean((ph - y_up) ** 2))
        models[name] = {
            "head": "logistic_direction",
            "features": "representation" if name.startswith("repr") else "lagged",
            "brier": brier,
            "brier_minus_prior": brier - prior_brier,
            "brier_skill_vs_prior": 1.0 - brier / prior_brier if prior_brier > 0 else float("nan"),
            "direction_acc": direction_accuracy(ph > 0.5, y_up),
            "auc": auc_score(ph, y_up),
        }
    return {
        "zero_terminal_mae": zero_mae,
        "train_prior_brier": prior_brier,
        "models": models,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device.index} does not exist")
    print(f"device={device}", flush=True)

    df = load_data(args.data)
    n_rows = len(df)
    train_end, val_end = split_boundaries(n_rows)
    anchors = make_anchors(n_rows)

    print("loading frozen local models...", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True
    ).to(device).eval()
    model = Kronos.from_pretrained(
        str(args.predictor_dir), local_files_only=True
    ).to(device).eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)
    print(
        f"predictor={args.predictor_dir} | tokenizer={args.tokenizer_dir} | "
        f"d_model={model.d_model}",
        flush=True,
    )

    # Targets and ordinary past-only features.
    y_all = {name: target_returns(df, anchors[name]) for name in SPLIT_NAMES}
    y_up_all = {name: (y_all[name] > 0).astype(int) for name in SPLIT_NAMES}
    X_lag_all = {name: build_feature_matrix(df, anchors[name]) for name in SPLIT_NAMES}

    # Frozen-representation extraction.
    reps = {}
    for name in SPLIT_NAMES:
        print(f"extracting {name} representations...", flush=True)
        reps[name] = extract_representations(
            df, anchors[name], tokenizer, model, device,
            batch_size=args.batch_size, label=name,
        )

    # Standardization fitted on the training split only, reused unchanged.
    lag_scaler = FeatureScaler().fit(X_lag_all["train"])
    rep_scaler = FeatureScaler().fit(reps["train"])
    X_lag = {name: lag_scaler.transform(X_lag_all[name]) for name in SPLIT_NAMES}
    X_rep = {name: rep_scaler.transform(reps[name]) for name in SPLIT_NAMES}

    # Validation-only penalty selection.
    ridge_selection = {
        "repr_ridge": select_ridge(
            X_rep["train"], y_all["train"], X_rep["validation"], y_all["validation"]
        ),
        "feat_ridge": select_ridge(
            X_lag["train"], y_all["train"], X_lag["validation"], y_all["validation"]
        ),
    }
    logistic_selection = {
        "repr_logistic": select_logistic(
            X_rep["train"], y_up_all["train"],
            X_rep["validation"], y_up_all["validation"],
        ),
        "feat_logistic": select_logistic(
            X_lag["train"], y_up_all["train"],
            X_lag["validation"], y_up_all["validation"],
        ),
    }

    print("\nvalidation-only selection:", flush=True)
    for name, (best, _) in ridge_selection.items():
        print(f"  {name:16s} lambda={best['lambda']:g} val_MAE={best['validation_mae']:.6f}", flush=True)
    for name, (best, _) in logistic_selection.items():
        print(
            f"  {name:16s} lambda={best['lambda']:g} "
            f"val_logloss={best['validation_logloss']:.6f} "
            f"val_Brier={best['validation_brier']:.6f}",
            flush=True,
        )

    # Final coefficients: train only.  Validation is not reused for fitting.
    beta_repr_ridge = fit_ridge(
        X_rep["train"], y_all["train"], ridge_selection["repr_ridge"][0]["lambda"]
    )
    beta_feat_ridge = fit_ridge(
        X_lag["train"], y_all["train"], ridge_selection["feat_ridge"][0]["lambda"]
    )
    beta_repr_log = fit_logistic(
        X_rep["train"], y_up_all["train"], logistic_selection["repr_logistic"][0]["lambda"]
    )
    beta_feat_log = fit_logistic(
        X_lag["train"], y_up_all["train"], logistic_selection["feat_logistic"][0]["lambda"]
    )

    # The test split is evaluated only after every choice above is frozen.
    y = y_all["test"]
    y_up = y_up_all["test"]
    train_prior = float(np.mean(y_up_all["train"]))
    r = {
        "repr_ridge": predict_ridge(X_rep["test"], beta_repr_ridge),
        "feat_ridge": predict_ridge(X_lag["test"], beta_feat_ridge),
    }
    p = {
        "repr_logistic": predict_logistic(X_rep["test"], beta_repr_log),
        "feat_logistic": predict_logistic(X_lag["test"], beta_feat_log),
    }

    # Exact legacy test-anchor check against the saved 244-window CSV.
    legacy = pd.read_csv(args.legacy_test_windows)
    c = df["close"].to_numpy(dtype=float)
    close0 = c[anchors["test"] - 1]
    legacy_r = c[anchors["test"] + HORIZON - 1] / close0 - 1.0
    if len(legacy) != len(anchors["test"]):
        raise ValueError(
            f"legacy test windows={len(legacy)} but probe test anchors={len(anchors['test'])}"
        )
    close0_diff = float(np.max(np.abs(legacy["close0"].to_numpy() - close0)))
    r_diff = float(np.max(np.abs(legacy["r_actual"].to_numpy() - legacy_r)))
    if not (np.allclose(legacy["close0"], close0, rtol=1e-9, atol=1e-11)
            and np.allclose(legacy["r_actual"], legacy_r, rtol=1e-9, atol=1e-11)):
        raise ValueError("test anchors do not match the legacy 244-window evaluation")
    print(
        f"\nlegacy test-anchor check: {len(legacy)} windows; "
        f"max|Δclose0|={close0_diff:.3g}, max|Δr_actual|={r_diff:.3g}",
        flush=True,
    )

    point = point_metrics(r, p, y, y_up, train_prior)
    block_len = max(1, int(round(len(y) ** (1.0 / 3.0))))
    boot = bootstrap_metrics(
        len(y), block_len, args.n_boot, args.seed,
        r, p, y, y_up, train_prior,
    )
    print(
        f"bootstrap: {args.n_boot} paired circular moving-block draws, "
        f"block={block_len}, seed={args.seed}",
        flush=True,
    )

    # Attach bootstrap intervals to point metrics.
    for name, metrics in point["models"].items():
        for key in (
            "terminal_mae", "terminal_mae_minus_zero", "brier",
            "brier_minus_prior", "brier_skill_vs_prior",
            "direction_acc", "auc",
        ):
            if key in metrics:
                metrics[f"{key}_ci95"] = ci95(boot[f"{name}_{key}"])
    point["zero_terminal_mae_ci95"] = ci95(boot["zero_terminal_mae"])
    point["train_prior_brier_ci95"] = ci95(boot["train_prior_brier"])

    paired = {}
    for key in (
        "ridge_repr_minus_feat_terminal_mae",
        "ridge_repr_minus_feat_direction_acc",
        "ridge_repr_minus_feat_auc",
        "logistic_repr_minus_feat_brier",
        "logistic_repr_minus_feat_brier_skill_vs_prior",
        "logistic_repr_minus_feat_direction_acc",
        "logistic_repr_minus_feat_auc",
    ):
        paired[key] = {
            "point": float(np.mean(boot[key][np.isfinite(boot[key])])),
            "ci95": ci95(boot[key]),
        }

    # Point estimates for the paired differences, computed directly from test
    # predictions (not from the bootstrap-average above).
    paired["ridge_repr_minus_feat_terminal_mae"]["point"] = (
        point["models"]["repr_ridge"]["terminal_mae"]
        - point["models"]["feat_ridge"]["terminal_mae"]
    )
    paired["ridge_repr_minus_feat_direction_acc"]["point"] = (
        point["models"]["repr_ridge"]["direction_acc"]
        - point["models"]["feat_ridge"]["direction_acc"]
    )
    paired["ridge_repr_minus_feat_auc"]["point"] = (
        point["models"]["repr_ridge"]["auc"]
        - point["models"]["feat_ridge"]["auc"]
    )
    paired["logistic_repr_minus_feat_brier"]["point"] = (
        point["models"]["repr_logistic"]["brier"]
        - point["models"]["feat_logistic"]["brier"]
    )
    paired["logistic_repr_minus_feat_brier_skill_vs_prior"]["point"] = (
        point["models"]["repr_logistic"]["brier_skill_vs_prior"]
        - point["models"]["feat_logistic"]["brier_skill_vs_prior"]
    )
    paired["logistic_repr_minus_feat_direction_acc"]["point"] = (
        point["models"]["repr_logistic"]["direction_acc"]
        - point["models"]["feat_logistic"]["direction_acc"]
    )
    paired["logistic_repr_minus_feat_auc"]["point"] = (
        point["models"]["repr_logistic"]["auc"]
        - point["models"]["feat_logistic"]["auc"]
    )

    split_info = {}
    for name in SPLIT_NAMES:
        a = anchors[name]
        split_info[name] = {
            "n_anchors": int(len(a)),
            "first_anchor": int(a[0]),
            "last_anchor": int(a[-1]),
            "first_target_start": str(df["timestamps"].iloc[int(a[0])]),
            "last_target_end": str(df["timestamps"].iloc[int(a[-1]) + HORIZON - 1]),
            "first_close0_timestamp": str(df["timestamps"].iloc[int(a[0]) - 1]),
            "nonoverlapping_horizon": HORIZON,
            "targets_wholly_inside_split": True,
        }
    split_info["boundaries"] = {
        "n_rows": int(n_rows),
        "train_end_exclusive": int(train_end),
        "validation_end_exclusive": int(val_end),
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VAL_RATIO,
        "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    }

    result = {
        "schema": "kronos-ewz-direct-probe/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Retrospective direct probe on the existing 80/10/10 split. The test "
            "period has already been inspected in earlier project work; this is "
            "not fresh out-of-sample confirmation."
        ),
        "no_tuning_after_test": True,
        "data": {
            "path": str(args.data.resolve()),
            "n_rows": int(n_rows),
            "first_timestamp": str(df["timestamps"].iloc[0]),
            "last_timestamp": str(df["timestamps"].iloc[-1]),
        },
        "models": {
            "predictor": str(args.predictor_dir.resolve()),
            "tokenizer": str(args.tokenizer_dir.resolve()),
            "predictor_mode": "frozen, eval, inference",
            "tokenizer_mode": "frozen, eval, inference",
            "pretrained": True,
            "downloads": "disabled with local_files_only=True",
            "device": str(device),
        },
        "protocol": {
            "lookback_bars": LOOKBACK,
            "horizon_bars": HORIZON,
            "stride_bars": STRIDE,
            "context_normalization": (
                "per-window mean/std over the 128 past bars only, then clip to [-5, 5]"
            ),
            "representation": (
                "tokenizer.encode(half=True) -> model.decode_s1 -> context[:, -1, :] "
                "(last hidden vector, d_model dimensions)"
            ),
            "ordinary_features": {
                "source": "scripts/walk_forward_baselines.py:feature_row",
                "names": FEATURE_NAMES,
                "warmup_bars": FEATURE_WARMUP,
                "past_only": True,
            },
            "standardization": (
                "mean/std fitted on training rows only, reused unchanged; clip to [-8, 8]"
            ),
            "lambda_grid": [float(x) for x in LAMBDA_GRID],
            "ridge_selection": "minimum validation terminal MAE",
            "logistic_selection": "minimum validation log loss",
            "final_fit": (
                "coefficients fit on training rows only after selecting lambda; "
                "validation is not reused for fitting"
            ),
            "test_use": "test labels used once, after all choices were frozen",
            "bootstrap": {
                "method": "paired circular moving-block bootstrap over chronological test windows",
                "n_boot": int(args.n_boot),
                "block_length_windows": int(block_len),
                "seed": int(args.seed),
            },
            "direction_rule": {
                "ridge_return": "r_hat > 0",
                "logistic_direction": "p_up > 0.5",
            },
            "brier_reference": "training-split up-rate (never the test base rate)",
        },
        "splits": split_info,
        "legacy_test_anchor_check": {
            "source": str(args.legacy_test_windows.resolve()),
            "n_windows": int(len(legacy)),
            "matches": True,
            "max_abs_close0_diff": close0_diff,
            "max_abs_r_actual_diff": r_diff,
            "test_first_anchor": int(anchors["test"][0]),
            "test_last_anchor": int(anchors["test"][-1]),
        },
        "selection": {
            "repr_ridge": {
                "chosen": ridge_selection["repr_ridge"][0],
                "grid": ridge_selection["repr_ridge"][1],
            },
            "feat_ridge": {
                "chosen": ridge_selection["feat_ridge"][0],
                "grid": ridge_selection["feat_ridge"][1],
            },
            "repr_logistic": {
                "chosen": logistic_selection["repr_logistic"][0],
                "grid": logistic_selection["repr_logistic"][1],
            },
            "feat_logistic": {
                "chosen": logistic_selection["feat_logistic"][0],
                "grid": logistic_selection["feat_logistic"][1],
            },
        },
        "test": {
            "n_windows": int(len(y)),
            "test_base_rate_up": float(np.mean(y_up)),
            "train_prior_up": train_prior,
            "zero_return": {
                "terminal_mae": point["zero_terminal_mae"],
                "terminal_mae_ci95": point["zero_terminal_mae_ci95"],
            },
            "training_prior": {
                "brier": point["train_prior_brier"],
                "brier_ci95": point["train_prior_brier_ci95"],
            },
            "models": point["models"],
            "paired_representation_minus_lagged": paired,
            "test_anchor_indices": [int(x) for x in anchors["test"]],
        },
        "interpretation_guardrails": [
            "Retrospective only; the same EWZ test period has informed prior design work.",
            "One asset; overlap with the original Kronos pretraining corpus is unknown.",
            "No transaction costs, slippage, or capacity analysis.",
            "A positive or negative difference here is not a fresh confirmation.",
            "The two feature sets are not claimed to be information-equivalent: the "
            "lagged features use their standard 169-bar warmup, while the frozen "
            "representation uses exactly the 128-bar context.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== test metrics (244 legacy anchors) ===", flush=True)
    print(
        f"zero return terminal MAE={point['zero_terminal_mae']:.6f} | "
        f"training prior Brier={point['train_prior_brier']:.6f} | "
        f"test up-rate={np.mean(y_up):.4f}",
        flush=True,
    )
    for name, m in point["models"].items():
        if m["head"] == "ridge_return":
            print(
                f"  {name:16s} termMAE={m['terminal_mae']:.6f} "
                f"Δzero={m['terminal_mae_minus_zero']:+.6f} "
                f"dir={m['direction_acc']:.4f} AUC={m['auc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  {name:16s} Brier={m['brier']:.6f} "
                f"Δprior={m['brier_minus_prior']:+.6f} "
                f"skill={m['brier_skill_vs_prior']:+.4f} "
                f"dir={m['direction_acc']:.4f} AUC={m['auc']:.4f}",
                flush=True,
            )
    print("\npaired representation minus lagged-feature:", flush=True)
    for key, val in paired.items():
        print(f"  {key:56s} {val['point']:+.6f}  CI95={val['ci95']}", flush=True)


def json_safe(obj):
    """Convert numpy scalars/arrays and non-finite floats for strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


if __name__ == "__main__":
    main()
