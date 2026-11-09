#!/usr/bin/env python3
"""Leakage-aware walk-forward baselines for EWZ hourly bars.

This script is deliberately independent of Kronos.  It asks whether simple,
deployable forecasts have any terminal/path skill on the existing EWZ hourly
series, using expanding chronological folds with validation-only model
selection.

Guarantees enforced by the code (and by tests/test_walk_forward_baselines.py):

* Features at anchor ``a`` use only bars strictly before ``a``.
* A fold's training labels end before the validation boundary and validation
  labels end before the test boundary.  Test labels are never used for fitting
  or selecting anything.
* Feature standardization is fitted on the training slice only.
* Ridge/logistic penalties and shrinkage toward zero return are selected on the
  validation slice only.  The final model is then refit on train+validation
  with those choices held fixed (documented in ``refit_policy``).
* Uncertainty bands are trailing-vol, EWMA-vol, or a moving-block bootstrap of
  validation residuals standardized by trailing vol.
* Reported confidence intervals use a moving-block bootstrap over test windows,
  not an iid bootstrap.

Everything here is retrospective: the same EWZ history and test period have
already been inspected in earlier project work.  These folds are not a fresh
out-of-sample confirmation, and overlap with the original Kronos pretraining
corpus is unknown.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/ewz_h1.csv"
DEFAULT_OUT = ROOT / "results/walk_forward_baselines_v5.json"
DEFAULT_WINDOWS = ROOT / "results/walk_forward_baselines_v5_windows.csv"

FEATURE_NAMES = [
    "ret_1", "ret_6", "ret_24", "ret_72", "ret_168",
    "vol_24", "vol_72", "vol_168", "vol_ratio",
    "range", "vol_z", "sma24", "sma168",
    "rsi14", "skew_72", "kurt_72", "hour_sin", "hour_cos", "dow",
]
FEATURE_WARMUP = 169
HORIZONS = (7, 24)
MODELS = ("unchanged", "historical_drift", "logistic", "ridge")
BANDS = ("trailing_vol", "ewma_vol", "residual_bootstrap")
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
Z90 = 1.6448536269514722
INTERVAL_ALPHA = 0.10
EPS = 1e-12


# ---------------------------------------------------------------------------
# Data and past-only features
# ---------------------------------------------------------------------------
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


def trailing_vol(df: pd.DataFrame, anchor: int, window: int = 168) -> float:
    """Past-only log-return volatility ending at bar anchor-1."""
    c = df["close"].to_numpy(dtype=float)[:anchor]
    if len(c) < 2:
        return float("nan")
    lr = np.diff(np.log(c))
    return float(np.std(lr[-min(window, len(lr)):]) + EPS)


def ewma_vol(df: pd.DataFrame, anchor: int, window: int = 168, lam: float = 0.94) -> float:
    """Past-only EWMA volatility of log returns ending at bar anchor-1."""
    c = df["close"].to_numpy(dtype=float)[:anchor]
    if len(c) < 2:
        return float("nan")
    lr = np.diff(np.log(c))[-min(window, len(c) - 1):]
    var = float(lr[0] ** 2)
    for r in lr[1:]:
        var = lam * var + (1.0 - lam) * float(r ** 2)
    return float(math.sqrt(var + EPS))


def feature_row(df: pd.DataFrame, anchor: int) -> np.ndarray:
    """Features available immediately before bar ``anchor``.

    Every slice ends at ``anchor`` (exclusive), so changing bars at or after the
    anchor cannot change this row.  This is asserted by the unit tests.
    """
    if anchor < FEATURE_WARMUP:
        raise ValueError(f"anchor {anchor} is below feature warmup {FEATURE_WARMUP}")
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    ts = df["timestamps"].iloc[anchor - 1]

    c0 = float(c[anchor - 1])
    lr = np.diff(np.log(c[:anchor]))  # last value ends at close[anchor-1]

    def ret(k: int) -> float:
        return float(np.log(c0 / c[anchor - 1 - k]))

    def vol(k: int) -> float:
        return float(np.std(lr[-k:]))

    lv = np.log1p(v[:anchor])
    v_mean = float(np.mean(lv[-169:-1]))
    v_std = float(np.std(lv[-169:-1])) + 1e-9
    vol_z = float((lv[-1] - v_mean) / v_std)

    d14 = np.diff(c[anchor - 15:anchor])
    gain = float(np.mean(np.maximum(d14, 0.0)))
    loss = float(np.mean(np.maximum(-d14, 0.0)))
    rsi = gain / (gain + loss + 1e-9)

    r72 = lr[-72:]
    mu = float(np.mean(r72))
    sd = float(np.std(r72)) + 1e-9
    skew = float(np.mean(((r72 - mu) / sd) ** 3))
    kurt = float(np.mean(((r72 - mu) / sd) ** 4))

    hour = ts.hour + ts.minute / 60.0
    row = [
        ret(1), ret(6), ret(24), ret(72), ret(168),
        vol(24), vol(72), vol(168), vol(24) / (vol(168) + 1e-9),
        float((h[anchor - 1] - l[anchor - 1]) / c0),
        vol_z,
        float(c0 / np.mean(c[anchor - 24:anchor]) - 1.0),
        float(c0 / np.mean(c[anchor - 168:anchor]) - 1.0),
        rsi, skew, kurt,
        float(np.sin(2 * np.pi * hour / 24.0)),
        float(np.cos(2 * np.pi * hour / 24.0)),
        float(ts.weekday() / 4.0),
    ]
    out = np.asarray(row, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"non-finite feature row at anchor {anchor}")
    return out


def build_xy(df: pd.DataFrame, anchors: np.ndarray, horizon: int):
    """Return features, terminal return, close0, and the actual price path."""
    anchors = np.asarray(anchors, dtype=int)
    if len(anchors) == 0:
        raise ValueError("empty anchor set")
    X = np.vstack([feature_row(df, int(a)) for a in anchors])
    c = df["close"].to_numpy(dtype=float)
    close0 = np.array([c[a - 1] for a in anchors], dtype=float)
    actual = np.vstack([c[a:a + horizon] for a in anchors])
    y = actual[:, -1] / close0 - 1.0
    return X, y, close0, actual


class FeatureScaler:
    """Standardization fitted on training rows only."""

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.clip((X - self.mean_) / self.std_, -8.0, 8.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Chronological folds
# ---------------------------------------------------------------------------
def make_folds(n_rows: int, horizon: int, n_folds: int = 5,
               min_history: int = 200, test_fraction: float = 0.30):
    """Expanding walk-forward folds; labels never cross a fold boundary."""
    if n_rows <= min_history + horizon:
        raise ValueError(f"not enough rows ({n_rows}) for horizon {horizon}")
    anchors = np.arange(min_history, n_rows - horizon + 1, horizon, dtype=int)
    if len(anchors) < 20:
        raise ValueError(f"only {len(anchors)} anchors for horizon {horizon}")
    test_start = int(math.floor(len(anchors) * (1.0 - test_fraction)))
    test_indices = np.arange(test_start, len(anchors))
    blocks = np.array_split(test_indices, n_folds)
    folds = []
    for k, block in enumerate(blocks):
        if len(block) < 2:
            raise ValueError(f"fold {k} has too few test anchors ({len(block)})")
        ts, te = int(block[0]), int(block[-1]) + 1
        block_len = te - ts
        vs = max(0, ts - block_len)
        if vs < 20:
            raise ValueError(f"fold {k} has too few training/validation anchors (vs={vs})")
        train_idx = np.arange(0, vs)
        val_idx = np.arange(vs, ts)
        test_idx = np.arange(ts, te)
        if len(train_idx) < 20 or len(val_idx) < 5:
            raise ValueError(
                f"fold {k} insufficient anchors: train={len(train_idx)}, val={len(val_idx)}"
            )
        # Explicit boundary checks on matured label endpoints.
        if anchors[train_idx[-1]] + horizon - 1 >= anchors[val_idx[0]]:
            raise AssertionError(f"fold {k}: training label crosses validation boundary")
        if anchors[val_idx[-1]] + horizon - 1 >= anchors[test_idx[0]]:
            raise AssertionError(f"fold {k}: validation label crosses test boundary")
        if anchors[test_idx[-1]] + horizon - 1 >= n_rows:
            raise AssertionError(f"fold {k}: test label extends past available data")
        folds.append({
            "fold": k,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_start_anchor": int(anchors[train_idx[0]]),
            "train_end_anchor": int(anchors[train_idx[-1]]),
            "val_start_anchor": int(anchors[val_idx[0]]),
            "val_end_anchor": int(anchors[val_idx[-1]]),
            "test_start_anchor": int(anchors[test_idx[0]]),
            "test_end_anchor": int(anchors[test_idx[-1]]),
        })
    return anchors, folds


# ---------------------------------------------------------------------------
# Small numerical models (numpy only)
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    pen = np.ones(p + 1)
    pen[0] = 0.0
    beta = np.zeros(p + 1)
    for _ in range(100):
        prob = sigmoid(Xb @ beta)
        w = prob * (1.0 - prob) + 1e-9
        grad = Xb.T @ (y - prob) - lam * pen * beta
        H = Xb.T @ (Xb * w[:, None]) + lam * np.diag(pen) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xb @ beta)


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    pen = np.ones(p + 1)
    pen[0] = 0.0
    A = Xb.T @ Xb + lam * np.diag(pen) + 1e-9 * np.eye(p + 1)
    return np.linalg.solve(A, Xb.T @ y)


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ beta


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def select_logistic(Xtr, ytr, Xva, yva,
                    lambdas=LAMBDAS, shrinkages=SHRINKAGES):
    ytr_up = (ytr > 0).astype(int)
    yva_up = (yva > 0).astype(int)
    """Validation-only penalty/probability-shrinkage/return-shrinkage choice."""
    prior = float(np.mean(ytr_up))
    best = None
    for lam in lambdas:
        beta = fit_logistic(Xtr, ytr_up, lam)
        p_raw = predict_logistic(Xva, beta)
        for alpha_p in shrinkages:
            p = prior + alpha_p * (p_raw - prior)
            ll = logloss(yva_up, p)
            if best is None or ll < best["val_logloss"]:
                best = {
                    "lambda": float(lam),
                    "shrinkage_probability": float(alpha_p),
                    "val_logloss": float(ll),
                    "p_val": p,
                }
    mu_up = float(np.mean(ytr[ytr_up == 1])) if np.any(ytr_up == 1) else 0.0
    mu_down = float(np.mean(ytr[ytr_up == 0])) if np.any(ytr_up == 0) else 0.0
    p_best = best.pop("p_val")
    r_raw_val = p_best * mu_up + (1.0 - p_best) * mu_down
    best_r = None
    for alpha_r in shrinkages:
        r = alpha_r * r_raw_val
        mae = float(np.mean(np.abs(r - yva)))
        if best_r is None or mae < best_r["val_terminal_mae"]:
            best_r = {"shrinkage_return": float(alpha_r), "val_terminal_mae": float(mae)}
    best.update(best_r)
    best["prior_train"] = prior
    best["mu_up_train"] = mu_up
    best["mu_down_train"] = mu_down
    return best


def select_ridge(Xtr, ytr, Xva, yva, lambdas=LAMBDAS, shrinkages=SHRINKAGES):
    """Validation-only penalty/shrinkage choice for a direct return model."""
    best = None
    for lam in lambdas:
        beta = fit_ridge(Xtr, ytr, lam)
        r_raw = predict_ridge(Xva, beta)
        for alpha in shrinkages:
            r = alpha * r_raw
            mae = float(np.mean(np.abs(r - yva)))
            if best is None or mae < best["val_terminal_mae"]:
                best = {
                    "lambda": float(lam),
                    "shrinkage_return": float(alpha),
                    "val_terminal_mae": float(mae),
                }
    return best


# ---------------------------------------------------------------------------
# Dependence-aware bootstrap and interval metrics
# ---------------------------------------------------------------------------
def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block sample of length n."""
    if n <= 0:
        raise ValueError("n must be positive")
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    pieces = [np.arange(s, s + block_len) % n for s in starts]
    return np.concatenate(pieces)[:n]


def block_bootstrap_ci(values: np.ndarray, stat, block_len: int,
                       n_boot: int = 2000, seed: int = 42):
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        out[i] = stat(values[idx])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def block_bootstrap_quantiles(values: np.ndarray, block_len: int,
                              n_boot: int = 2000, seed: int = 42):
    """Estimate 5%/95% residual quantiles with a moving-block bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        raise ValueError("need at least 4 residuals for a bootstrap interval")
    rng = np.random.default_rng(seed)
    qs = np.empty((n_boot, 2), dtype=float)
    for i in range(n_boot):
        idx = moving_block_indices(len(values), block_len, rng)
        qs[i] = np.percentile(values[idx], [5, 95])
    med = np.median(qs, axis=0)
    return float(med[0]), float(med[1])


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   alpha: float = INTERVAL_ALPHA) -> np.ndarray:
    width = hi - lo
    return width + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def interval_summary(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "mean_interval_score": float(np.mean(interval_score(y, lo, hi))),
        "alpha": INTERVAL_ALPHA,
    }


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_fold(df: pd.DataFrame, anchors: np.ndarray, fold: dict, horizon: int,
             seed: int, residual_boot: int = 2000):
    tr = fold["train_idx"]
    va = fold["val_idx"]
    te = fold["test_idx"]
    Xtr_raw, ytr, _, _ = build_xy(df, anchors[tr], horizon)
    Xva_raw, yva, _, _ = build_xy(df, anchors[va], horizon)
    Xte_raw, yte, close0_te, actual_te = build_xy(df, anchors[te], horizon)

    # Transform fitted on training only, used unchanged for validation/test/refit.
    scaler = FeatureScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    ytr_up = (ytr > 0).astype(int)
    yva_up = (yva > 0).astype(int)
    prior_train = float(np.mean(ytr_up))
    prior_refit = float(np.mean(np.concatenate([ytr_up, yva_up])))

    log_choice = select_logistic(Xtr, ytr, Xva, yva)
    ridge_choice = select_ridge(Xtr, ytr, Xva, yva)

    # Out-of-sample validation forecasts from train-only fits, used only for
    # residual uncertainty pools (labels are matured before the test boundary).
    val_scales = np.array([trailing_vol(df, int(a)) for a in anchors[va]], dtype=float)
    val_residuals = {}

    r_val_unchanged = np.zeros_like(yva)
    val_residuals["unchanged"] = (yva - r_val_unchanged) / np.maximum(val_scales, EPS)

    drift_train = float(np.mean(ytr))
    val_residuals["historical_drift"] = (yva - drift_train) / np.maximum(val_scales, EPS)

    beta_log_train = fit_logistic(Xtr, ytr_up, log_choice["lambda"])
    p_val_raw = predict_logistic(Xva, beta_log_train)
    p_val = prior_train + log_choice["shrinkage_probability"] * (p_val_raw - prior_train)
    r_val_raw = p_val * log_choice["mu_up_train"] + (1.0 - p_val) * log_choice["mu_down_train"]
    val_residuals["logistic"] = (
        yva - log_choice["shrinkage_return"] * r_val_raw
    ) / np.maximum(val_scales, EPS)

    beta_ridge_train = fit_ridge(Xtr, ytr, ridge_choice["lambda"])
    r_val_ridge = ridge_choice["shrinkage_return"] * predict_ridge(Xva, beta_ridge_train)
    val_residuals["ridge"] = (yva - r_val_ridge) / np.maximum(val_scales, EPS)

    # Refit on train+validation with selected hyperparameters.  Feature
    # standardization remains training-only by design.
    Xfit = np.vstack([Xtr, Xva])
    yfit = np.concatenate([ytr, yva])
    yfit_up = (yfit > 0).astype(int)
    beta_log = fit_logistic(Xfit, yfit_up, log_choice["lambda"])
    beta_ridge = fit_ridge(Xfit, yfit, ridge_choice["lambda"])

    mu_up = float(np.mean(yfit[yfit_up == 1])) if np.any(yfit_up == 1) else 0.0
    mu_down = float(np.mean(yfit[yfit_up == 0])) if np.any(yfit_up == 0) else 0.0
    p_log_raw = predict_logistic(Xte, beta_log)
    p_log = prior_refit + log_choice["shrinkage_probability"] * (p_log_raw - prior_refit)
    r_log_raw = p_log * mu_up + (1.0 - p_log) * mu_down
    r_log = log_choice["shrinkage_return"] * r_log_raw
    r_ridge = ridge_choice["shrinkage_return"] * predict_ridge(Xte, beta_ridge)
    r_drift = np.full(len(yte), float(np.mean(yfit)), dtype=float)

    test_scales = np.array([trailing_vol(df, int(a)) for a in anchors[te]], dtype=float)
    ewma_scales = np.array([ewma_vol(df, int(a)) for a in anchors[te]], dtype=float)
    residual_block = max(1, int(round(len(yva) ** (1.0 / 3.0))))

    preds = {
        "unchanged": {"r": np.zeros_like(yte), "p": np.full(len(yte), prior_refit),
                      "direction_rule": "r_hat>0"},
        "historical_drift": {"r": r_drift, "p": np.full(len(yte), prior_refit),
                             "direction_rule": "r_hat>0"},
        "logistic": {"r": r_log, "p": p_log, "direction_rule": "p_up>0.5"},
        "ridge": {"r": r_ridge, "p": np.full(len(yte), prior_refit),
                  "direction_rule": "r_hat>0"},
    }
    intervals = {}
    residual_quantiles = {}
    seed_offsets = {model: i for i, model in enumerate(MODELS)}
    for model in MODELS:
        r = preds[model]["r"]
        q_lo, q_hi = block_bootstrap_quantiles(
            val_residuals[model], residual_block,
            n_boot=residual_boot, seed=seed + seed_offsets[model],
        )
        residual_quantiles[model] = {"q05": q_lo, "q95": q_hi}
        half_trail = Z90 * np.sqrt(horizon) * test_scales
        half_ewma = Z90 * np.sqrt(horizon) * ewma_scales
        intervals[model] = {
            "trailing_vol": (r - half_trail, r + half_trail),
            "ewma_vol": (r - half_ewma, r + half_ewma),
            "residual_bootstrap": (
                r + q_lo * test_scales,
                r + q_hi * test_scales,
            ),
        }

    fold_summary = {
        "fold": fold["fold"],
        "train_anchors": int(len(tr)),
        "val_anchors": int(len(va)),
        "test_anchors": int(len(te)),
        "train_start": str(df["timestamps"].iloc[anchors[tr[0]] - 1]),
        "train_last_label": str(df["timestamps"].iloc[anchors[tr[-1]] + horizon - 1]),
        "val_start": str(df["timestamps"].iloc[anchors[va[0]] - 1]),
        "val_last_label": str(df["timestamps"].iloc[anchors[va[-1]] + horizon - 1]),
        "test_start": str(df["timestamps"].iloc[anchors[te[0]] - 1]),
        "test_last_label": str(df["timestamps"].iloc[anchors[te[-1]] + horizon - 1]),
        "prior_train_up": prior_train,
        "prior_refit_up": prior_refit,
        "logistic": log_choice,
        "ridge": ridge_choice,
        "residual_quantiles_standardized": residual_quantiles,
        "models": {},
    }
    for model in MODELS:
        r = preds[model]["r"]
        p = preds[model]["p"]
        path_fc = close0_te[:, None] * (1.0 + r[:, None])
        path_mae = float(np.mean(np.abs(path_fc - actual_te) / close0_te[:, None]))
        terminal_mae = float(np.mean(np.abs(r - yte)))
        fold_summary["models"][model] = {
            "terminal_mae": terminal_mae,
            "path_mae": path_mae,
            "direction_acc": _direction_accuracy(r, p, preds[model]["direction_rule"], yte),
        }
    return {
        "preds": preds,
        "intervals": intervals,
        "yte": yte,
        "close0": close0_te,
        "actual": actual_te,
        "test_anchors": anchors[te],
        "prior_refit": prior_refit,
        "fold_summary": fold_summary,
        "scaler": scaler,
        "choices": {"logistic": log_choice, "ridge": ridge_choice},
    }


def _direction_accuracy(r_hat, p_up, rule, y):
    actual_up = y > 0
    if rule == "p_up>0.5":
        pred_up = p_up > 0.5
    else:
        pred_up = r_hat > 0
    return float(np.mean(pred_up == actual_up))


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------
def evaluate_overall(stacked: dict, horizon: int, block_len: int, seed: int,
                     n_boot: int = 2000):
    y = stacked["yte"]
    close0 = stacked["close0"]
    actual = stacked["actual"]
    prior = stacked["prior_refit"]
    y_up = (y > 0).astype(int)
    prior_brier = float(np.mean((prior - y_up) ** 2))

    unchanged_path = np.mean(
        np.abs(close0[:, None] - actual) / close0[:, None], axis=1
    )
    unchanged_terminal = np.abs(y)
    out = {
        "n_test_windows": int(len(y)),
        "test_base_rate_up": float(np.mean(y_up)),
        "deployable_prior_up": float(np.mean(prior)),
        "always_up_direction_acc": float(np.mean(y_up == 1)),
        "always_down_direction_acc": float(np.mean(y_up == 0)),
        "brier_prior": prior_brier,
        "models": {},
    }
    for model in MODELS:
        r = stacked["models"][model]["r"]
        p = stacked["models"][model]["p"]
        path_fc = close0[:, None] * (1.0 + r[:, None])
        path_mae = np.mean(np.abs(path_fc - actual) / close0[:, None], axis=1)
        terminal_err = np.abs(r - y)
        direction = _direction_array(r, p, stacked["models"][model]["direction_rule"], y)
        brier = (p - y_up) ** 2
        skill = 1.0 - float(np.mean(brier)) / prior_brier if prior_brier > 0 else float("nan")

        model_out = {
            "direction_rule": stacked["models"][model]["direction_rule"],
            "terminal_mae": float(np.mean(terminal_err)),
            "terminal_rmse": float(np.sqrt(np.mean((r - y) ** 2))),
            "path_mae": float(np.mean(path_mae)),
            "path_rmse": float(np.sqrt(np.mean((path_fc - actual) ** 2 / close0[:, None] ** 2))),
            "direction_acc": float(np.mean(direction)),
            "brier_up": float(np.mean(brier)),
            "brier_skill_vs_train_prior": float(skill),
            "terminal_mae_diff_vs_unchanged": float(np.mean(terminal_err - unchanged_terminal)),
            "path_mae_diff_vs_unchanged": float(np.mean(path_mae - unchanged_path)),
            "terminal_mae_diff_vs_unchanged_ci95": block_bootstrap_ci(
                terminal_err - unchanged_terminal, np.mean, block_len, n_boot, seed
            ),
            "path_mae_diff_vs_unchanged_ci95": block_bootstrap_ci(
                path_mae - unchanged_path, np.mean, block_len, n_boot, seed + 1
            ),
            "direction_acc_ci95": block_bootstrap_ci(direction, np.mean, block_len, n_boot, seed + 2),
            "brier_skill_ci95": _bootstrap_brier_skill_fixed(
                p, y_up, prior, block_len, n_boot, seed + 3
            ),
            "intervals": {},
        }
        for band in BANDS:
            lo, hi = stacked["models"][model]["intervals"][band]
            model_out["intervals"][band] = interval_summary(y, lo, hi)
        out["models"][model] = model_out
    out["block_bootstrap"] = {
        "block_length_windows": int(block_len),
        "n_boot": int(n_boot),
        "method": "circular moving-block bootstrap over chronological test windows",
    }
    return out


def _direction_array(r_hat, p_up, rule, y):
    actual_up = y > 0
    pred_up = (p_up > 0.5) if rule == "p_up>0.5" else (r_hat > 0)
    return (pred_up == actual_up).astype(float)


def _bootstrap_brier_skill_fixed(p, y_up, prior, block_len, n_boot, seed):
    n = len(y_up)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        brier = float(np.mean((p[idx] - y_up[idx]) ** 2))
        base = float(np.mean((prior[idx] - y_up[idx]) ** 2))
        out[i] = 1.0 - brier / base if base > 0 else np.nan
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--windows-csv", default=str(DEFAULT_WINDOWS))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-history", type=int, default=200)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--residual-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    windows_path = Path(args.windows_csv)
    for path in (out_path, windows_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing result: {path}")
    df = load_data(Path(args.data))
    n_rows = len(df)
    result = {
        "schema": "kronos-ewz-walk-forward-baselines/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(Path(args.data).resolve()),
        "source_rows": int(n_rows),
        "source_first": str(df["timestamps"].iloc[0]),
        "source_last": str(df["timestamps"].iloc[-1]),
        "retrospective_warning": (
            "All folds are retrospective. The same EWZ history/test period was inspected in "
            "earlier project work, so these numbers are not a fresh confirmation. Overlap with "
            "the original Kronos pretraining corpus is unknown."
        ),
        "settings": {
            "horizons": list(HORIZONS),
            "n_folds": args.n_folds,
            "min_history": args.min_history,
            "test_fraction": args.test_fraction,
            "feature_names": FEATURE_NAMES,
            "feature_transform": "mean/std standardization fitted on each fold's training rows only",
            "models": list(MODELS),
            "uncertainty_bands": list(BANDS),
            "lambda_grid": list(LAMBDAS),
            "shrinkage_grid": list(SHRINKAGES),
            "refit_policy": (
                "Select lambda and shrinkage on validation only; refit coefficients on "
                "train+validation with selected hyperparameters; feature scaler remains fitted "
                "on training only; test labels are never used for fitting or selection."
            ),
            "target": "terminal close-to-close return over h bars; constant-return path for path error",
            "prior": "deployable up-rate from matured train+validation labels (never test base rate)",
            "block_bootstrap": {
                "n_boot": args.n_boot,
                "method": "circular moving-block over chronological test windows",
            },
            "interval_score_alpha": INTERVAL_ALPHA,
        },
        "horizons": {},
    }
    window_rows = []

    for horizon in HORIZONS:
        anchors, folds = make_folds(
            n_rows, horizon, n_folds=args.n_folds,
            min_history=args.min_history, test_fraction=args.test_fraction,
        )
        stacked = {
            "yte": [], "close0": [], "actual": [], "prior_refit": [],
            "models": {m: {"r": [], "p": [], "direction_rule": None, "intervals": {b: ([], []) for b in BANDS}}
                       for m in MODELS},
        }
        fold_summaries = []
        for fold in folds:
            run = run_fold(df, anchors, fold, horizon, args.seed, args.residual_boot)
            fold_summaries.append(run["fold_summary"])
            stacked["yte"].append(run["yte"])
            stacked["close0"].append(run["close0"])
            stacked["actual"].append(run["actual"])
            stacked["prior_refit"].append(np.full(len(run["yte"]), run["prior_refit"]))
            for model in MODELS:
                stacked["models"][model]["r"].append(run["preds"][model]["r"])
                stacked["models"][model]["p"].append(run["preds"][model]["p"])
                stacked["models"][model]["direction_rule"] = run["preds"][model]["direction_rule"]
                for band in BANDS:
                    lo, hi = run["intervals"][model][band]
                    stacked["models"][model]["intervals"][band][0].append(lo)
                    stacked["models"][model]["intervals"][band][1].append(hi)

            for pos, anchor in enumerate(run["test_anchors"]):
                for model in MODELS:
                    for band in BANDS:
                        lo, hi = run["intervals"][model][band]
                        window_rows.append({
                            "horizon": horizon,
                            "fold": fold["fold"],
                            "anchor_index": int(anchor),
                            "anchor_timestamp": str(df["timestamps"].iloc[int(anchor) - 1]),
                            "model": model,
                            "band": band,
                            "r_hat": float(run["preds"][model]["r"][pos]),
                            "p_up": float(run["preds"][model]["p"][pos]),
                            "actual_r": float(run["yte"][pos]),
                            "direction_correct": int(_direction_array(
                                np.array([run["preds"][model]["r"][pos]]),
                                np.array([run["preds"][model]["p"][pos]]),
                                run["preds"][model]["direction_rule"],
                                np.array([run["yte"][pos]]),
                            )[0]),
                            "interval_low": float(lo[pos]),
                            "interval_high": float(hi[pos]),
                            "covered": int(lo[pos] <= run["yte"][pos] <= hi[pos]),
                            "interval_score": float(interval_score(
                                np.array([run["yte"][pos]]),
                                np.array([lo[pos]]), np.array([hi[pos]]),
                            )[0]),
                        })

        stacked["yte"] = np.concatenate(stacked["yte"])
        stacked["close0"] = np.concatenate(stacked["close0"])
        stacked["actual"] = np.vstack(stacked["actual"])
        stacked["prior_refit"] = np.concatenate(stacked["prior_refit"])
        for model in MODELS:
            stacked["models"][model]["r"] = np.concatenate(stacked["models"][model]["r"])
            stacked["models"][model]["p"] = np.concatenate(stacked["models"][model]["p"])
            for band in BANDS:
                stacked["models"][model]["intervals"][band] = (
                    np.concatenate(stacked["models"][model]["intervals"][band][0]),
                    np.concatenate(stacked["models"][model]["intervals"][band][1]),
                )

        n_test = len(stacked["yte"])
        block_len = max(1, int(round(n_test ** (1.0 / 3.0))))
        overall = evaluate_overall(stacked, horizon, block_len, args.seed, args.n_boot)
        result["horizons"][str(horizon)] = {
            "n_anchors": int(len(anchors)),
            "test_start": str(df["timestamps"].iloc[int(anchors[folds[0]["test_idx"][0]]) - 1]),
            "test_last_label": str(df["timestamps"].iloc[
                int(anchors[folds[-1]["test_idx"][-1]]) + horizon - 1
            ]),
            "fold_summaries": fold_summaries,
            **overall,
        }
        print(f"\n=== horizon={horizon} | test windows={n_test} | folds={len(folds)} ===")
        print(f"always_up={overall['always_up_direction_acc']:.3f} "
              f"always_down={overall['always_down_direction_acc']:.3f} "
              f"prior_up={overall['deployable_prior_up']:.3f}")
        for model in MODELS:
            m = overall["models"][model]
            print(
                f"  {model:16s} termMAE={m['terminal_mae']:.5f} pathMAE={m['path_mae']:.5f} "
                f"dir={m['direction_acc']:.3f} brier_skill={m['brier_skill_vs_train_prior']:+.3f} "
                f"termΔvsRW={m['terminal_mae_diff_vs_unchanged']:+.5f} "
                f"pathΔvsRW={m['path_mae_diff_vs_unchanged']:+.5f}"
            )
            for band in BANDS:
                b = m["intervals"][band]
                print(
                    f"    {band:18s} cov90={b['coverage']:.3f} width={b['mean_width']:.5f} "
                    f"interval_score={b['mean_interval_score']:.5f}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    pd.DataFrame(window_rows).to_csv(windows_path, index=False)
    print(f"\nwrote {out_path}\nwrote {windows_path}")


if __name__ == "__main__":
    main()
