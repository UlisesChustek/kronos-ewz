#!/usr/bin/env python3
"""Attribution diagnostics for the EWZ/Kronos result.

Question: is there conditional signal in EWZ at 1-5 day horizons that a simple
model can find? If yes, Kronos's failure is model/objective-specific. If no,
the constraint is the asset/horizon, not just the generative model.

Runs on CPU, uses the exact same chronological split and window anchors as
scripts/evaluate.py, and evaluates on the held-out test windows only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/ewz_h1.csv"
CSV = {
    24: ROOT / "results/eval_v4_test_24_cal.windows.csv",
    7: ROOT / "results/eval_v3_test_7_cal.windows.csv",
}
LOOKBACK = 512
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42


# ----------------------------------------------------------------------------
# data + features
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


FEATURE_NAMES = [
    "ret_1", "ret_6", "ret_24", "ret_72", "ret_168",
    "vol_24", "vol_72", "vol_168", "vol_ratio",
    "range", "vol_z", "sma24", "sma168",
    "rsi14", "skew_72", "kurt_72", "hour_sin", "hour_cos", "dow",
]


def feature_row(df: pd.DataFrame, anchor: int) -> np.ndarray:
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    ts = df["timestamps"].iloc[anchor - 1]

    c0 = c[anchor - 1]
    lr = np.diff(np.log(c[:anchor]))  # last element = 1-bar return into close0

    def ret(k):
        return float(np.log(c0 / c[anchor - 1 - k]))

    def vol(k):
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
    return np.asarray(row, dtype=float)


def build_dataset(df: pd.DataFrame, anchors: list[int], pred_len: int):
    X = np.vstack([feature_row(df, a) for a in anchors])
    c = df["close"].to_numpy(dtype=float)
    y = np.array([c[a + pred_len - 1] / c[a - 1] - 1.0 for a in anchors])
    return X, y


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks = pd.Series(score).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def accuracy(pred_r: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sign(pred_r) == np.sign(y)))


def const_mae(pred_r: np.ndarray, actual_close: np.ndarray, close0: np.ndarray) -> float:
    """MAE of a constant forecast at close0*(1+pred_r), normalized by close0."""
    fc = close0[:, None] * (1.0 + pred_r[:, None])
    return float(np.mean(np.abs(fc - actual_close) / close0[:, None]))


def bootstrap_ci(values: np.ndarray, stat, n_boot: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = stat(values[idx])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def boot_auc(score: np.ndarray, y: np.ndarray):
    idxs = np.arange(len(score))

    def stat(_):
        return auc(score, y)

    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(2000):
        idx = rng.integers(0, len(score), len(score))
        out.append(auc(score[idx], y[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def model_diagnostics(csv_path: Path) -> dict:
    d = pd.read_csv(csv_path)
    y = (d.r_actual > 0).to_numpy()
    base = float(y.mean())
    p = d.p_up.to_numpy()
    binom = float(np.sqrt(base * (1 - base) / d.n_samples.iloc[0]))
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    return {
        "n": int(len(d)),
        "base_up": base,
        "pred_down": float((d.median_r < 0).mean()),
        "dir_acc": float((np.sign(d.median_r) == np.sign(d.r_actual)).mean()),
        "const_down_acc": float(1 - base),
        "const_up_acc": base,
        "auc_p_up": auc(p, y),
        "auc_median_r": auc(d.median_r.to_numpy(), y),
        "auc_median_r_neg": auc(-d.median_r.to_numpy(), y),
        "corr_median_actual": float(np.corrcoef(d.median_r, d.r_actual)[0, 1]),
        "mean_median_r": float(d.median_r.mean()),
        "mean_actual_r": float(d.r_actual.mean()),
        "p_up_mean": float(p.mean()),
        "p_up_std": float(p.std()),
        "binom_std": binom,
        "brier": brier,
        "brier_base": brier_base,
        "brier_skill": float(1 - brier / brier_base),
        "mae": float(d.mae_norm.mean()),
        "rw_mae": float(d.rw_mae_norm.mean()),
        "mae_diff": float((d.mae_norm - d.rw_mae_norm).mean()),
        "beat_rw": float((d.mae_norm < d.rw_mae_norm).mean()),
    }


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    pen = np.ones(p + 1)
    pen[0] = 0.0
    beta = np.zeros(p + 1)
    for _ in range(100):
        prob = sigmoid(Xb @ beta)
        w = prob * (1 - prob) + 1e-9
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


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    pen = np.ones(p + 1)
    pen[0] = 0.0
    A = Xb.T @ Xb + lam * np.diag(pen) + 1e-9 * np.eye(p + 1)
    return np.linalg.solve(A, Xb.T @ y)


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ beta


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic_cv(X, y):
    # chronological 80/20 split of the training data
    cut = int(len(X) * 0.8)
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:]
    best = None
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        beta = fit_logistic(Xtr, ytr, lam)
        ll = logloss(yva, predict_logistic(Xva, beta))
        if best is None or ll < best[0]:
            best = (ll, lam)
    lam = best[1]
    return fit_logistic(X, y, lam), lam, best[0]


def fit_ridge_cv(X, y):
    cut = int(len(X) * 0.8)
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:]
    best = None
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        beta = fit_ridge(Xtr, ytr, lam)
        mse = float(np.mean((predict_ridge(Xva, beta) - yva) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam)
    lam = best[1]
    return fit_ridge(X, y, lam), lam, best[0]


def signal_forecast(tr_signal, tr_y, te_signal):
    """Two-bin conditional mean: predict the train mean return for each sign."""
    hi = tr_y[tr_signal > 0]
    lo = tr_y[tr_signal <= 0]
    mu_hi = float(np.mean(hi)) if len(hi) else 0.0
    mu_lo = float(np.mean(lo)) if len(lo) else 0.0
    return np.where(te_signal > 0, mu_hi, mu_lo)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def evaluate_pred_len(df: pd.DataFrame, pred_len: int) -> dict:
    n = len(df)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_anchors = list(range(max(val_end, LOOKBACK), n - pred_len + 1, pred_len))
    train_anchors = list(range(LOOKBACK + 1, val_end - pred_len + 1, pred_len))

    Xtr, ytr = build_dataset(df, train_anchors, pred_len)
    Xte, yte = build_dataset(df, test_anchors, pred_len)

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Xtr = np.clip((Xtr - mu) / sd, -8, 8)
    Xte = np.clip((Xte - mu) / sd, -8, 8)

    c = df["close"].to_numpy(dtype=float)
    close0 = np.array([c[a - 1] for a in test_anchors])
    actual_close = np.vstack([c[a:a + pred_len] for a in test_anchors])
    y_up = (yte > 0).astype(int)

    # fit (training labels only; test labels are never seen by the fit)
    y_tr_up = (ytr > 0).astype(int)
    beta_log, lam_log, ll_val = fit_logistic_cv(Xtr, y_tr_up)
    p_te = predict_logistic(Xte, beta_log)

    beta_ridge, lam_ridge, mse_val = fit_ridge_cv(Xtr, ytr)
    r_te = predict_ridge(Xte, beta_ridge)

    # momentum/reversal from raw features (before standardization)
    raw_tr = build_dataset(df, train_anchors, pred_len)[0]
    raw_te = build_dataset(df, test_anchors, pred_len)[0]
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def add_signal(name, tr_sig, te_sig, direction=1):
        f_tr = direction * tr_sig
        f_te = direction * te_sig
        pred_r = signal_forecast(f_tr, ytr, f_te)
        out = {
            "auc": auc(f_te, y_up),
            "auc_ci": boot_auc(f_te, y_up),
            "acc": accuracy(f_te, yte),
            "acc_ci": bootstrap_ci((np.sign(f_te) == np.sign(yte)).astype(float), np.mean),
            "mae": const_mae(pred_r, actual_close, close0),
            "mean_pred_r": float(np.mean(pred_r)),
        }
        return out

    baselines = {}
    baselines["momentum_24"] = add_signal("momentum_24", raw_tr[:, idx["ret_24"]], raw_te[:, idx["ret_24"]])
    baselines["momentum_168"] = add_signal("momentum_168", raw_tr[:, idx["ret_168"]], raw_te[:, idx["ret_168"]])
    baselines["reversal_24"] = add_signal("reversal_24", raw_tr[:, idx["ret_24"]], raw_te[:, idx["ret_24"]], -1)
    baselines["reversal_6"] = add_signal("reversal_6", raw_tr[:, idx["ret_6"]], raw_te[:, idx["ret_6"]], -1)

    # constant / drift baselines
    train_drift = float(np.mean(ytr))
    test_drift = float(np.mean(yte))
    for name, pred_r in [
        ("random_walk", np.zeros(len(yte))),
        ("train_drift", np.full(len(yte), train_drift)),
        ("oracle_drift", np.full(len(yte), test_drift)),
    ]:
        baselines[name] = {
            "auc": 0.5 if name != "oracle_drift" else float("nan"),
            "auc_ci": [float("nan"), float("nan")],
            "acc": float(np.mean(np.sign(pred_r) == np.sign(yte))),
            "acc_ci": bootstrap_ci((np.sign(pred_r) == np.sign(yte)).astype(float), np.mean),
            "mae": const_mae(pred_r, actual_close, close0),
            "mean_pred_r": float(np.mean(pred_r)),
        }

    # learned baselines
    baselines["logistic"] = {
        "auc": auc(p_te, y_up),
        "auc_ci": boot_auc(p_te, y_up),
        "acc": accuracy(p_te - 0.5, yte),
        "acc_ci": bootstrap_ci(((p_te > 0.5) == (yte > 0)).astype(float), np.mean),
        "mae": const_mae(np.where(p_te > 0.5, train_drift, -train_drift), actual_close, close0),
        "mean_pred_r": float(np.mean(np.where(p_te > 0.5, train_drift, -train_drift))),
        "lam": lam_log,
        "val_logloss": ll_val,
        "coef": {name: float(b) for name, b in zip(["intercept"] + FEATURE_NAMES, beta_log)},
    }
    baselines["ridge"] = {
        "auc": auc(r_te, y_up),
        "auc_ci": boot_auc(r_te, y_up),
        "acc": accuracy(r_te, yte),
        "acc_ci": bootstrap_ci((np.sign(r_te) == np.sign(yte)).astype(float), np.mean),
        "mae": const_mae(r_te, actual_close, close0),
        "mean_pred_r": float(np.mean(r_te)),
        "lam": lam_ridge,
        "val_mse": mse_val,
        "coef": {name: float(b) for name, b in zip(["intercept"] + FEATURE_NAMES, beta_ridge)},
    }

    return {
        "pred_len": pred_len,
        "n_train": len(train_anchors),
        "n_test": len(test_anchors),
        "test_start": str(df["timestamps"].iloc[test_anchors[0] - 1]),
        "test_end": str(df["timestamps"].iloc[test_anchors[-1] + pred_len - 1]),
        "base_up": float(y_up.mean()),
        "train_drift": train_drift,
        "test_drift": test_drift,
        "kronos": model_diagnostics(CSV[pred_len]),
        "baselines": baselines,
    }


def main() -> None:
    np.random.seed(SEED)
    df = load_data()
    out = {
        "data": str(DATA),
        "lookback": LOOKBACK,
        "split": f"{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/10% chronological",
        "feature_names": FEATURE_NAMES,
        "results": {},
    }
    for pred_len in (24, 7):
        print(f"\n=== pred_len={pred_len} ===")
        r = evaluate_pred_len(df, pred_len)
        out["results"][str(pred_len)] = r
        k = r["kronos"]
        print(f"windows={r['n_test']} base_up={r['base_up']:.3f} train_drift={r['train_drift']:+.4f} test_drift={r['test_drift']:+.4f}")
        print(f"Kronos: dir={k['dir_acc']:.3f} const_down={k['const_down_acc']:.3f} "
              f"AUC(p_up)={k['auc_p_up']:.3f} corr={k['corr_median_actual']:+.3f} "
              f"MAE={k['mae']:.5f} RW={k['rw_mae']:.5f}")
        for name, b in r["baselines"].items():
            if name in ("logistic", "ridge"):
                print(f"  {name:12s} AUC={b['auc']:.3f} {b['auc_ci']} acc={b['acc']:.3f} "
                      f"MAE={b['mae']:.5f} lam={b['lam']}")
            elif name.startswith("momentum") or name.startswith("reversal"):
                print(f"  {name:12s} AUC={b['auc']:.3f} {b['auc_ci']} acc={b['acc']:.3f} MAE={b['mae']:.5f}")
            else:
                print(f"  {name:12s} acc={b['acc']:.3f} MAE={b['mae']:.5f} pred_r={b['mean_pred_r']:+.4f}")

    out_path = ROOT / "results/why_baselines.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
