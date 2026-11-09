"""Leakage and selection tests for the walk-forward baseline benchmark."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_baselines as wf  # noqa: E402


def synthetic_frame(n: int = 1400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Mild mean reversion keeps the series non-degenerate but predictable only
    # weakly, which is exactly the regime this benchmark is about.
    shocks = rng.normal(0.0, 0.004, size=n)
    close = np.empty(n)
    close[0] = 30.0
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(shocks[i] - 0.02 * shocks[i - 1])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0, 0.0008, size=n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0, 0.0008, size=n)))
    volume = rng.integers(1_000, 80_000, size=n).astype(float)
    return pd.DataFrame({
        "timestamps": pd.date_range("2020-01-01", periods=n, freq="h"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": volume * close,
    })


class FoldBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_frame()
        self.anchors, self.folds = wf.make_folds(1400, horizon=5, n_folds=3, min_history=200)

    def test_folds_are_expanding_and_own_test_block_is_holdout(self):
        previous_train_end = -1
        for fold in self.folds:
            self.assertEqual(int(fold["train_idx"][0]), 0)
            self.assertGreater(int(fold["train_idx"][-1]), previous_train_end)
            previous_train_end = int(fold["train_idx"][-1])
            own_test = set(int(i) for i in fold["test_idx"])
            self.assertFalse(own_test & set(int(i) for i in fold["train_idx"]))
            self.assertFalse(own_test & set(int(i) for i in fold["val_idx"]))
            self.assertTrue(np.all(np.diff(fold["test_idx"]) == 1))

    def test_matured_labels_do_not_cross_boundaries(self):
        horizon = 5
        for fold in self.folds:
            tr, va, te = fold["train_idx"], fold["val_idx"], fold["test_idx"]
            self.assertLess(
                int(self.anchors[tr[-1]]) + horizon - 1,
                int(self.anchors[va[0]]),
            )
            self.assertLess(
                int(self.anchors[va[-1]]) + horizon - 1,
                int(self.anchors[te[0]]),
            )
            self.assertLessEqual(
                int(self.anchors[te[-1]]) + horizon - 1,
                len(self.df) - 1,
            )

    def test_later_folds_may_reuse_earlier_test_blocks(self):
        # This is documented expanding-window behaviour, not leakage: only the
        # current fold's own test block is held out.
        self.assertGreaterEqual(
            int(self.folds[-1]["train_idx"][-1]),
            int(self.folds[0]["test_idx"][-1]),
        )

    def test_too_little_data_is_rejected(self):
        with self.assertRaises(ValueError):
            wf.make_folds(120, horizon=24, n_folds=5, min_history=200)


class FeatureLeakageTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_frame()
        self.anchor = 500

    def test_features_ignore_bars_at_or_after_anchor(self):
        before = wf.feature_row(self.df, self.anchor)
        mutated = self.df.copy()
        mutated.loc[self.anchor:, ["open", "high", "low", "close", "volume", "amount"]] *= 3.0
        after = wf.feature_row(mutated, self.anchor)
        np.testing.assert_array_equal(before, after)

    def test_features_use_the_bar_before_the_anchor(self):
        before = wf.feature_row(self.df, self.anchor)
        mutated = self.df.copy()
        mutated.loc[self.anchor - 1, "close"] *= 1.01
        after = wf.feature_row(mutated, self.anchor)
        self.assertFalse(np.allclose(before, after))

    def test_trailing_volatility_is_past_only(self):
        before = wf.trailing_vol(self.df, self.anchor)
        mutated = self.df.copy()
        mutated.loc[self.anchor:, "close"] *= 2.0
        self.assertEqual(before, wf.trailing_vol(mutated, self.anchor))

    def test_target_path_starts_at_anchor(self):
        anchors = np.array([self.anchor, self.anchor + 5])
        X, y, close0, actual = wf.build_xy(self.df, anchors, horizon=5)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(actual.shape, (2, 5))
        np.testing.assert_allclose(close0[0], self.df["close"].iloc[self.anchor - 1])
        np.testing.assert_allclose(actual[0], self.df["close"].iloc[self.anchor:self.anchor + 5].to_numpy())
        np.testing.assert_allclose(y[0], actual[0, -1] / close0[0] - 1.0)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.Xtr = rng.normal(size=(200, 6))
        self.Xva = rng.normal(size=(80, 6))
        self.ytr = 0.01 * (self.Xtr[:, 0] + rng.normal(size=200))
        self.yva = 0.01 * (self.Xva[:, 0] + rng.normal(size=80))

    def test_ridge_choice_minimizes_validation_mae(self):
        choice = wf.select_ridge(self.Xtr, self.ytr, self.Xva, self.yva)
        recomputed = []
        for lam in wf.LAMBDAS:
            beta = wf.fit_ridge(self.Xtr, self.ytr, lam)
            raw = wf.predict_ridge(self.Xva, beta)
            for alpha in wf.SHRINKAGES:
                recomputed.append(float(np.mean(np.abs(alpha * raw - self.yva))))
        self.assertAlmostEqual(choice["val_terminal_mae"], min(recomputed), places=12)
        self.assertIn(choice["lambda"], wf.LAMBDAS)
        self.assertIn(choice["shrinkage_return"], wf.SHRINKAGES)

    def test_logistic_choice_minimizes_validation_logloss(self):
        choice = wf.select_logistic(self.Xtr, self.ytr, self.Xva, self.yva)
        self.assertAlmostEqual(choice["prior_train"], float(np.mean(self.ytr > 0)))
        recomputed = []
        for lam in wf.LAMBDAS:
            beta = wf.fit_logistic(self.Xtr, (self.ytr > 0).astype(int), lam)
            raw = wf.predict_logistic(self.Xva, beta)
            for alpha in wf.SHRINKAGES:
                p = choice["prior_train"] + alpha * (raw - choice["prior_train"])
                recomputed.append(wf.logloss((self.yva > 0).astype(int), p))
        self.assertAlmostEqual(choice["val_logloss"], min(recomputed), places=12)


class BootstrapAndIntervalTest(unittest.TestCase):
    def test_moving_block_indices_length_and_seed(self):
        a = wf.moving_block_indices(50, 7, np.random.default_rng(1))
        b = wf.moving_block_indices(50, 7, np.random.default_rng(1))
        self.assertEqual(len(a), 50)
        self.assertTrue(np.all((a >= 0) & (a < 50)))
        np.testing.assert_array_equal(a, b)

    def test_interval_score_and_summary(self):
        y = np.array([0.0, 2.0])
        lo = np.array([-1.0, -1.0])
        hi = np.array([1.0, 1.0])
        scores = wf.interval_score(y, lo, hi, alpha=0.10)
        # width 2, plus (2/0.1) * exceedance for the second point.
        np.testing.assert_allclose(scores, [2.0, 22.0])
        summary = wf.interval_summary(y, lo, hi)
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["mean_width"], 2.0)
        self.assertAlmostEqual(summary["mean_interval_score"], 12.0)


class NoTestLabelLeakageTest(unittest.TestCase):
    def test_changing_a_final_test_label_does_not_change_predictions(self):
        df = synthetic_frame()
        anchors, folds = wf.make_folds(len(df), horizon=5, n_folds=3, min_history=200)
        fold = folds[0]
        run_a = wf.run_fold(df, anchors, fold, horizon=5, seed=42, residual_boot=20)

        last_anchor = int(anchors[fold["test_idx"][-1]])
        label_row = last_anchor + 5 - 1
        mutated = df.copy()
        mutated.loc[label_row, "close"] *= 1.25

        run_b = wf.run_fold(mutated, anchors, fold, horizon=5, seed=42, residual_boot=20)

        for model in wf.MODELS:
            np.testing.assert_array_equal(
                run_a["preds"][model]["r"], run_b["preds"][model]["r"],
                err_msg=f"{model} point forecast changed after editing a test label",
            )
            np.testing.assert_array_equal(
                run_a["preds"][model]["p"], run_b["preds"][model]["p"],
            )
            for band in wf.BANDS:
                np.testing.assert_array_equal(
                    run_a["intervals"][model][band][0],
                    run_b["intervals"][model][band][0],
                )
                np.testing.assert_array_equal(
                    run_a["intervals"][model][band][1],
                    run_b["intervals"][model][band][1],
                )
        self.assertFalse(np.array_equal(run_a["yte"], run_b["yte"]))
        self.assertEqual(run_a["choices"], run_b["choices"])


if __name__ == "__main__":
    unittest.main()
