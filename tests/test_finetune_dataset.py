"""Regression tests for the chronological predictor dataset.

Run from the repository root with::

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Kronos"))
sys.path.insert(0, str(ROOT / "third_party/Kronos/finetune_csv"))

from finetune_base_model import CustomKlineDataset  # noqa: E402


def make_frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1.0 + rng.normal(0, 0.0005, size=n))
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    volume = rng.integers(1_000, 50_000, size=n).astype(float)
    amount = volume * close
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamps": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    })


class DatasetAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "bars.csv"
        make_frame(120).to_csv(self.data_path, index=False)
        self.kwargs = dict(
            data_path=str(self.data_path),
            lookback_window=8,
            predict_window=4,
            clip=5.0,
            seed=11,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )

    def _dataset(self, split):
        return CustomKlineDataset(data_type=split, **self.kwargs)

    def test_window_has_no_extra_bar(self):
        ds = self._dataset("train")
        self.assertEqual(ds.window, ds.lookback_window + ds.predict_window)
        self.assertEqual(ds.window, 12)

    def test_train_windows_end_inside_training_split(self):
        ds = self._dataset("train")
        self.assertEqual(int(ds.start_indices[0]), 0)
        self.assertEqual(int(ds.start_indices[-1]) + ds.window, ds.train_end)
        self.assertTrue(np.all(ds.start_indices + ds.window <= ds.train_end))

    def test_validation_labels_lie_in_validation_split(self):
        ds = self._dataset("val")
        # Context may start inside training, but the forecast bars
        # [start+lookback, start+window) must be validation labels.
        forecast_starts = ds.start_indices + ds.lookback_window
        self.assertTrue(np.all(forecast_starts >= ds.train_end))
        self.assertTrue(np.all(ds.start_indices + ds.window <= ds.val_end))
        # The first window uses exactly the last `lookback` training bars as
        # context and forecasts the first validation bar.
        self.assertEqual(int(ds.start_indices[0] + ds.lookback_window), ds.train_end)
        self.assertEqual(int(ds.start_indices[-1] + ds.window), ds.val_end)

    def test_test_labels_lie_in_test_split(self):
        ds = self._dataset("test")
        self.assertTrue(np.all(ds.start_indices + ds.lookback_window >= ds.val_end))
        self.assertTrue(np.all(ds.start_indices + ds.window <= len(ds.data)))
        self.assertEqual(int(ds.start_indices[0] + ds.lookback_window), ds.val_end)

    def test_index_out_of_range_is_rejected(self):
        ds = self._dataset("val")
        with self.assertRaises(IndexError):
            _ = ds[len(ds)]
        with self.assertRaises(IndexError):
            _ = ds[-1]

    def test_epoch_order_is_deterministic_and_distinct(self):
        ds = self._dataset("train")
        first = ds._epoch_order(0)
        again = ds._epoch_order(0)
        later = ds._epoch_order(1)
        self.assertEqual(first, again)
        self.assertNotEqual(first, later)
        self.assertEqual(sorted(first), list(range(len(ds))))

    def test_normalization_uses_past_bars_only(self):
        ds = self._dataset("val")
        idx = 0
        start = int(ds.start_indices[idx])
        x, _ = ds[idx]
        past_np = x[: ds.lookback_window].numpy()

        # Rewrite only the forecast portion of this window (bars at/after the
        # anchor) and verify the normalized context is byte-for-byte unchanged.
        frame = make_frame(120)
        mutated = frame.copy()
        forecast_rows = slice(start + ds.lookback_window, start + ds.window)
        for col in ("open", "high", "low", "close", "volume", "amount"):
            mutated.loc[forecast_rows, col] = mutated.loc[forecast_rows, col] * 1.5
        second_path = Path(self.tmp.name) / "mutated.csv"
        mutated.to_csv(second_path, index=False)
        ds2 = CustomKlineDataset(data_type="val", **{**self.kwargs, "data_path": str(second_path)})
        x2, _ = ds2[idx]
        np.testing.assert_allclose(x[: ds.lookback_window].numpy(), x2[: ds.lookback_window].numpy())
        self.assertFalse(np.allclose(x[ds.lookback_window:].numpy(), x2[ds.lookback_window:].numpy()))

        # Sanity: per-feature normalization is fitted on the context.  The
        # epsilon in `(x - mean) / (std + 1e-5)` leaves std just below one.
        mean = past_np.mean(axis=0)
        std = past_np.std(axis=0)
        np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-5)
        np.testing.assert_allclose(std, np.ones_like(std), atol=5e-4)


class DatasetBoundaryRejectionTest(unittest.TestCase):
    def test_validation_split_shorter_than_forecast_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            # n=100, train_end=80, val_end=81: only one validation bar, but the
            # forecast needs seven matured bars.
            make_frame(100).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                CustomKlineDataset(
                    data_path=str(path), data_type="val",
                    lookback_window=5, predict_window=7,
                    train_ratio=0.80, val_ratio=0.01, test_ratio=0.19,
                )

    def test_invalid_ratio_total_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            make_frame(60).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                CustomKlineDataset(
                    data_path=str(path), data_type="train",
                    lookback_window=5, predict_window=2,
                    train_ratio=0.8, val_ratio=0.3, test_ratio=0.2,
                )


if __name__ == "__main__":
    unittest.main()
