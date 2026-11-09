"""Tests for loss-mask, accumulation, teacher-forcing, and run-path helpers."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Kronos"))
sys.path.insert(0, str(ROOT / "third_party/Kronos/finetune_csv"))

from finetune_base_model import (  # noqa: E402
    _model_forward_loss,
    accumulation_group_size,
    build_predict_only_loss_mask,
    count_valid_targets,
    ensure_run_path_available,
    optimizer_steps_per_epoch,
)


class LossMaskTest(unittest.TestCase):
    def test_mask_selects_only_forecast_targets(self):
        lookback, predict = 8, 4
        target_len = lookback + predict - 1
        mask = build_predict_only_loss_mask(3, target_len, lookback, predict)
        self.assertEqual(tuple(mask.shape), (3, target_len))
        # 0 == valid, 1 == ignored. Token target j is bar j+1, so the forecast
        # bars [lookback, lookback+predict) are targets [lookback-1, target_len).
        expected_zeros = set(range(lookback - 1, target_len))
        for row in range(3):
            for j in range(target_len):
                self.assertEqual(float(mask[row, j]), 0.0 if j in expected_zeros else 1.0)
        self.assertEqual(count_valid_targets(mask), 3 * predict)

    def test_mask_rejects_misaligned_target_length(self):
        with self.assertRaises(ValueError):
            build_predict_only_loss_mask(2, 12, 8, 4)  # should be 11
        with self.assertRaises(ValueError):
            build_predict_only_loss_mask(2, 10, 8, 4)

    def test_count_valid_targets_none(self):
        self.assertIsNone(count_valid_targets(None))


class AccumulationTest(unittest.TestCase):
    def test_optimizer_steps_use_ceiling(self):
        self.assertEqual(optimizer_steps_per_epoch(10, 3), 4)
        self.assertEqual(optimizer_steps_per_epoch(9, 3), 3)
        self.assertEqual(optimizer_steps_per_epoch(10, 1), 10)
        self.assertEqual(optimizer_steps_per_epoch(10, 0), 10)

    def test_partial_final_group_is_scaled_correctly(self):
        sizes = [accumulation_group_size(i, 10, 3) for i in range(10)]
        self.assertEqual(sizes, [3, 3, 3, 3, 3, 3, 3, 3, 3, 1])
        # A group of one should not divide by the nominal accumulation size.
        self.assertEqual(accumulation_group_size(9, 10, 3), 1)
        with self.assertRaises(IndexError):
            accumulation_group_size(10, 10, 3)


class _FakeHead:
    def __init__(self):
        self.calls = None

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        self.calls = {
            "s1_targets": s1_targets,
            "s2_targets": s2_targets,
            "padding_mask": padding_mask,
        }
        return torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)


class _FakeModel:
    def __init__(self):
        self.head = _FakeHead()
        self.calls = None

    def __call__(self, s1_in, s2_in, stamps, use_teacher_forcing, s1_targets):
        self.calls = {
            "s1_in": s1_in,
            "s2_in": s2_in,
            "stamps": stamps,
            "use_teacher_forcing": use_teacher_forcing,
            "s1_targets": s1_targets,
        }
        # Logit tensors only need to be passed through to the fake head.
        return ("s1_logits", "s2_logits")


class TeacherForcingTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.s1_in = torch.zeros(2, 5, dtype=torch.long)
        self.s2_in = torch.zeros(2, 5, dtype=torch.long)
        self.s1_out = torch.arange(10).reshape(2, 5)
        self.s2_out = torch.arange(10).reshape(2, 5)
        self.stamps = torch.zeros(2, 5, 5)
        self.mask = torch.zeros(2, 5)

    def test_teacher_forcing_conditions_on_target_s1(self):
        _model_forward_loss(
            self.model,
            [self.s1_in, self.s2_in],
            [self.s1_out, self.s2_out],
            self.stamps,
            self.mask,
            teacher_forcing=True,
        )
        self.assertTrue(self.model.calls["use_teacher_forcing"])
        self.assertIs(self.model.calls["s1_targets"], self.s1_out)
        self.assertIs(self.model.head.calls["s1_targets"], self.s1_out)
        self.assertIs(self.model.head.calls["padding_mask"], self.mask)

    def test_wrapped_model_forward_is_used_and_head_is_unwrapped(self):
        class Wrapper:
            def __init__(self, module):
                self.module = module
                self.calls = 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                return self.module(*args, **kwargs)
        wrapper = Wrapper(self.model)
        _model_forward_loss(
            wrapper, [self.s1_in, self.s2_in], [self.s1_out, self.s2_out],
            self.stamps, self.mask, teacher_forcing=True,
        )
        self.assertEqual(wrapper.calls, 1)
        self.assertIs(self.model.head.calls["padding_mask"], self.mask)

    def test_sampling_mode_passes_no_targets(self):
        _model_forward_loss(
            self.model,
            [self.s1_in, self.s2_in],
            [self.s1_out, self.s2_out],
            self.stamps,
            self.mask,
            teacher_forcing=False,
        )
        self.assertFalse(self.model.calls["use_teacher_forcing"])
        self.assertIsNone(self.model.calls["s1_targets"])


class RunPathGuardTest(unittest.TestCase):
    def test_missing_and_empty_paths_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "missing"
            empty = Path(tmp) / "empty"
            empty.mkdir()
            ensure_run_path_available(str(base), str(empty), overwrite=False)

    def test_existing_run_is_refused_without_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            (base / "checkpoint.bin").write_bytes(b"x")
            other = Path(tmp) / "other"
            with self.assertRaises(FileExistsError):
                ensure_run_path_available(str(base), str(other), overwrite=False)

    def test_explicit_overwrite_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            (base / "checkpoint.bin").write_bytes(b"x")
            ensure_run_path_available(str(base), str(Path(tmp) / "other"), overwrite=True)


if __name__ == "__main__":
    unittest.main()
