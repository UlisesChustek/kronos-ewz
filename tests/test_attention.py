"""Causality tests for the RoPE cross-attention used by the predictor."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Kronos"))

from model.module import MultiHeadCrossAttentionWithRoPE  # noqa: E402


class CrossAttentionCausalityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.attn = MultiHeadCrossAttentionWithRoPE(
            d_model=16, n_heads=2, attn_dropout_p=0.0, resid_dropout=0.0
        )
        self.attn.eval()
        self.x = torch.randn(2, 12, 16)

    def test_future_bars_cannot_change_past_full_sequence_outputs(self):
        with torch.no_grad():
            before = self.attn(self.x, self.x, self.x)
            future = self.x.clone()
            future[:, -1, :] += 5.0
            after = self.attn(future, future, future)
        # Token at position t only attends to positions <= t.  Perturbing the
        # final position must leave every earlier output unchanged.
        torch.testing.assert_close(after[:, :-1], before[:, :-1], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(after[:, -1], before[:, -1]))

    def test_single_query_decode_attends_to_full_history(self):
        query = self.x[:, -1:, :]  # q_len == 1 < seq_len
        with torch.no_grad():
            before = self.attn(query, self.x, self.x)
            history = self.x.clone()
            history[:, 0, :] += 3.0
            after = self.attn(query, history, history)
        # A one-query decode must be able to see the earliest history; a
        # collapsed or mistakenly causal mask would hide it.
        self.assertEqual(tuple(before.shape), (2, 1, 16))
        self.assertFalse(torch.allclose(after, before))

    def test_padding_mask_is_accepted(self):
        mask = torch.zeros(2, 12)
        mask[:, -2:] = 1.0  # 1 == ignore for this module
        with torch.no_grad():
            out = self.attn(self.x, self.x, self.x, key_padding_mask=mask)
        self.assertTrue(torch.isfinite(out).all())

    def test_padding_mask_does_not_break_causality(self):
        mask = torch.zeros(2, 12)
        with torch.no_grad():
            before = self.attn(self.x, self.x, self.x, key_padding_mask=mask)
            future = self.x.clone()
            future[:, -1, :] += 5.0
            after = self.attn(future, future, future, key_padding_mask=mask)
        torch.testing.assert_close(after[:, :-1], before[:, :-1], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(after[:, -1], before[:, -1]))


if __name__ == "__main__":
    unittest.main()
