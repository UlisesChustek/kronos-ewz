#!/usr/bin/env python3
"""Compare the fixed v5 pilot with saved forecasts on identical retrospective targets.

Sampling is fixed at T=1, top_p=.9, seed42 for the new v5/reference runs.
Legacy v3 used its own validation-selected sampling settings; this is a recipe
comparison, not a controlled attribution to any single change.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from walk_forward_baselines import block_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]

def auc(p, y):
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/v5_pilot_comparison.json')
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    specs = {
        'v5_frozen_128_7': ROOT/'results/eval_v5_test_7bar_raw.windows.csv',
        'v3_legacy_512_7': ROOT/'results/eval_v3_test_7_cal.windows.csv',
        'pretrained_small_128_7': ROOT/'results/eval_v5_reference_zeroshot_test_7bar_raw.windows.csv',
    }
    frames = {k: pd.read_csv(p) for k, p in specs.items()}
    candidate = frames['v5_frozen_128_7']
    for k, frame in frames.items():
        if len(frame) != len(candidate):
            raise ValueError(f'{k}: different target count')
        for col in ('close0', 'r_actual', 'rw_mae_norm'):
            np.testing.assert_allclose(candidate[col], frame[col], rtol=1e-9, atol=1e-11,
                                       err_msg=f'{k}: targets/order differ for {col}')
    data = pd.read_csv(ROOT/'data/processed/ewz_h1.csv')
    c = data.close.to_numpy()
    train_end = int(len(c)*.8)
    anchors = np.arange(128, train_end-7+1)
    train_prior = float(np.mean(c[anchors+6] > c[anchors-1]))
    actual = candidate.r_actual.to_numpy()
    y = actual > 0
    prior_losses = (train_prior-y)**2
    rw = candidate.rw_mae_norm.to_numpy()
    block = max(1, round(len(candidate)**(1/3)))
    def ci(d):
        return block_bootstrap_ci(np.asarray(d), np.mean, block,
                                  n_boot=args.bootstrap, seed=42)
    result = {
        'scope': 'Retrospective single split; already inspected during design. No fresh confirmation.',
        'sampling_note': 'New runs fixed T=1 top_p=.9 seed42 samples64; legacy settings differ.',
        'training_note': 'Multiple recipe changes jointly; causal attribution requires separate ablations.',
        'n_test_windows': len(candidate), 'horizon_bars': 7,
        'train_prior_up': train_prior, 'always_up_accuracy': float(y.mean()),
        'always_down_accuracy': float((~y).mean()),
        'confidence_method': f'Circular moving-block bootstrap, block={block}, draws={args.bootstrap}; window uncertainty only, conditional on these runs.',
        'unchanged': {'path_mae': float(rw.mean()), 'terminal_mae': float(np.abs(actual).mean()),
                      'brier_training_prior': float(prior_losses.mean())},
        'models': {},
    }
    for name, frame in frames.items():
        p = frame.p_up_raw.to_numpy()
        terminal = np.abs(frame.median_r.to_numpy()-actual)
        delta = frame.mae_norm.to_numpy()-rw
        result['models'][name] = {
            'source': str(specs[name].relative_to(ROOT)),
            'path_mae': float(frame.mae_norm.mean()),
            'path_mae_ratio_vs_unchanged': float(frame.mae_norm.mean()/rw.mean()),
            'path_mae_delta_vs_unchanged_ci95': ci(delta),
            'terminal_mae': float(terminal.mean()),
            'terminal_mae_delta_vs_unchanged_ci95': ci(terminal-np.abs(actual)),
            'direction_accuracy': float(np.mean(np.sign(frame.median_r)==np.sign(actual))),
            'auc_raw_p_up': auc(p,y),
            'brier_raw': float(np.mean((p-y)**2)),
            'brier_skill_vs_training_prior': float(1-np.mean((p-y)**2)/prior_losses.mean()),
            'raw_pointwise_90_coverage': float(frame.coverage_90_raw.mean()),
        }
    legacy = frames['v3_legacy_512_7']
    result['v5_vs_v3'] = {
        'relative_path_mae_reduction': float(1-candidate.mae_norm.mean()/legacy.mae_norm.mean()),
        'paired_path_mae_delta_ci95': ci(candidate.mae_norm.to_numpy()-legacy.mae_norm.to_numpy()),
    }
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
