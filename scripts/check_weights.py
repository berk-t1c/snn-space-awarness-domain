#!/usr/bin/env python3
"""Analyze learned weights to diagnose training issues."""

import torch
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='runs/spikeseg_20260122_080644/checkpoints/checkpoint_best.pt')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    print("=== WEIGHT ANALYSIS ===\n")

    for key in sorted(state.keys()):
        if 'weight' in key and 'conv' in key:
            w = state[key]
            print(f"{key}: shape={list(w.shape)}")
            print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

            # Check if weights are diverse or collapsed
            if len(w.shape) == 4:  # Conv weights
                # Check per-filter variance
                per_filter_std = w.view(w.shape[0], -1).std(dim=1)
                print(f"  per-filter std: min={per_filter_std.min():.4f}, max={per_filter_std.max():.4f}")

                # Check if all filters are similar (collapsed)
                filter_flat = w.view(w.shape[0], -1)
                cross_corr = torch.corrcoef(filter_flat)
                off_diag = cross_corr[~torch.eye(cross_corr.shape[0], dtype=bool)]
                print(f"  filter similarity (off-diag corr): mean={off_diag.mean():.4f}")

                # Check weight distribution
                w_flat = w.flatten()
                near_zero = (w_flat.abs() < 0.1).float().mean()
                near_one = (w_flat > 0.9).float().mean()
                print(f"  weights near 0 (<0.1): {near_zero*100:.1f}%")
                print(f"  weights near 1 (>0.9): {near_one*100:.1f}%")
            print()

    # Check convergence stats if available
    if 'convergence' in ckpt:
        print("=== CONVERGENCE STATS ===")
        conv = ckpt['convergence']
        for k, v in conv.items():
            print(f"  {k}: {v}")

    # Check epoch info
    print(f"\n=== TRAINING INFO ===")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Best metric: {ckpt.get('best_metric', 'N/A')}")

if __name__ == '__main__':
    main()
