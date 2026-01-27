#!/usr/bin/env python3
"""
Visualize individual spikes for a specific sample from the EBSSA dataset.

Creates a 3D animated visualization showing ground truth trajectory (blue dots)
and network detections (red stars) appearing over time.

Usage:
    python scripts/visualize_spike.py --sample 1
    python scripts/visualize_spike.py --sample 1 --checkpoint /path/to/checkpoint.pt
    python scripts/visualize_spike.py --sample 0 --output sample0_spikes.gif
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from spikeseg.data.datasets import EBSSADataset
from scripts.inference import load_model, detect_satellites, animate_3d_trajectory


def load_trajectory(mat_path: str) -> dict:
    """
    Load trajectory from MATLAB .mat file, handling 0-d structured arrays.

    Args:
        mat_path: Path to the .mat file containing Obj structure

    Returns:
        Dictionary with 'x', 'y', 't' arrays
    """
    mat = sio.loadmat(mat_path, squeeze_me=True)

    if 'Obj' not in mat:
        print(f"Warning: No 'Obj' field in {mat_path}")
        return None

    obj = mat['Obj']

    # Handle 0-d structured array (MATLAB quirk)
    trajectory = {
        'x': obj['x'].item() if obj['x'].ndim == 0 else obj['x'],
        'y': obj['y'].item() if obj['y'].ndim == 0 else obj['y'],
        't': obj['ts'].item() if obj['ts'].ndim == 0 else obj['ts']
    }

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description='Visualize individual spikes for a specific EBSSA sample'
    )
    parser.add_argument(
        '--sample', '-s', type=int, default=1,
        help='Sample index to visualize (default: 1)'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, default='/home/ubuntu/checkpoint_best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-root', '-d', type=str, default='/home/ubuntu/ebssa-data',
        help='EBSSA dataset root directory'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output file path (default: sample{N}_spikes.gif)'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=0.05,
        help='Inference threshold (default: 0.05)'
    )
    parser.add_argument(
        '--n-timesteps', type=int, default=20,
        help='Number of timesteps (default: 20)'
    )
    parser.add_argument(
        '--height', type=int, default=128,
        help='Input height (default: 128)'
    )
    parser.add_argument(
        '--width', type=int, default=128,
        help='Input width (default: 128)'
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Animation frames per second (default: 10)'
    )
    parser.add_argument(
        '--trail', type=int, default=0,
        help='Trail length for animation (0 = show all history, default: 0)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )
    parser.add_argument(
        '--no-inference-mode', action='store_true',
        help='Disable inference mode (use fire-once constraint)'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Generate static 3D plot instead of animation'
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Set default output path
    if args.output is None:
        args.output = f'sample{args.sample}_spikes.gif'

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.threshold)
    print(f"Model loaded with threshold={args.threshold}")

    # Load dataset
    print(f"Loading dataset from: {args.data_root}")
    ds = EBSSADataset(
        root=args.data_root,
        n_timesteps=args.n_timesteps,
        height=args.height,
        width=args.width,
        split='all'
    )
    print(f"Dataset size: {len(ds)} samples")

    # Validate sample index
    if args.sample < 0 or args.sample >= len(ds):
        print(f"Error: Sample index {args.sample} out of range [0, {len(ds)-1}]")
        return 1

    # Load sample
    print(f"\nLoading sample {args.sample}...")
    x, label = ds[args.sample]
    x = x.unsqueeze(1)  # Add batch dimension: (T, C, H, W) -> (T, 1, C, H, W)

    # Run detection
    inference_mode = not args.no_inference_mode
    print(f"Running detection (inference_mode={inference_mode})...")
    _, raw_spikes = detect_satellites(
        model, x, device,
        return_spikes=True,
        inference_mode=inference_mode
    )

    # Load trajectory
    print("Loading trajectory...")
    rec = ds.recordings[args.sample]
    trajectory = load_trajectory(rec['event_path'])

    if trajectory is not None:
        print(f"Trajectory: {len(trajectory['x'])} points")
        print(f"X range: {trajectory['x'].min():.1f} to {trajectory['x'].max():.1f}")
        print(f"Y range: {trajectory['y'].min():.1f} to {trajectory['y'].max():.1f}")
    else:
        print("Warning: No trajectory data available")

    # Generate visualization
    title = f'Sample {args.sample}: Satellite Detection'

    if args.static:
        # Static 3D plot
        from scripts.inference import visualize_3d_trajectory
        print(f"\nGenerating static 3D visualization...")
        output_path = args.output.replace('.gif', '.png')
        visualize_3d_trajectory(
            x, raw_spikes, label,
            trajectory=trajectory,
            output_path=output_path,
            title=title
        )
    else:
        # Animated 3D visualization
        print(f"\nGenerating animated 3D visualization...")
        print(f"  FPS: {args.fps}")
        print(f"  Trail length: {args.trail} (0 = show all)")

        animate_3d_trajectory(
            x, raw_spikes, label,
            trajectory=trajectory,
            output_path=args.output,
            title=title,
            fps=args.fps,
            trail_length=args.trail
        )

    print(f"\nDone! Output saved to: {args.output}")
    return 0


if __name__ == '__main__':
    exit(main())
