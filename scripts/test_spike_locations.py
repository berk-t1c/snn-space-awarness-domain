#!/usr/bin/env python3
"""
Test spike detection at different samples and analyze spike locations.

Usage:
    python scripts/test_spike_locations.py --checkpoint /path/to/checkpoint.pt
    python scripts/test_spike_locations.py --samples 0 1 2 --animate
    python scripts/test_spike_locations.py --all-samples --output-dir results/
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset
from scripts.inference import load_model, detect_satellites


def analyze_spikes_per_timestep(raw_spikes: torch.Tensor, sample_id: int = 0):
    """
    Analyze spikes at each timestep and print locations.

    Args:
        raw_spikes: Spike tensor of shape (T, B, C, H, W)
        sample_id: Sample ID for display
    """
    n_timesteps = raw_spikes.shape[0]

    print(f"\n=== SAMPLE {sample_id} ===")

    spike_data = []

    for t in range(n_timesteps):
        spikes_t = raw_spikes[t, 0]  # (C, H, W), batch=0
        n_spikes = int(spikes_t.sum().item())

        if n_spikes > 0:
            # Find spike locations
            locs = torch.nonzero(spikes_t, as_tuple=False)  # (N, 3) for C, H, W

            # Get center of mass
            if len(locs) > 0:
                center_y = locs[:, 1].float().mean().item()
                center_x = locs[:, 2].float().mean().item()

                # Get spread (std dev)
                std_y = locs[:, 1].float().std().item() if len(locs) > 1 else 0
                std_x = locs[:, 2].float().std().item() if len(locs) > 1 else 0

                print(f"t={t:2d}: {n_spikes:4d} spikes @ center ({center_x:.1f}, {center_y:.1f}) "
                      f"± ({std_x:.1f}, {std_y:.1f})")

                spike_data.append({
                    'timestep': t,
                    'n_spikes': n_spikes,
                    'center_x': center_x,
                    'center_y': center_y,
                    'std_x': std_x,
                    'std_y': std_y
                })
        else:
            print(f"t={t:2d}:    0 spikes")

    # Summary statistics
    if spike_data:
        centers_x = [d['center_x'] for d in spike_data]
        centers_y = [d['center_y'] for d in spike_data]

        x_range = max(centers_x) - min(centers_x)
        y_range = max(centers_y) - min(centers_y)
        total_spikes = sum(d['n_spikes'] for d in spike_data)

        print(f"\nSummary:")
        print(f"  Total spikes: {total_spikes}")
        print(f"  Timesteps with spikes: {len(spike_data)}/{n_timesteps}")
        print(f"  Center X range: {min(centers_x):.1f} to {max(centers_x):.1f} (Δ={x_range:.1f})")
        print(f"  Center Y range: {min(centers_y):.1f} to {max(centers_y):.1f} (Δ={y_range:.1f})")

        if x_range < 1 and y_range < 1 and len(spike_data) > 3:
            print(f"  ⚠️  WARNING: Spikes at same location every timestep!")
            print(f"     This may indicate the bug is still present.")
        else:
            print(f"  ✓ Spike locations vary across timesteps (expected behavior)")

    return spike_data


def create_animation(x, raw_spikes, label, trajectory, output_path, title):
    """Create animation if animate_3d_trajectory is available."""
    try:
        from scripts.inference import animate_3d_trajectory
        import scipy.io as sio

        animate_3d_trajectory(
            x, raw_spikes, label,
            trajectory=trajectory,
            output_path=output_path,
            title=title
        )
        print(f"  Animation saved to: {output_path}")
    except Exception as e:
        print(f"  Could not create animation: {e}")


def load_trajectory(event_path):
    """Load trajectory from .mat file."""
    try:
        import scipy.io as sio
        mat = sio.loadmat(event_path, squeeze_me=True)
        obj = mat['Obj']
        trajectory = {
            'x': obj['x'].item() if obj['x'].ndim == 0 else obj['x'],
            'y': obj['y'].item() if obj['y'].ndim == 0 else obj['y'],
            't': obj['ts'].item() if obj['ts'].ndim == 0 else obj['ts']
        }
        return trajectory
    except Exception as e:
        print(f"  Could not load trajectory: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test spike detection at different samples')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/ubuntu/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str,
                        default='/home/ubuntu/ebssa-data',
                        help='Path to EBSSA dataset')
    parser.add_argument('--samples', type=int, nargs='+', default=[0, 1, 2],
                        help='Sample indices to test')
    parser.add_argument('--all-samples', action='store_true',
                        help='Test all samples in dataset')
    parser.add_argument('--n-timesteps', type=int, default=20,
                        help='Number of timesteps')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Detection threshold')
    parser.add_argument('--animate', action='store_true',
                        help='Create animation GIFs')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for animations')
    parser.add_argument('--inference-mode', action='store_true', default=True,
                        help='Use inference mode (fire_once=False)')
    parser.add_argument('--no-inference-mode', action='store_true',
                        help='Disable inference mode (fire_once=True)')

    args = parser.parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.threshold)

    # Load dataset
    print(f"Loading dataset from: {args.data_root}")
    ds = EBSSADataset(
        root=args.data_root,
        n_timesteps=args.n_timesteps,
        height=128,
        width=128,
        split='all'
    )
    print(f"Dataset size: {len(ds)} samples")

    # Determine which samples to test
    if args.all_samples:
        sample_indices = list(range(len(ds)))
    else:
        sample_indices = args.samples

    # Inference mode
    inference_mode = not args.no_inference_mode
    print(f"Inference mode: {inference_mode} (fire_once={not inference_mode})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test each sample
    all_results = {}

    for sample_id in sample_indices:
        if sample_id >= len(ds):
            print(f"Sample {sample_id} out of range, skipping")
            continue

        # Load sample
        x, label = ds[sample_id]
        x = x.unsqueeze(1)  # Add batch dim: (T, C, H, W) -> (T, B, C, H, W)

        # Detect satellites
        with torch.no_grad():
            boxes, raw_spikes = detect_satellites(
                model, x, device,
                return_spikes=True,
                inference_mode=inference_mode
            )

        # Analyze spikes
        spike_data = analyze_spikes_per_timestep(raw_spikes, sample_id)
        all_results[sample_id] = spike_data

        # Create animation if requested
        if args.animate:
            trajectory = None
            if hasattr(ds, 'recordings') and sample_id < len(ds.recordings):
                event_path = ds.recordings[sample_id].get('event_path')
                if event_path:
                    trajectory = load_trajectory(event_path)

            output_path = output_dir / f'sample{sample_id}_spikes.gif'
            create_animation(
                x, raw_spikes, label, trajectory,
                str(output_path),
                f'Sample {sample_id}'
            )

    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    samples_with_motion = 0
    samples_stuck = 0

    for sample_id, spike_data in all_results.items():
        if spike_data:
            centers_x = [d['center_x'] for d in spike_data]
            centers_y = [d['center_y'] for d in spike_data]
            x_range = max(centers_x) - min(centers_x)
            y_range = max(centers_y) - min(centers_y)

            if x_range < 1 and y_range < 1 and len(spike_data) > 3:
                samples_stuck += 1
                status = "⚠️  STUCK"
            else:
                samples_with_motion += 1
                status = "✓"

            print(f"Sample {sample_id}: {status} "
                  f"(X Δ={x_range:.1f}, Y Δ={y_range:.1f}, "
                  f"{len(spike_data)} timesteps with spikes)")

    print()
    print(f"Samples with temporal variation: {samples_with_motion}")
    print(f"Samples stuck at same location: {samples_stuck}")

    if samples_stuck > 0:
        print("\n⚠️  Some samples have spikes stuck at the same location.")
        print("   This may indicate the latency encoding fix is not working correctly.")
    else:
        print("\n✓ All samples show temporal variation in spike locations.")


if __name__ == '__main__':
    main()
