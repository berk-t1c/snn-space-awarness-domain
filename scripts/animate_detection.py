#!/usr/bin/env python3
"""
Satellite Detection 3D Animation Script

Creates animated 3D visualizations showing satellite detections appearing
one by one over time, matching the IGARSS paper Figure 4 style.

Usage:
    # Basic animation (GIF)
    python scripts/animate_detection.py --checkpoint runs/.../checkpoint_best.pt --data-root ../ebssa-data-utah/ebssa

    # Custom settings
    python scripts/animate_detection.py --checkpoint ... --data-root ... --fps 15 --trail 10 --samples 5

    # MP4 output (requires ffmpeg)
    python scripts/animate_detection.py --checkpoint ... --data-root ... --output-format mp4
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import from inference module
from inference import (
    load_model,
    detect_satellites,
    animate_3d_trajectory,
    visualize_3d_trajectory,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate animated 3D satellite detection visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate GIF animation for first 3 samples
    python scripts/animate_detection.py -c checkpoint.pt -d ../data --samples 3

    # High FPS animation with trail effect
    python scripts/animate_detection.py -c checkpoint.pt -d ../data --fps 15 --trail 10

    # Generate both static and animated visualizations
    python scripts/animate_detection.py -c checkpoint.pt -d ../data --static
        """
    )

    # Required arguments
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data-root', '-d', required=True,
                        help='Path to EBSSA dataset root directory')

    # Animation settings
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation frames per second (default: 10)')
    parser.add_argument('--trail', type=int, default=0,
                        help='Trail length - number of timesteps to keep visible. '
                             '0 = show all history (default: 0)')
    parser.add_argument('--interval', type=int, default=100,
                        help='Delay between frames in milliseconds (default: 100)')

    # Output settings
    parser.add_argument('--output-dir', '-o', default='animations',
                        help='Output directory for animations (default: animations)')
    parser.add_argument('--output-format', choices=['gif', 'mp4'], default='gif',
                        help='Output format (default: gif)')
    parser.add_argument('--samples', '-n', type=int, default=5,
                        help='Number of samples to animate (default: 5)')
    parser.add_argument('--sample-indices', type=int, nargs='+',
                        help='Specific sample indices to animate (overrides --samples)')

    # Additional options
    parser.add_argument('--static', action='store_true',
                        help='Also generate static 3D plots (paper style)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Inference threshold (default: 0.05)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Compute device (default: cuda if available)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to use (default: test)')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.threshold)
    print(f"Model loaded with threshold={args.threshold}")

    # Load dataset
    from spikeseg.data.datasets import EBSSADataset

    dataset = EBSSADataset(
        root=args.data_root,
        sensor='all',
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=1,
        split=args.split,
    )
    print(f"Dataset ({args.split} split): {len(dataset)} samples")

    # Determine which samples to process
    if args.sample_indices:
        indices = args.sample_indices
    else:
        indices = list(range(min(args.samples, len(dataset))))

    print(f"Processing {len(indices)} samples: {indices}")

    # Process each sample
    for i, idx in enumerate(indices):
        print(f"\n[{i+1}/{len(indices)}] Processing sample {idx}...")

        # Get data
        x, label = dataset[idx]
        x = x.unsqueeze(1)  # Add batch dimension

        # Run inference with spikes
        boxes, raw_spikes = detect_satellites(model, x, device, return_spikes=True)

        print(f"  Detections: {len(boxes)} satellite(s)")

        # Generate animation
        ext = args.output_format
        anim_path = output_dir / f'detection_sample_{idx:03d}.{ext}'

        print(f"  Generating animation...")
        animate_3d_trajectory(
            x, raw_spikes, label,
            output_path=str(anim_path),
            title=f'Sample {idx}: Satellite Detection',
            fps=args.fps,
            interval=args.interval,
            trail_length=args.trail,
        )
        print(f"  Saved: {anim_path}")

        # Generate static plot if requested
        if args.static:
            static_path = output_dir / f'static_sample_{idx:03d}.png'
            visualize_3d_trajectory(
                x, raw_spikes, label,
                output_path=str(static_path),
                title=f'Sample {idx}: Satellite Trajectory'
            )
            print(f"  Saved static: {static_path}")

    print(f"\nAll animations saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
