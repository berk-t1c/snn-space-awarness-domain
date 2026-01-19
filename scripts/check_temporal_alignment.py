#!/usr/bin/env python3
"""
Diagnostic: Check if temporal alignment between events and labels is correct.

This script verifies that:
1. Label timestamps are being filtered to the event window time range
2. Events and labels overlap spatially within each time window
3. Satellites generate MORE events than background (learnable signal)
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset


def main():
    print("=" * 70)
    print("TEMPORAL ALIGNMENT DIAGNOSTIC")
    print("=" * 70)

    # Create dataset with multiple windows per recording
    # Use absolute path or path relative to repo root
    data_root = Path(__file__).parent.parent.parent / "ebssa-data-utah" / "ebssa"
    if not data_root.exists():
        # Try alternative paths
        for alt in ["../ebssa-data-utah/ebssa", "../../ebssa-data-utah/ebssa", "/home/user/ebssa-data-utah/ebssa"]:
            alt_path = Path(alt)
            if alt_path.exists():
                data_root = alt_path
                break

    print(f"Data root: {data_root} (exists: {data_root.exists()})")

    dataset = EBSSADataset(
        root=str(data_root),
        split="all",
        sensor="all",
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=50,  # Multiple windows
        window_overlap=0.5,
    )

    print(f"Dataset: {len(dataset)} samples ({len(dataset.recordings)} recordings x 50 windows)")
    print()

    # Statistics
    total_windows_with_satellites = 0
    total_windows_without_satellites = 0
    total_events_on_satellites = 0
    total_events_on_background = 0
    total_satellite_pixels = 0
    total_background_pixels = 0

    # Check a sample of windows
    n_samples = min(200, len(dataset))
    for i in range(0, n_samples, 10):  # Check every 10th sample
        x, label = dataset[i]

        # x shape: (T, C, H, W) - sum over time and channels to get event locations
        event_mask = (x.sum(dim=(0, 1)) > 0)  # (H, W) where events occurred
        satellite_mask = (label > 0)  # (H, W) where satellites are

        sat_pixels = satellite_mask.sum().item()
        total_pixels = label.numel()

        if sat_pixels > 0:
            total_windows_with_satellites += 1

            # Count events on satellites vs background
            events_on_sat = (event_mask & satellite_mask).sum().item()
            events_on_bg = (event_mask & ~satellite_mask).sum().item()
            bg_pixels = (~satellite_mask).sum().item()

            total_events_on_satellites += events_on_sat
            total_events_on_background += events_on_bg
            total_satellite_pixels += sat_pixels
            total_background_pixels += bg_pixels

            # Calculate densities for this window
            sat_density = events_on_sat / sat_pixels if sat_pixels > 0 else 0
            bg_density = events_on_bg / bg_pixels if bg_pixels > 0 else 0

            if i < 50:  # Print first few detailed
                print(f"Sample {i:3d}: sat_pixels={sat_pixels:4d} ({100*sat_pixels/total_pixels:.1f}%), "
                      f"events_on_sat={events_on_sat:4d}, sat_density={sat_density:.3f}, "
                      f"bg_density={bg_density:.4f}")
        else:
            total_windows_without_satellites += 1
            if i < 50:
                print(f"Sample {i:3d}: NO SATELLITES (empty mask)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Windows with satellites:      {total_windows_with_satellites}")
    print(f"Windows WITHOUT satellites:   {total_windows_without_satellites}")

    if total_windows_with_satellites > 0:
        print(f"\nFor windows WITH satellites:")
        print(f"  Total satellite pixels:     {total_satellite_pixels}")
        print(f"  Total background pixels:    {total_background_pixels}")
        print(f"  Events on satellites:       {total_events_on_satellites}")
        print(f"  Events on background:       {total_events_on_background}")

        if total_satellite_pixels > 0 and total_background_pixels > 0:
            sat_density = total_events_on_satellites / total_satellite_pixels
            bg_density = total_events_on_background / total_background_pixels
            print(f"\n  Satellite event density:    {sat_density:.4f}")
            print(f"  Background event density:   {bg_density:.4f}")

            if bg_density > 0:
                ratio = sat_density / bg_density
                print(f"  Ratio (sat/bg):             {ratio:.2f}x")

                if ratio > 2.0:
                    print("\n  [OK] Satellites generate MORE events than background!")
                    print("       Model should be able to learn this pattern.")
                elif ratio > 1.0:
                    print("\n  [WARN] Satellites generate slightly more events.")
                    print("         Signal is weak but may still be learnable.")
                else:
                    print("\n  [ERROR] Satellites generate FEWER/EQUAL events than background!")
                    print("          Model cannot learn from this data.")
            else:
                print("\n  [INFO] No background events (unusual)")
    else:
        print("\n[ERROR] No windows have satellite labels!")
        print("        Check that temporal alignment is working.")


if __name__ == "__main__":
    main()
