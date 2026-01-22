#!/usr/bin/env python3
"""Debug samples with sat_pixels=0 to understand why."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset

def main():
    dataset = EBSSADataset(
        root="../ebssa-data-utah/ebssa",
        split="all",
        sensor="all",
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=1,
    )

    print(f"Dataset: {len(dataset)} samples\n")
    print("Analyzing samples with sat_pixels=0...\n")

    empty_count = 0
    good_count = 0

    for i in range(len(dataset)):
        x, label = dataset[i]

        event_mask = (x.sum(dim=(0, 1)) > 0)
        satellite_mask = (label > 0)

        sat_pixels = satellite_mask.sum().item()
        total_events = event_mask.sum().item()

        if sat_pixels == 0:
            empty_count += 1
            # Get recording info
            rec_idx = i // dataset.windows_per_recording
            if rec_idx < len(dataset.recordings):
                rec_info = dataset.recordings[rec_idx]
                if isinstance(rec_info, dict):
                    rec_name = rec_info.get('events', rec_info.get('path', 'unknown'))
                    if hasattr(rec_name, 'name'):
                        rec_name = rec_name.name
                else:
                    rec_name = rec_info.name if hasattr(rec_info, 'name') else str(rec_info)
                print(f"Sample {i}: sat_pixels=0, events={total_events}, recording={rec_name}")
        else:
            good_count += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Samples with satellites:    {good_count}")
    print(f"Samples WITHOUT satellites: {empty_count}")
    print(f"Total:                      {len(dataset)}")
    print(f"\nEmpty label ratio: {empty_count/len(dataset)*100:.1f}%")

if __name__ == "__main__":
    main()
