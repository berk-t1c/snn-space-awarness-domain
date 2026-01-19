#!/usr/bin/env python3
"""Check timestamp units between events (TD.ts) and labels (Obj.ts)."""

import numpy as np
import scipy.io as sio
from pathlib import Path


def to_scalar(arr):
    """Convert numpy array to scalar, handling nested arrays."""
    while hasattr(arr, '__len__') and len(arr) == 1:
        arr = arr[0]
    if hasattr(arr, 'item'):
        return arr.item()
    return float(arr)


def flatten_numeric(arr):
    """Recursively flatten and extract numeric values from nested arrays."""
    arr = np.asarray(arr)
    # Keep flattening until we get a 1D numeric array
    while arr.dtype == object or (arr.ndim > 1):
        if arr.dtype == object:
            # Try to extract from object array
            flat = []
            for item in arr.flat:
                if np.isscalar(item):
                    flat.append(float(item))
                elif hasattr(item, '__len__'):
                    flat.extend(flatten_numeric(item))
                else:
                    flat.append(float(item))
            return np.array(flat, dtype=float)
        else:
            arr = arr.flatten()
    return arr.astype(float)


def main():
    # Find a data file
    data_root = Path("/home/ubuntu/ebssa-data-utah/ebssa")
    if not data_root.exists():
        data_root = Path("../ebssa-data-utah/ebssa")

    labelled_dir = data_root / "Labelled Data"
    mat_files = list(labelled_dir.rglob("*.mat"))

    # Filter to *_labelled.mat files (contain both events and labels)
    labelled_files = [f for f in mat_files if '_labelled' in f.stem.lower()]

    print(f"Found {len(labelled_files)} labelled files")

    for mat_file in labelled_files[:3]:  # Check first 3 files
        print(f"\n{'='*60}")
        print(f"File: {mat_file.name}")
        print('='*60)

        mat = sio.loadmat(mat_file, squeeze_me=True)

        event_ts = None
        e_min, e_max = None, None

        # Event timestamps
        if 'TD' in mat:
            td = mat['TD']
            try:
                if hasattr(td, 'dtype') and td.dtype.names:
                    raw_ts = td['ts'] if 'ts' in td.dtype.names else None
                else:
                    raw_ts = td[0, 0]['ts'] if hasattr(td, 'shape') else None

                if raw_ts is not None:
                    event_ts = flatten_numeric(raw_ts)
                    if len(event_ts) > 0:
                        e_min, e_max = event_ts.min(), event_ts.max()
                        print(f"\nEvent timestamps (TD.ts):")
                        print(f"  Count: {len(event_ts):,}")
                        print(f"  Min: {e_min:,.0f}")
                        print(f"  Max: {e_max:,.0f}")
                        print(f"  Duration: {(e_max - e_min):,.0f}")
                        print(f"  Duration (seconds, if microsec): {(e_max - e_min) / 1e6:.2f}s")
                        print(f"  Duration (seconds, if millisec): {(e_max - e_min) / 1e3:.2f}s")
            except Exception as e:
                print(f"\nError reading event timestamps: {e}")

        # Label timestamps
        if 'Obj' in mat:
            obj = mat['Obj']
            try:
                if hasattr(obj, 'dtype') and obj.dtype.names:
                    raw_ts = obj['ts'] if 'ts' in obj.dtype.names else None

                    if raw_ts is not None:
                        label_ts = flatten_numeric(raw_ts)
                        if len(label_ts) > 0:
                            l_min, l_max = label_ts.min(), label_ts.max()
                            print(f"\nLabel timestamps (Obj.ts):")
                            print(f"  Count: {len(label_ts):,}")
                            print(f"  Min: {l_min:,.0f}")
                            print(f"  Max: {l_max:,.0f}")
                            print(f"  Duration: {(l_max - l_min):,.0f}")
                            print(f"  Duration (seconds, if microsec): {(l_max - l_min) / 1e6:.2f}s")
                            print(f"  Duration (seconds, if millisec): {(l_max - l_min) / 1e3:.2f}s")

                            # Compare ranges
                            if e_min is not None:
                                print(f"\nTimestamp comparison:")
                                print(f"  Event range: [{e_min:,.0f}, {e_max:,.0f}]")
                                print(f"  Label range: [{l_min:,.0f}, {l_max:,.0f}]")

                                overlap = (l_min <= e_max) and (l_max >= e_min)
                                print(f"  Ranges overlap: {overlap}")

                                if not overlap:
                                    ratio = e_min / l_min if l_min != 0 else 0
                                    print(f"  Event/Label ratio: {ratio:.2f}x (if ~1000, labels in ms)")
            except Exception as e:
                print(f"\nError reading label timestamps: {e}")


if __name__ == "__main__":
    main()
