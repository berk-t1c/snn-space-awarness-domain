#!/usr/bin/env python3
"""Debug time range extraction to find why some samples have empty labels."""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import scipy.io as sio

def unwrap_field(field):
    """Extract data from potentially nested 0-d object arrays."""
    if field is None:
        return None
    if hasattr(field, 'shape') and field.shape == () and hasattr(field, 'dtype') and field.dtype == object:
        inner = field.item()
        return unwrap_field(inner)
    if hasattr(field, 'flatten'):
        return field.flatten()
    return field

def main():
    data_root = Path("../ebssa-data-utah/ebssa")
    labelled_dir = data_root / "Labelled Data"
    if not labelled_dir.exists():
        labelled_dir = data_root / "Labelled_Data"

    mat_files = sorted(labelled_dir.rglob("*.mat"))

    print(f"Checking {len(mat_files)} recordings for time range issues...\n")

    problem_files = []

    for i, mat_path in enumerate(mat_files):
        try:
            mat = sio.loadmat(mat_path, squeeze_me=True)

            if 'Obj' not in mat:
                print(f"{i}: {mat_path.name} - NO Obj field!")
                problem_files.append(mat_path.name)
                continue

            obj = mat['Obj']

            # Extract fields
            if hasattr(obj, 'dtype') and obj.dtype.names:
                obj_ts = unwrap_field(obj['ts']) if 'ts' in obj.dtype.names else None
                obj_x = unwrap_field(obj['x']) if 'x' in obj.dtype.names else None
                obj_y = unwrap_field(obj['y']) if 'y' in obj.dtype.names else None
            else:
                obj_ts = None
                obj_x = None
                obj_y = None

            # Check timestamps
            if obj_ts is None:
                print(f"{i}: {mat_path.name} - NO timestamps!")
                problem_files.append(mat_path.name)
                continue

            try:
                obj_ts_flat = np.asarray(obj_ts).flatten()
                if obj_ts_flat.dtype == object:
                    obj_ts_flat = np.array([float(x) if np.isscalar(x) else float(np.asarray(x).flatten()[0]) for x in obj_ts_flat])
                else:
                    obj_ts_flat = obj_ts_flat.astype(float)

                if len(obj_ts_flat) == 0:
                    print(f"{i}: {mat_path.name} - EMPTY timestamps!")
                    problem_files.append(mat_path.name)
                    continue

                t_min = float(obj_ts_flat.min())
                t_max = float(obj_ts_flat.max())
                duration = t_max - t_min

                # Also get event timestamps
                if 'TD' in mat:
                    td = mat['TD']
                    if hasattr(td, 'dtype') and td.dtype.names and 'ts' in td.dtype.names:
                        evt_ts = unwrap_field(td['ts'])
                        evt_ts_flat = np.asarray(evt_ts).flatten().astype(float)
                        evt_t_min = float(evt_ts_flat.min())
                        evt_t_max = float(evt_ts_flat.max())
                        evt_duration = evt_t_max - evt_t_min

                        # Check overlap
                        overlap_start = max(t_min, evt_t_min)
                        overlap_end = min(t_max, evt_t_max)
                        has_overlap = overlap_start < overlap_end

                        if not has_overlap:
                            print(f"{i}: {mat_path.name} - NO OVERLAP!")
                            print(f"    Obj time: [{t_min:.3f}, {t_max:.3f}] ({duration:.3f}s)")
                            print(f"    Evt time: [{evt_t_min:.3f}, {evt_t_max:.3f}] ({evt_duration:.3f}s)")
                            problem_files.append(mat_path.name)
                        elif duration < 0.1:
                            print(f"{i}: {mat_path.name} - Very short duration: {duration:.4f}s")
                    else:
                        print(f"{i}: {mat_path.name} - TD has no ts field")
                elif 'ts' in mat:
                    # Direct ts field
                    evt_ts = np.asarray(mat['ts']).flatten().astype(float)
                    evt_t_min = float(evt_ts.min())
                    evt_t_max = float(evt_ts.max())

                    overlap_start = max(t_min, evt_t_min)
                    overlap_end = min(t_max, evt_t_max)
                    has_overlap = overlap_start < overlap_end

                    if not has_overlap:
                        print(f"{i}: {mat_path.name} - NO OVERLAP!")
                        print(f"    Obj time: [{t_min:.3f}, {t_max:.3f}]")
                        print(f"    Evt time: [{evt_t_min:.3f}, {evt_t_max:.3f}]")
                        problem_files.append(mat_path.name)

            except Exception as e:
                print(f"{i}: {mat_path.name} - Error processing timestamps: {e}")
                problem_files.append(mat_path.name)

        except Exception as e:
            print(f"{i}: {mat_path.name} - Error loading: {e}")
            problem_files.append(mat_path.name)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total recordings: {len(mat_files)}")
    print(f"Problem recordings: {len(problem_files)}")
    if problem_files:
        print(f"\nProblematic files:")
        for f in problem_files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
