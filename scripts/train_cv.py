#!/usr/bin/env python3
"""
SpikeSEG Cross-Validation Training Script

Implements k-fold cross-validation with a held-out test set:
1. Hold out 10% as test set (never touched during training)
2. Split remaining 90% into k folds
3. Train on k-1 folds, validate on 1 fold
4. Repeat for all k folds
5. Report mean ± std metrics

Usage:
    python scripts/train_cv.py --config configs/config.yaml --n-folds 10
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def get_valid_recordings(data_root: str, sensor: str = "all") -> List[str]:
    """Get list of valid recording paths (those with non-empty labels)."""
    import scipy.io as sio

    data_path = Path(data_root)
    labelled_dir = data_path / "Labelled Data"

    recordings = []

    # Collect all .mat files
    sensor_dirs = []
    if sensor in ["ATIS", "all"]:
        sensor_dirs.append(labelled_dir / "ATIS")
    if sensor in ["DAVIS", "all"]:
        sensor_dirs.append(labelled_dir / "DAVIS")

    for sensor_dir in sensor_dirs:
        if sensor_dir.exists():
            for mat_file in sorted(sensor_dir.glob("*.mat")):
                recordings.append(str(mat_file))

    # Filter valid recordings (non-empty labels)
    def unwrap_field(field):
        if field is None:
            return None
        if hasattr(field, 'shape') and field.shape == () and hasattr(field, 'dtype') and field.dtype == object:
            return unwrap_field(field.item())
        if hasattr(field, 'flatten'):
            return field.flatten()
        return field

    valid = []
    for rec_path in recordings:
        try:
            mat = sio.loadmat(rec_path, squeeze_me=True)
            if 'Obj' not in mat:
                continue
            obj = mat['Obj']
            if hasattr(obj, 'dtype') and obj.dtype.names:
                obj_x = unwrap_field(obj['x']) if 'x' in obj.dtype.names else None
                x_len = len(np.asarray(obj_x).flatten()) if obj_x is not None else 0
                if x_len > 0:
                    valid.append(rec_path)
        except Exception:
            continue

    return valid


def create_fold_splits(recordings: List[str], n_folds: int, test_ratio: float = 0.1, seed: int = 42):
    """
    Create train/val/test splits for k-fold cross-validation.

    Returns:
        test_set: List of test recording paths (held out)
        folds: List of (train_indices, val_indices) for each fold
    """
    rng = random.Random(seed)

    # Shuffle recordings
    shuffled = recordings.copy()
    rng.shuffle(shuffled)

    # Hold out test set
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_set = shuffled[:n_test]
    train_val_set = shuffled[n_test:]

    # Create k folds
    n = len(train_val_set)
    fold_size = n // n_folds

    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n

        val_indices = list(range(val_start, val_end))
        train_indices = [j for j in range(n) if j not in val_indices]

        folds.append((train_indices, val_indices))

    return test_set, train_val_set, folds


def write_fold_file(recordings: List[str], indices: List[int], output_path: Path):
    """Write recording paths to a file for the dataset to use."""
    with open(output_path, 'w') as f:
        for idx in indices:
            f.write(recordings[idx] + '\n')


def run_training(fold_num: int, train_file: Path, val_file: Path, config_path: str, output_dir: Path):
    """Run training for a single fold."""
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config_path,
        "--train-recordings", str(train_file),
        "--val-recordings", str(val_file),
        "--output", str(output_dir),
        "--name", f"fold_{fold_num:02d}"
    ]

    print(f"\n{'='*60}")
    print(f"  Training Fold {fold_num + 1}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode == 0


def run_evaluation(checkpoint_path: Path, data_root: str, recordings_file: Path, threshold: float = 0.05):
    """Run evaluation and return metrics."""
    import torch

    # Import here to avoid issues
    sys.path.insert(0, str(Path.cwd()))
    from scripts.evaluate import load_model, evaluate_volume_based
    from spikeseg.data.datasets import EBSSADataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, _ = load_model(checkpoint_path, device, create_hulk=False, inference_threshold=threshold)

    # Load dataset with specific recordings
    with open(recordings_file, 'r') as f:
        recording_paths = [line.strip() for line in f if line.strip()]

    dataset = EBSSADataset(
        root=data_root,
        split="all",
        sensor="all",
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=1,
    )

    # Filter to only requested recordings
    dataset.recordings = [r for r in dataset.recordings if r['event_path'] in recording_paths]
    dataset._build_sample_index()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate
    metrics = evaluate_volume_based(model, dataloader, device, spatial_tolerance=1.0)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='SpikeSEG K-Fold Cross-Validation')
    parser.add_argument('--config', '-c', default='configs/config.yaml', help='Config file')
    parser.add_argument('--data-root', '-d', default='../ebssa-data-utah/ebssa', help='Data root')
    parser.add_argument('--n-folds', '-k', type=int, default=10, help='Number of folds')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', '-o', default='runs/cv', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.05, help='Inference threshold')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation on existing folds')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get valid recordings
    print("Finding valid recordings...")
    recordings = get_valid_recordings(args.data_root)
    print(f"Found {len(recordings)} valid recordings")

    # Create fold splits
    test_set, train_val_set, folds = create_fold_splits(
        recordings, args.n_folds, args.test_ratio, args.seed
    )

    print(f"Test set: {len(test_set)} recordings")
    print(f"Train/Val set: {len(train_val_set)} recordings")
    print(f"Folds: {args.n_folds}")

    # Save test set
    test_file = output_dir / "test_recordings.txt"
    with open(test_file, 'w') as f:
        for rec in test_set:
            f.write(rec + '\n')

    # Save fold info
    fold_info = {
        'n_folds': args.n_folds,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'n_test': len(test_set),
        'n_train_val': len(train_val_set),
        'folds': [{'train': len(t), 'val': len(v)} for t, v in folds]
    }
    with open(output_dir / "fold_info.json", 'w') as f:
        json.dump(fold_info, f, indent=2)

    if not args.eval_only:
        # Train each fold
        for fold_num, (train_indices, val_indices) in enumerate(folds):
            fold_dir = output_dir / f"fold_{fold_num:02d}"
            fold_dir.mkdir(exist_ok=True)

            # Write recording files
            train_file = fold_dir / "train_recordings.txt"
            val_file = fold_dir / "val_recordings.txt"

            write_fold_file(train_val_set, train_indices, train_file)
            write_fold_file(train_val_set, val_indices, val_file)

            print(f"\nFold {fold_num + 1}/{args.n_folds}: {len(train_indices)} train, {len(val_indices)} val")

            # Run training
            success = run_training(fold_num, train_file, val_file, args.config, fold_dir)
            if not success:
                print(f"WARNING: Fold {fold_num + 1} training failed!")

    # Evaluate all folds
    print("\n" + "="*60)
    print("  Cross-Validation Results")
    print("="*60)

    val_metrics = []
    test_metrics = []

    for fold_num in range(args.n_folds):
        fold_dir = output_dir / f"fold_{fold_num:02d}"
        checkpoint = fold_dir / "checkpoints" / "checkpoint_best.pt"

        if not checkpoint.exists():
            # Try alternate path
            for run_dir in fold_dir.glob("spikeseg_*"):
                alt_checkpoint = run_dir / "checkpoints" / "checkpoint_best.pt"
                if alt_checkpoint.exists():
                    checkpoint = alt_checkpoint
                    break

        if not checkpoint.exists():
            print(f"Fold {fold_num + 1}: No checkpoint found, skipping")
            continue

        val_file = fold_dir / "val_recordings.txt"

        # Evaluate on val fold
        print(f"\nFold {fold_num + 1} validation:")
        try:
            metrics = run_evaluation(checkpoint, args.data_root, val_file, args.threshold)
            val_metrics.append(metrics)
            print(f"  Informedness: {metrics['informedness']*100:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")

        # Evaluate on test set
        print(f"Fold {fold_num + 1} test:")
        try:
            metrics = run_evaluation(checkpoint, args.data_root, test_file, args.threshold)
            test_metrics.append(metrics)
            print(f"  Informedness: {metrics['informedness']*100:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary statistics
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    if val_metrics:
        val_inf = [m['informedness'] for m in val_metrics]
        print(f"\nValidation Informedness:")
        print(f"  Mean: {np.mean(val_inf)*100:.1f}%")
        print(f"  Std:  {np.std(val_inf)*100:.1f}%")
        print(f"  Min:  {np.min(val_inf)*100:.1f}%")
        print(f"  Max:  {np.max(val_inf)*100:.1f}%")

    if test_metrics:
        test_inf = [m['informedness'] for m in test_metrics]
        print(f"\nTest Informedness:")
        print(f"  Mean: {np.mean(test_inf)*100:.1f}%")
        print(f"  Std:  {np.std(test_inf)*100:.1f}%")
        print(f"  Min:  {np.min(test_inf)*100:.1f}%")
        print(f"  Max:  {np.max(test_inf)*100:.1f}%")

    # Save results
    results = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_informedness_mean': float(np.mean(val_inf)) if val_metrics else None,
        'val_informedness_std': float(np.std(val_inf)) if val_metrics else None,
        'test_informedness_mean': float(np.mean(test_inf)) if test_metrics else None,
        'test_informedness_std': float(np.std(test_inf)) if test_metrics else None,
    }

    with open(output_dir / "cv_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {output_dir / 'cv_results.json'}")


if __name__ == '__main__':
    main()
