#!/usr/bin/env python3
"""
SpikeSEG Evaluation Script

Evaluates trained SpikeSEG model on EBSSA dataset using paper metrics:
- Informedness = Sensitivity + Specificity - 1
- Sensitivity = TP / (TP + FN)  (true positive rate)
- Specificity = TN / (TN + FP)  (true negative rate)

Paper Reference:
    Kirkland et al. 2023 (IGARSS) - "Neuromorphic sensing and processing
    for space domain awareness" reports 89.1% informedness on EBSSA.

Usage:
    python scripts/evaluate.py --checkpoint runs/experiment/checkpoints/checkpoint_best.pt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from spikeseg.data.datasets import EBSSADataset
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute segmentation metrics for binary classification.

    Args:
        predictions: Predicted class labels (B, H, W) or (H, W)
        labels: Ground truth labels (B, H, W) or (H, W)

    Returns:
        Dictionary with TP, TN, FP, FN, sensitivity, specificity, informedness
    """
    # Flatten to 1D
    pred_flat = predictions.flatten().float()
    label_flat = labels.flatten().float()

    # Binary metrics (class 1 = satellite, class 0 = background)
    tp = ((pred_flat == 1) & (label_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (label_flat == 0)).sum().item()
    fp = ((pred_flat == 1) & (label_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (label_flat == 1)).sum().item()

    # Calculate rates with smoothing to avoid division by zero
    eps = 1e-7
    sensitivity = tp / (tp + fn + eps)  # True positive rate
    specificity = tn / (tn + fp + eps)  # True negative rate
    informedness = sensitivity + specificity - 1.0

    # Also compute accuracy and IoU
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)  # Intersection over union for satellite class

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'informedness': informedness,
        'accuracy': accuracy,
        'iou': iou
    }


def predict_sample(
    model: SpikeSEGEncoder,
    x: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Run inference on a single sample.

    Args:
        model: Trained SpikeSEG encoder
        x: Input tensor (B, T, C, H, W) or (T, C, H, W)
        device: Target device

    Returns:
        Predicted class labels (B, H, W)
    """
    model.eval()

    with torch.no_grad():
        # Ensure correct shape
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dim

        # Convert (B, T, C, H, W) -> (T, B, C, H, W) for encoder
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)

        x = x.to(device)

        # Forward pass
        output = model(x)

        # Get classification spikes: (T, B, C, H, W) -> (B, C, H, W)
        # Sum spikes over time
        class_spikes = output.classification_spikes.sum(dim=0)  # (B, C, H, W)

        # Predict class with most spikes at each spatial location
        # For binary: class 0 = background, class 1 = satellite
        predictions = class_spikes.argmax(dim=1)  # (B, H, W)

    return predictions


def load_model(checkpoint_path: Path, device: torch.device) -> SpikeSEGEncoder:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Target device

    Returns:
        Loaded SpikeSEG encoder
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint or use defaults
    # Use IGARSS 2023 defaults with proper LayerConfig structure
    thresholds = [0.1, 0.1, 0.1]
    leaks = [0.09, 0.01, 0.0]

    enc_config = EncoderConfig(
        input_channels=2,
        conv1=LayerConfig(out_channels=4, kernel_size=5, threshold=thresholds[0], leak=leaks[0]),
        conv2=LayerConfig(out_channels=36, kernel_size=5, threshold=thresholds[1], leak=leaks[1]),
        conv3=LayerConfig(out_channels=2, kernel_size=7, threshold=thresholds[2], leak=leaks[2]),
    )

    model = SpikeSEGEncoder(enc_config)

    # Handle state dict with 'encoder.' prefix from training
    state_dict = checkpoint['model_state_dict']

    # Strip 'encoder.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[len('encoder.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Filter out non-parameter keys (membrane, has_fired, etc.)
    param_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in param_keys}

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {enc_config.conv1.out_channels}→{enc_config.conv2.out_channels}→{enc_config.conv3.out_channels} features")

    return model


def evaluate(
    model: SpikeSEGEncoder,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Target device

    Returns:
        Aggregated metrics
    """
    # Accumulators
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    n_samples = 0

    logger.info(f"Evaluating on {len(dataloader)} batches...")

    for batch_idx, (x, labels) in enumerate(dataloader):
        # Move labels to device
        labels = labels.to(device)

        # Get predictions
        predictions = predict_sample(model, x, device)

        # Resize predictions to match label size if needed
        if predictions.shape[-2:] != labels.shape[-2:]:
            predictions = F.interpolate(
                predictions.unsqueeze(1).float(),
                size=labels.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        # Compute metrics for this batch
        metrics = compute_metrics(predictions, labels)

        total_tp += metrics['tp']
        total_tn += metrics['tn']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        n_samples += x.shape[0] if x.dim() == 5 else 1

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  [{batch_idx + 1}/{len(dataloader)}] samples processed")

    # Final aggregated metrics
    eps = 1e-7
    sensitivity = total_tp / (total_tp + total_fn + eps)
    specificity = total_tn / (total_tn + total_fp + eps)
    informedness = sensitivity + specificity - 1.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)

    return {
        'n_samples': n_samples,
        'total_tp': total_tp,
        'total_tn': total_tn,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'informedness': informedness,
        'accuracy': accuracy,
        'iou': iou
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpikeSEG model")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default='../ebssa-data-utah/ebssa',
        help='Path to EBSSA dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--sensor',
        type=str,
        default='all',
        choices=['ATIS', 'DAVIS', 'all'],
        help='Sensor type to evaluate'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file for metrics (optional)'
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, device)

    # Load dataset
    logger.info(f"Loading dataset from: {args.data_root}")
    dataset = EBSSADataset(
        root=args.data_root,
        split=args.split,
        sensor=args.sensor,
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=1,  # Single window for evaluation
    )

    logger.info(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate
    logger.info("=" * 60)
    logger.info("  SpikeSEG Evaluation")
    logger.info("=" * 60)

    metrics = evaluate(model, dataloader, device)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Results")
    logger.info("=" * 60)
    logger.info(f"  Samples:       {metrics['n_samples']}")
    logger.info(f"  TP/TN/FP/FN:   {metrics['total_tp']}/{metrics['total_tn']}/{metrics['total_fp']}/{metrics['total_fn']}")
    logger.info("")
    logger.info(f"  Sensitivity:   {metrics['sensitivity']*100:.1f}%")
    logger.info(f"  Specificity:   {metrics['specificity']*100:.1f}%")
    logger.info(f"  Informedness:  {metrics['informedness']*100:.1f}%  (paper: 89.1%)")
    logger.info(f"  Accuracy:      {metrics['accuracy']*100:.1f}%")
    logger.info(f"  IoU:           {metrics['iou']*100:.1f}%")
    logger.info("=" * 60)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {output_path}")

    return metrics


if __name__ == '__main__':
    main()
