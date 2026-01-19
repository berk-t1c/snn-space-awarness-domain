#!/usr/bin/env python3
"""
SpikeSEG Evaluation Script with HULK-SMASH Instance Segmentation

Evaluates trained SpikeSEG model on EBSSA dataset using paper metrics:
- Informedness = Sensitivity + Specificity - 1
- Sensitivity = TP / (TP + FN)  (true positive rate)
- Specificity = TN / (TN + FP)  (true negative rate)

Implements the full pipeline from IGARSS 2023:
1. Encoder: Extract STDP features and classification spikes
2. HULK: Unravel spikes back to pixel space using decoder with unpooling
3. SMASH: Group instances by featural-temporal similarity + spatial proximity
4. Evaluation: Compare against ground truth with 1-pixel spatial tolerance

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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from spikeseg.data.datasets import EBSSADataset
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig, EncoderOutput
from spikeseg.algorithms import HULKDecoder, group_instances_to_objects, Instance, BoundingBox


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def dilate_mask(mask: torch.Tensor, pixels: int = 1) -> torch.Tensor:
    """
    Dilate a binary mask by a given number of pixels.

    Used to implement 1-pixel spatial tolerance from paper:
    "A correct prediction was determined if the centroid of the detection
    lands within 1 pixel of the centroid of the ground-truth"

    Args:
        mask: Binary mask (H, W) or (B, H, W)
        pixels: Dilation amount (default 1 for paper methodology)

    Returns:
        Dilated mask with same shape as input
    """
    if pixels == 0:
        return mask

    # Add batch and channel dims for max_pool2d
    needs_batch = mask.dim() == 2
    if needs_batch:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Dilate using max pooling with padding
    kernel_size = 2 * pixels + 1
    dilated = F.max_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=pixels
    )

    # Remove added dimensions
    if needs_batch:
        dilated = dilated.squeeze(0).squeeze(0)
    elif mask.dim() == 4:
        dilated = dilated.squeeze(1)

    return dilated


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    spatial_tolerance: int = 1
) -> Dict[str, float]:
    """
    Compute segmentation metrics for binary classification with spatial tolerance.

    Paper methodology (IGARSS 2023):
    "A correct prediction was determined if the centroid of the detection
    lands within 1 pixel of the centroid of the ground-truth"

    This is implemented by dilating the label mask before comparison,
    which allows predictions within 1 pixel of ground truth to count as TP.

    Args:
        predictions: Predicted class labels (B, H, W) or (H, W)
        labels: Ground truth labels (B, H, W) or (H, W)
        spatial_tolerance: Pixels of tolerance (default 1 per paper)

    Returns:
        Dictionary with TP, TN, FP, FN, sensitivity, specificity, informedness
    """
    # Dilate labels for spatial tolerance
    labels_dilated = dilate_mask(labels, spatial_tolerance) if spatial_tolerance > 0 else labels

    # Flatten to 1D
    pred_flat = predictions.flatten().float()
    label_flat = labels.flatten().float()
    label_dilated_flat = labels_dilated.flatten().float()

    # Binary metrics with tolerance:
    # - TP: prediction=1 AND within tolerance of label=1
    # - TN: prediction=0 AND label=0
    # - FP: prediction=1 AND NOT within tolerance of any label=1
    # - FN: label=1 AND prediction=0 (or not within tolerance)

    # Use dilated labels for TP/FP calculation (spatial tolerance)
    tp = ((pred_flat == 1) & (label_dilated_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (label_dilated_flat == 0)).sum().item()

    # Use original labels for TN/FN (actual ground truth)
    tn = ((pred_flat == 0) & (label_flat == 0)).sum().item()
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


def predict_sample_simple(
    model: SpikeSEGEncoder,
    x: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int] = (128, 128)
) -> torch.Tensor:
    """
    Run inference on a single sample using simple interpolation.

    This is the fallback method that doesn't use HULK decoder.

    Args:
        model: Trained SpikeSEG encoder
        x: Input tensor (B, T, C, H, W) or (T, C, H, W)
        device: Target device
        target_size: Output size for upscaling

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

        # Upscale to target size
        if predictions.shape[-2:] != target_size:
            predictions = F.interpolate(
                predictions.unsqueeze(1).float(),
                size=target_size,
                mode='nearest'
            ).squeeze(1).long()

    return predictions


def predict_sample_hulk(
    model: SpikeSEGEncoder,
    hulk_decoder: HULKDecoder,
    x: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int] = (128, 128),
    threshold: float = 0.5,
    invert_classes: bool = False
) -> Tuple[torch.Tensor, List[Instance]]:
    """
    Run inference using HULK decoder for proper spatial reconstruction.

    This implements the full IGARSS 2023 pipeline:
    1. Encoder produces classification spikes + pooling indices
    2. HULK unravels each spike back to pixel space
    3. Returns both pixel mask and instances for SMASH

    Args:
        model: Trained SpikeSEG encoder
        hulk_decoder: HULK decoder created from encoder
        x: Input tensor (B, T, C, H, W) or (T, C, H, W)
        device: Target device
        target_size: Output size (H, W)
        threshold: HULK threshold for contributing pixels
        invert_classes: If True, use class 0 as satellite instead of class 1

    Returns:
        Tuple of (predictions (B, H, W), instances list)
    """
    model.eval()

    with torch.no_grad():
        # Ensure correct shape
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dim

        batch_size = x.shape[0]

        # Convert (B, T, C, H, W) -> (T, B, C, H, W) for encoder
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)

        n_timesteps = x.shape[0]
        x = x.to(device)

        # Forward pass
        output = model(x)

        # Get pooling indices
        pool_indices = output.pooling_indices

        # Initialize output mask
        H, W = target_size
        all_predictions = torch.zeros(batch_size, H, W, device=device)
        all_instances = []

        # Process each batch item
        for b in range(batch_size):
            # Get classification spikes for this batch item: (T, C, H, W)
            class_spikes = output.classification_spikes[:, b]

            # Check if any spikes occurred
            if class_spikes.sum() == 0:
                continue

            # Debug: Count spikes per class and check overlap
            class0_mask = (class_spikes[:, 0, :, :].sum(dim=0) > 0)  # (H, W)
            class1_mask = (class_spikes[:, 1, :, :].sum(dim=0) > 0)  # (H, W)
            class0_spikes = class0_mask.sum().item()
            class1_spikes = class1_mask.sum().item()
            overlap = (class0_mask & class1_mask).sum().item()
            if b == 0:  # Only log for first batch item
                logger.debug(f"Spikes: class0={class0_spikes:.0f}, class1={class1_spikes:.0f}, overlap={overlap:.0f}")

            # CRITICAL FIX: Only process satellite class spikes
            # HULK processes all classes by default, but we only want satellite detections
            # Create a filtered spike tensor with only satellite class
            satellite_spikes = class_spikes.clone()
            if invert_classes:
                # Use class 0 as satellite (model may have learned inverted labels)
                satellite_spikes[:, 1, :, :] = 0  # Zero out class 1 spikes
            else:
                # Normal: class 1 is satellite
                satellite_spikes[:, 0, :, :] = 0  # Zero out class 0 spikes

            # Use HULK to process satellite spikes to instances
            try:
                instances = hulk_decoder.process_to_instances(
                    classification_spikes=satellite_spikes,
                    pool1_indices=pool_indices.pool1_indices[b:b+1],
                    pool2_indices=pool_indices.pool2_indices[b:b+1],
                    pool1_output_size=pool_indices.pool1_output_size,
                    pool2_output_size=pool_indices.pool2_output_size,
                    n_timesteps=n_timesteps,
                    threshold=threshold
                )

                # Combine all instance masks into prediction
                for inst in instances:
                    if inst.mask is not None:
                        # Resize mask to target size if needed
                        mask = inst.mask
                        if mask.shape != (H, W):
                            mask = F.interpolate(
                                mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(H, W),
                                mode='nearest'
                            ).squeeze()

                        # Mark satellite pixels (class 1)
                        all_predictions[b] = torch.maximum(
                            all_predictions[b],
                            mask.float()
                        )

                all_instances.extend(instances)

            except Exception as e:
                # Fallback to simple argmax if HULK fails
                logger.warning(f"HULK failed for batch {b}: {e}, using fallback")
                class_sum = class_spikes.sum(dim=0)  # (C, H, W)
                pred = class_sum.argmax(dim=0)  # (H, W)
                if pred.shape != (H, W):
                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode='nearest'
                    ).squeeze()
                all_predictions[b] = pred.float()

    return all_predictions.long(), all_instances


def predict_sample(
    model: SpikeSEGEncoder,
    x: torch.Tensor,
    device: torch.device,
    hulk_decoder: Optional[HULKDecoder] = None,
    target_size: Tuple[int, int] = (128, 128),
    use_hulk: bool = True,
    invert_classes: bool = False
) -> Tuple[torch.Tensor, Optional[List[Instance]]]:
    """
    Run inference on a single sample.

    Args:
        model: Trained SpikeSEG encoder
        x: Input tensor (B, T, C, H, W) or (T, C, H, W)
        device: Target device
        hulk_decoder: Optional HULK decoder for proper spatial reconstruction
        target_size: Output size for predictions
        use_hulk: Whether to use HULK (if decoder provided)
        invert_classes: If True, use class 0 as satellite instead of class 1

    Returns:
        Tuple of (predictions (B, H, W), instances or None)
    """
    if use_hulk and hulk_decoder is not None:
        return predict_sample_hulk(model, hulk_decoder, x, device, target_size, invert_classes=invert_classes)
    else:
        predictions = predict_sample_simple(model, x, device, target_size)
        return predictions, None


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    create_hulk: bool = True
) -> Tuple[SpikeSEGEncoder, Optional[HULKDecoder]]:
    """
    Load trained model from checkpoint and create HULK decoder.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Target device
        create_hulk: Whether to create HULK decoder

    Returns:
        Tuple of (SpikeSEG encoder, HULK decoder or None)
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
        # IGARSS 2023: 2x2 pooling (not Kheradpisheh's 7x7 stride 6)
        pool1_kernel_size=2,
        pool1_stride=2,
        pool2_kernel_size=2,
        pool2_stride=2,
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

    # IGARSS 2023 paper leak values (percentage of threshold):
    # - Layer 1: 90% leak (short-term memory for fast objects)
    # - Layer 2: 10% leak (long-term memory for slow objects)
    # - Layer 3: 0% leak (classification)
    # These temporal dynamics are CRITICAL for detecting both fast and slow objects
    model.conv1.neuron.leak.data.fill_(0.09)  # 90% of threshold=0.1
    model.conv2.neuron.leak.data.fill_(0.01)  # 10% of threshold=0.1
    model.conv3.neuron.leak.data.fill_(0.0)   # No leak for classification

    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {enc_config.conv1.out_channels}→{enc_config.conv2.out_channels}→{enc_config.conv3.out_channels} features")

    # Create HULK decoder from encoder
    hulk_decoder = None
    if create_hulk:
        try:
            hulk_decoder = HULKDecoder.from_encoder(model)
            hulk_decoder.to(device)
            hulk_decoder.eval()
            logger.info(f"HULK decoder created: {hulk_decoder.n_features} features")
        except Exception as e:
            logger.warning(f"Failed to create HULK decoder: {e}")
            logger.warning("Falling back to simple interpolation")

    return model, hulk_decoder


def evaluate(
    model: SpikeSEGEncoder,
    dataloader: DataLoader,
    device: torch.device,
    hulk_decoder: Optional[HULKDecoder] = None,
    spatial_tolerance: int = 1,
    use_hulk: bool = True,
    use_smash: bool = True,
    invert_classes: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on dataset using HULK-SMASH pipeline.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Target device
        hulk_decoder: HULK decoder for proper spatial reconstruction
        spatial_tolerance: Pixels of tolerance for TP (default 1 per paper)
        use_hulk: Whether to use HULK decoder
        use_smash: Whether to use SMASH grouping
        invert_classes: If True, use class 0 as satellite instead of class 1

    Returns:
        Aggregated metrics
    """
    # Accumulators
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    n_samples = 0

    logger.info(f"Evaluating on {len(dataloader)} batches...")
    logger.info(f"  Using HULK decoder: {use_hulk and hulk_decoder is not None}")
    logger.info(f"  Spatial tolerance: {spatial_tolerance} pixel(s)")
    logger.info(f"  Using SMASH grouping: {use_smash}")
    logger.info(f"  Invert classes: {invert_classes} (satellite=class {'0' if invert_classes else '1'})")

    total_spikes = 0
    total_label_positives = 0
    total_instances = 0
    total_objects = 0

    for batch_idx, (x, labels) in enumerate(dataloader):
        # Move labels to device
        labels = labels.to(device)
        target_size = labels.shape[-2:]

        # Get predictions using HULK if available
        predictions, instances = predict_sample(
            model, x, device,
            hulk_decoder=hulk_decoder,
            target_size=target_size,
            use_hulk=use_hulk,
            invert_classes=invert_classes
        )

        # Debug: count spikes and positive labels
        total_spikes += (predictions == 1).sum().item()
        total_label_positives += (labels == 1).sum().item()

        # Track instances and objects if using HULK
        if instances:
            total_instances += len(instances)
            if use_smash:
                objects = group_instances_to_objects(instances, smash_threshold=0.1)
                total_objects += len(objects)

        # Debug output for first few samples
        if batch_idx < 3:
            n_inst = len(instances) if instances else 0
            logger.info(f"  Sample {batch_idx}: pred_sum={predictions.sum().item()}, "
                       f"label_sum={labels.sum().item()}, instances={n_inst}")

        # Ensure predictions match label size
        if predictions.shape[-2:] != labels.shape[-2:]:
            predictions = F.interpolate(
                predictions.unsqueeze(1).float(),
                size=labels.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        # Compute metrics for this batch with spatial tolerance
        metrics = compute_metrics(predictions, labels, spatial_tolerance=spatial_tolerance)

        total_tp += metrics['tp']
        total_tn += metrics['tn']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        n_samples += x.shape[0] if x.dim() == 5 else 1

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  [{batch_idx + 1}/{len(dataloader)}] samples processed")

    logger.info(f"  Total predicted positives: {total_spikes}")
    logger.info(f"  Total label positives: {total_label_positives}")
    if total_instances > 0:
        logger.info(f"  Total instances (HULK): {total_instances}")
        if use_smash:
            logger.info(f"  Total objects (SMASH): {total_objects}")

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
        'iou': iou,
        'total_instances': total_instances,
        'total_objects': total_objects,
        'spatial_tolerance': spatial_tolerance
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpikeSEG model with HULK-SMASH pipeline"
    )
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
    parser.add_argument(
        '--spatial-tolerance',
        type=int,
        default=1,
        help='Spatial tolerance in pixels for TP (default: 1 per IGARSS 2023)'
    )
    parser.add_argument(
        '--no-hulk',
        action='store_true',
        help='Disable HULK decoder (use simple interpolation instead)'
    )
    parser.add_argument(
        '--no-smash',
        action='store_true',
        help='Disable SMASH instance grouping'
    )
    parser.add_argument(
        '--invert-classes',
        action='store_true',
        help='Invert class assignment (use class 0 as satellite instead of class 1)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model and HULK decoder
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, hulk_decoder = load_model(
        checkpoint_path, device,
        create_hulk=not args.no_hulk
    )

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
    logger.info("  SpikeSEG Evaluation with HULK-SMASH")
    logger.info("=" * 60)

    metrics = evaluate(
        model, dataloader, device,
        hulk_decoder=hulk_decoder,
        spatial_tolerance=args.spatial_tolerance,
        use_hulk=not args.no_hulk,
        use_smash=not args.no_smash,
        invert_classes=args.invert_classes
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Results")
    logger.info("=" * 60)
    logger.info(f"  Samples:       {metrics['n_samples']}")
    logger.info(f"  TP/TN/FP/FN:   {metrics['total_tp']}/{metrics['total_tn']}/{metrics['total_fp']}/{metrics['total_fn']}")
    if metrics.get('total_instances', 0) > 0:
        logger.info(f"  Instances:     {metrics['total_instances']}")
        logger.info(f"  Objects:       {metrics['total_objects']}")
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
