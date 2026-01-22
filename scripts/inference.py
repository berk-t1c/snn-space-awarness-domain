#!/usr/bin/env python3
"""
SpikeSEG Inference Script - Satellite Detection with Bounding Boxes

Runs inference on event camera data and outputs detected satellite bounding boxes.

Usage:
    python scripts/inference.py --checkpoint runs/.../checkpoint_best.pt --input data.mat
    python scripts/inference.py --checkpoint ... --data-root ../ebssa-data-utah/ebssa --visualize
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig


@dataclass
class BoundingBox:
    """Detected satellite bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> dict:
        return {
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'center_x': self.center[0],
            'center_y': self.center[1],
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence
        }


def merge_overlapping_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.3) -> List[BoundingBox]:
    """
    Merge overlapping bounding boxes into single consolidated boxes.

    Args:
        boxes: List of detected bounding boxes
        iou_threshold: IoU threshold for merging (lower = more aggressive merging)

    Returns:
        List of merged bounding boxes
    """
    if len(boxes) <= 1:
        return boxes

    # Convert to numpy for easier manipulation
    coords = np.array([[b.x_min, b.y_min, b.x_max, b.y_max, b.confidence] for b in boxes])

    # Iteratively merge overlapping boxes
    merged = True
    while merged:
        merged = False
        n = len(coords)
        if n <= 1:
            break

        # Find pairs to merge
        to_merge = []
        used = set()

        for i in range(n):
            if i in used:
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue

                # Calculate IoU
                x1 = max(coords[i, 0], coords[j, 0])
                y1 = max(coords[i, 1], coords[j, 1])
                x2 = min(coords[i, 2], coords[j, 2])
                y2 = min(coords[i, 3], coords[j, 3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area_i = (coords[i, 2] - coords[i, 0]) * (coords[i, 3] - coords[i, 1])
                    area_j = (coords[j, 2] - coords[j, 0]) * (coords[j, 3] - coords[j, 1])
                    union = area_i + area_j - intersection
                    iou = intersection / union if union > 0 else 0

                    # Also check if boxes are close (within 10 pixels)
                    close = (abs(coords[i, 0] - coords[j, 2]) < 10 or
                             abs(coords[j, 0] - coords[i, 2]) < 10 or
                             abs(coords[i, 1] - coords[j, 3]) < 10 or
                             abs(coords[j, 1] - coords[i, 3]) < 10)

                    if iou > iou_threshold or (iou > 0 and close):
                        to_merge.append((i, j))
                        used.add(i)
                        used.add(j)
                        merged = True
                        break
            if merged:
                break

        # Merge found pairs
        if to_merge:
            new_coords = []
            merged_indices = set()

            for i, j in to_merge:
                # Merge boxes i and j
                new_box = [
                    min(coords[i, 0], coords[j, 0]),  # x_min
                    min(coords[i, 1], coords[j, 1]),  # y_min
                    max(coords[i, 2], coords[j, 2]),  # x_max
                    max(coords[i, 3], coords[j, 3]),  # y_max
                    max(coords[i, 4], coords[j, 4]),  # confidence
                ]
                new_coords.append(new_box)
                merged_indices.add(i)
                merged_indices.add(j)

            # Keep unmerged boxes
            for i in range(n):
                if i not in merged_indices:
                    new_coords.append(coords[i].tolist())

            coords = np.array(new_coords)

    # Convert back to BoundingBox objects
    return [
        BoundingBox(
            x_min=float(c[0]),
            y_min=float(c[1]),
            x_max=float(c[2]),
            y_max=float(c[3]),
            confidence=float(c[4])
        )
        for c in coords
    ]


def load_model(checkpoint_path: str, device: torch.device, inference_threshold: float = 0.05):
    """Load model from checkpoint with optimal inference threshold."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})

    n_classes = model_cfg.get('n_classes', 1)
    kernel_sizes = model_cfg.get('kernel_sizes', [5, 5, 7])
    conv1_channels = model_cfg.get('conv1_channels', 4)
    conv2_channels = model_cfg.get('conv2_channels', 36)
    input_channels = config.get('data', {}).get('input_channels', 2)

    # Use inference threshold (0.05 works best based on evaluation)
    thresholds = [inference_threshold] * 3
    leak_ratios = [0.9, 0.1, 0.0]
    leaks = [inference_threshold * r for r in leak_ratios]

    enc_config = EncoderConfig(
        input_channels=input_channels,
        conv1=LayerConfig(out_channels=conv1_channels, kernel_size=kernel_sizes[0],
                         threshold=thresholds[0], leak=leaks[0]),
        conv2=LayerConfig(out_channels=conv2_channels, kernel_size=kernel_sizes[1],
                         threshold=thresholds[1], leak=leaks[1]),
        conv3=LayerConfig(out_channels=n_classes, kernel_size=kernel_sizes[2],
                         threshold=thresholds[2], leak=leaks[2]),
    )

    model = SpikeSEGEncoder(enc_config)

    # Load weights
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_state_dict[key[len('encoder.'):]] = value
        else:
            new_state_dict[key] = value

    param_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in param_keys}
    model.load_state_dict(filtered_state_dict, strict=False)

    # Set inference thresholds AFTER load_state_dict
    for i, (layer_name, leak_val) in enumerate([
        ('conv1', leaks[0]), ('conv2', leaks[1]), ('conv3', leaks[2])
    ]):
        layer = getattr(model, layer_name)
        if hasattr(layer, 'neuron'):
            layer.neuron.threshold.fill_(inference_threshold)
            layer.neuron.leak.fill_(leak_val)

    model.to(device)
    model.eval()

    return model


def detect_satellites(
    model: SpikeSEGEncoder,
    events: torch.Tensor,
    device: torch.device,
    min_cluster_pixels: int = 2,
    return_spikes: bool = False,
) -> List[BoundingBox]:
    """
    Run inference and return detected satellite bounding boxes.

    Args:
        model: Loaded SpikeSEG model
        events: Input tensor (T, C, H, W) or (B, T, C, H, W)
        device: Compute device
        min_cluster_pixels: Minimum cluster size to count as detection
        return_spikes: If True, also return raw classification spikes for visualization

    Returns:
        List of BoundingBox for detected satellites
        If return_spikes=True: (boxes, classification_spikes)
    """
    model.eval()

    with torch.no_grad():
        # Ensure correct shape: (T, B, C, H, W)
        if events.dim() == 4:
            events = events.unsqueeze(1)  # Add batch dim -> (T, 1, C, H, W)
        elif events.dim() == 5 and events.shape[0] != events.shape[1]:
            # Likely (B, T, C, H, W), convert to (T, B, C, H, W)
            events = events.permute(1, 0, 2, 3, 4)

        events = events.to(device)
        input_h, input_w = events.shape[-2], events.shape[-1]

        # Forward pass
        output = model(events)

        # Store raw spikes for 3D visualization (T, B, C, H, W)
        raw_spikes = output.classification_spikes.clone()

        # Get classification spikes: (T, B, C, H, W) -> sum over time
        class_spikes = output.classification_spikes.sum(dim=0)  # (B, C, H, W)

        # Process each batch item
        all_boxes = []
        for b in range(class_spikes.shape[0]):
            # Sum over channels to get spike map
            spike_map = class_spikes[b].sum(dim=0).cpu().numpy()  # (H, W)

            if spike_map.sum() == 0:
                continue

            # Scale factor from classification space to input space
            class_h, class_w = spike_map.shape
            scale_y = input_h / class_h
            scale_x = input_w / class_w

            # Find connected components
            binary_map = (spike_map > 0).astype(np.uint8)
            labeled_array, num_features = ndimage.label(binary_map)

            # Extract bounding box for each component
            for i in range(1, num_features + 1):
                coords = np.where(labeled_array == i)
                if len(coords[0]) < min_cluster_pixels:
                    continue

                # Bounding box in classification space
                y_min_class = coords[0].min()
                y_max_class = coords[0].max()
                x_min_class = coords[1].min()
                x_max_class = coords[1].max()

                # Scale to input space
                x_min = x_min_class * scale_x
                x_max = (x_max_class + 1) * scale_x
                y_min = y_min_class * scale_y
                y_max = (y_max_class + 1) * scale_y

                # Confidence based on spike density
                cluster_spikes = spike_map[coords].sum()
                confidence = min(1.0, cluster_spikes / 10.0)

                all_boxes.append(BoundingBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                    confidence=float(confidence)
                ))

        if return_spikes:
            return all_boxes, raw_spikes
        return all_boxes


def visualize_detections(
    events: torch.Tensor,
    boxes: List[BoundingBox],
    label: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None
):
    """Visualize detections on event frame."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Sum events over time and channels
    if events.dim() == 5:
        event_frame = events.sum(dim=(0, 1, 2)).cpu().numpy()
    elif events.dim() == 4:
        event_frame = events.sum(dim=(0, 1)).cpu().numpy()
    else:
        event_frame = events.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(event_frame, cmap='gray')

    # Draw detected boxes
    for box in boxes:
        rect = patches.Rectangle(
            (box.x_min, box.y_min),
            box.width, box.height,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.plot(*box.center, 'g+', markersize=10, markeredgewidth=2)

    # Draw ground truth if available
    if label is not None:
        label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        if label_np.sum() > 0:
            ax.contour(label_np, colors='cyan', linewidths=1, linestyles='dashed')

    ax.set_title(f'Detected: {len(boxes)} satellite(s)')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_3d_trajectory(
    events: torch.Tensor,
    predictions: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
    title: str = "Satellite Detection"
):
    """
    3D visualization like in the paper: X, Y, Time axes.

    Shows ground truth (cyan) and network output (red) trajectories.

    Args:
        events: Input events (T, C, H, W) or (T, B, C, H, W)
        predictions: Model output spikes (T, B, C, H, W) or (T, C, H, W)
        label: Ground truth mask (H, W) or (T, H, W)
        output_path: Path to save figure
        title: Figure title
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Get input events for ground truth extraction
    events_np = events.cpu().numpy() if isinstance(events, torch.Tensor) else events

    # Handle different event shapes to get (T, H, W)
    if events_np.ndim == 5:  # (T, B, C, H, W)
        events_np = events_np[:, 0, :, :, :]  # Take first batch -> (T, C, H, W)
    if events_np.ndim == 4:  # (T, C, H, W)
        events_np = events_np.sum(axis=1)  # Sum channels -> (T, H, W)

    T_input, H_input, W_input = events_np.shape

    # Process predictions to get spike locations over time
    pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions

    # Handle different prediction shapes
    if pred_np.ndim == 5:  # (T, B, C, H, W)
        pred_np = pred_np[:, 0, :, :, :]  # Take first batch
    if pred_np.ndim == 4:  # (T, C, H, W)
        pred_np = pred_np.sum(axis=1)  # Sum channels -> (T, H, W)

    T_pred, H_pred, W_pred = pred_np.shape

    # Scale factors from prediction space to input space
    scale_y = H_input / H_pred
    scale_x = W_input / W_pred

    # Extract prediction coordinates and SCALE to input space
    pred_coords = []
    for t in range(T_pred):
        ys, xs = np.where(pred_np[t] > 0)
        for y, x in zip(ys, xs):
            # Scale coordinates to input space
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            pred_coords.append([x_scaled, y_scaled, t])
    pred_coords = np.array(pred_coords) if pred_coords else np.empty((0, 3))

    # Extract ground truth from INPUT EVENTS filtered by label mask
    # This gives actual temporal trajectory, not static walls
    gt_coords = []
    if label is not None:
        label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label

        # Create a mask for the input space
        if label_np.ndim == 2:  # Static mask (H, W)
            # Use input events where the label mask indicates satellite presence
            for t in range(T_input):
                # Find where events occur AND label is positive
                event_frame = np.abs(events_np[t])  # Get event activity at timestep t
                # Mask events by label
                masked_events = event_frame * label_np
                # Get coordinates where there are events in labeled regions
                ys, xs = np.where(masked_events > 0)
                for y, x in zip(ys, xs):
                    gt_coords.append([x, y, t])
        elif label_np.ndim == 3:  # (T, H, W) temporal labels
            for t in range(min(T_input, label_np.shape[0])):
                ys, xs = np.where(label_np[t] > 0)
                for y, x in zip(ys, xs):
                    gt_coords.append([x, y, t])
    gt_coords = np.array(gt_coords) if gt_coords else np.empty((0, 3))

    # Use input space dimensions for plotting
    T, H, W = T_input, H_input, W_input

    # Create figure with 3D plot and 2D overview inset
    fig = plt.figure(figsize=(14, 10))

    # Main 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth (cyan squares)
    if len(gt_coords) > 0:
        # Subsample if too many points
        if len(gt_coords) > 5000:
            idx = np.random.choice(len(gt_coords), 5000, replace=False)
            gt_plot = gt_coords[idx]
        else:
            gt_plot = gt_coords
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2],
                   c='cyan', marker='s', s=8, alpha=0.6, label='Ground Truth')

    # Plot predictions (red +)
    if len(pred_coords) > 0:
        # Subsample if too many points
        if len(pred_coords) > 5000:
            idx = np.random.choice(len(pred_coords), 5000, replace=False)
            pred_plot = pred_coords[idx]
        else:
            pred_plot = pred_coords
        ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2],
                   c='red', marker='+', s=20, alpha=0.8, label='Network Output')

    # Labels and styling
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Time', fontsize=12, labelpad=10)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, T)

    # Invert Y axis to match image coordinates
    ax.invert_yaxis()

    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    # Set viewing angle similar to paper
    ax.view_init(elev=20, azim=-60)

    # Add 2D overview inset
    inset_ax = fig.add_axes([0.7, 0.1, 0.25, 0.25])

    # Create 2D overview (sum over time)
    if len(gt_coords) > 0:
        inset_ax.scatter(gt_coords[:, 0], gt_coords[:, 1],
                        c='cyan', marker='s', s=1, alpha=0.3)
    if len(pred_coords) > 0:
        inset_ax.scatter(pred_coords[:, 0], pred_coords[:, 1],
                        c='red', marker='+', s=2, alpha=0.5)

    inset_ax.set_xlim(0, W)
    inset_ax.set_ylim(H, 0)  # Invert Y
    inset_ax.set_title('Overview', fontsize=9)
    inset_ax.set_xlabel('X', fontsize=8)
    inset_ax.set_ylabel('Y', fontsize=8)
    inset_ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved 3D visualization: {output_path}")
    else:
        plt.show()

    plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(description='SpikeSEG Satellite Detection')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--input', '-i', help='Input .mat file for single inference')
    parser.add_argument('--data-root', '-d', help='EBSSA dataset root for batch inference')
    parser.add_argument('--output', '-o', default='detections.json', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Inference threshold')
    parser.add_argument('--visualize', action='store_true', help='Save 2D visualization images')
    parser.add_argument('--visualize-3d', action='store_true', help='Save 3D trajectory visualization (paper style)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.threshold)
    print(f"Model loaded with threshold={args.threshold}")

    results = []

    if args.data_root:
        # Batch inference on dataset
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
        )
        print(f"Dataset: {len(dataset)} samples")

        for i in range(len(dataset)):
            x, label = dataset[i]

            # x is (T, C, H, W), need (T, B, C, H, W)
            x = x.unsqueeze(1)

            # Get boxes and optionally raw spikes for 3D viz
            need_spikes = args.visualize_3d and i < 10
            result_data = detect_satellites(model, x, device, return_spikes=need_spikes)

            if need_spikes:
                boxes, raw_spikes = result_data
            else:
                boxes = result_data

            # Merge overlapping boxes
            boxes = merge_overlapping_boxes(boxes, iou_threshold=0.3)

            result = {
                'sample_id': i,
                'num_detections': len(boxes),
                'boxes': [b.to_dict() for b in boxes]
            }
            results.append(result)

            if args.visualize and i < 10:  # 2D visualization
                visualize_detections(
                    x, boxes, label,
                    output_path=f'detection_{i:03d}.png'
                )

            if args.visualize_3d and i < 10:  # 3D paper-style visualization
                visualize_3d_trajectory(
                    x, raw_spikes, label,
                    output_path=f'trajectory_3d_{i:03d}.png',
                    title=f'Sample {i}: Satellite Trajectory'
                )

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)}")

        print(f"\nTotal detections: {sum(r['num_detections'] for r in results)}")

    elif args.input:
        # Single file inference
        import scipy.io as sio

        mat = sio.loadmat(args.input, squeeze_me=True)
        # Extract events - adapt based on your .mat format
        if 'TD' in mat:
            td = mat['TD']
            # Convert to tensor format...
            print("Single file inference not fully implemented - use --data-root")
        else:
            print(f"Unknown .mat format. Keys: {list(mat.keys())}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {args.output}")


if __name__ == '__main__':
    main()
