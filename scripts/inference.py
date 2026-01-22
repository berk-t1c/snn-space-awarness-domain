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
) -> List[BoundingBox]:
    """
    Run inference and return detected satellite bounding boxes.

    Args:
        model: Loaded SpikeSEG model
        events: Input tensor (T, C, H, W) or (B, T, C, H, W)
        device: Compute device
        min_cluster_pixels: Minimum cluster size to count as detection

    Returns:
        List of BoundingBox for detected satellites
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


def main():
    parser = argparse.ArgumentParser(description='SpikeSEG Satellite Detection')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--input', '-i', help='Input .mat file for single inference')
    parser.add_argument('--data-root', '-d', help='EBSSA dataset root for batch inference')
    parser.add_argument('--output', '-o', default='detections.json', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Inference threshold')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
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

            boxes = detect_satellites(model, x, device)

            result = {
                'sample_id': i,
                'num_detections': len(boxes),
                'boxes': [b.to_dict() for b in boxes]
            }
            results.append(result)

            if args.visualize and i < 10:  # Visualize first 10
                visualize_detections(
                    x, boxes, label,
                    output_path=f'detection_{i:03d}.png'
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
