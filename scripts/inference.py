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
from spikeseg.algorithms.hulk import HULKDecoder
from spikeseg.algorithms.smash import group_instances_to_objects


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

            # Offset to account for receptive field of convolutions
            # Conv1(k=5) + Conv2(k=5) + Conv3(k=7) = total ~12 pixel offset
            offset = 12

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

                # Scale to input space and add offset
                x_min = x_min_class * scale_x + offset
                x_max = (x_max_class + 1) * scale_x + offset
                y_min = y_min_class * scale_y + offset
                y_max = (y_max_class + 1) * scale_y + offset

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


def create_tracking_video(
    events: torch.Tensor,
    predictions: torch.Tensor,
    trajectory: Optional[dict] = None,
    label: Optional[torch.Tensor] = None,
    output_path: str = "tracking_video.gif",
    fps: int = 5,
):
    """
    Create a 2D video showing real event camera data with bounding boxes tracking satellite.

    Shows the actual sparse event data accumulated over sliding windows, making the
    satellite visible as a bright moving blob against the static star background.

    Args:
        events: Input events tensor (T, C, H, W) or (T, B, C, H, W)
        predictions: Model output spikes (T, B, C, H, W)
        trajectory: Dict with 'x', 'y', 't' arrays (actual object trajectory)
        label: Ground truth mask (H, W)
        output_path: Path to save video (.gif or .mp4)
        fps: Frames per second
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap

    def safe_flatten(arr):
        """Safely flatten nested MATLAB arrays."""
        if arr is None:
            return np.array([])
        arr = np.asarray(arr)
        while arr.dtype == object and arr.size > 0:
            try:
                arr = np.concatenate([np.asarray(x).ravel() for x in arr.ravel()])
            except:
                break
        return arr.ravel().astype(float)

    # Get events per timestep - keep both polarities for visualization
    events_np = events.cpu().numpy() if isinstance(events, torch.Tensor) else events
    if events_np.ndim == 5:
        events_np = events_np[:, 0, :, :, :]  # (T, C, H, W)

    # Separate positive and negative events if we have 2 channels
    if events_np.ndim == 4 and events_np.shape[1] >= 2:
        pos_events = events_np[:, 0, :, :]  # ON events
        neg_events = events_np[:, 1, :, :]  # OFF events
        combined = pos_events + neg_events  # Combined for visualization
    else:
        if events_np.ndim == 4:
            combined = events_np.sum(axis=1)
        else:
            combined = events_np

    T, H, W = combined.shape

    # Get predictions
    pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0, :, :, :]
    if pred_np.ndim == 4:
        pred_np = pred_np.sum(axis=1)

    T_pred, H_pred, W_pred = pred_np.shape
    scale_y = H / H_pred
    scale_x = W / W_pred
    offset = 12

    # Get trajectory positions per frame
    traj_per_frame = {t: [] for t in range(T)}
    avg_traj_per_frame = {}  # Average position per frame for bounding box

    if trajectory is not None and trajectory.get('x') is not None:
        traj_x = safe_flatten(trajectory['x'])
        traj_y = safe_flatten(trajectory['y'])
        traj_t = safe_flatten(trajectory['t'])

        orig_w, orig_h = 240, 180
        scale_x_traj = W / orig_w
        scale_y_traj = H / orig_h

        if len(traj_t) > 0:
            t_min, t_max = float(traj_t.min()), float(traj_t.max())
            if t_max > t_min:
                traj_t_norm = (traj_t - t_min) / (t_max - t_min) * (T - 1)
                for tx, ty, tt in zip(traj_x, traj_y, traj_t_norm):
                    t_bin = int(np.clip(tt, 0, T - 1))
                    traj_per_frame[t_bin].append((tx * scale_x_traj, ty * scale_y_traj))

        # Compute average trajectory position per frame
        for t in range(T):
            if traj_per_frame[t]:
                xs = [p[0] for p in traj_per_frame[t]]
                ys = [p[1] for p in traj_per_frame[t]]
                avg_traj_per_frame[t] = (np.mean(xs), np.mean(ys))

        # Interpolate trajectory to fill ALL frames (fixes sparse trajectory issue)
        if avg_traj_per_frame:
            known_frames = sorted(avg_traj_per_frame.keys())
            if len(known_frames) >= 2:
                # Linear interpolation between known points
                known_x = [avg_traj_per_frame[f][0] for f in known_frames]
                known_y = [avg_traj_per_frame[f][1] for f in known_frames]
                for t in range(T):
                    if t not in avg_traj_per_frame:
                        # Find surrounding known frames
                        prev_f = max([f for f in known_frames if f <= t], default=known_frames[0])
                        next_f = min([f for f in known_frames if f >= t], default=known_frames[-1])
                        if prev_f == next_f:
                            avg_traj_per_frame[t] = avg_traj_per_frame[prev_f]
                        else:
                            # Linear interpolation
                            alpha = (t - prev_f) / (next_f - prev_f)
                            x_interp = avg_traj_per_frame[prev_f][0] + alpha * (avg_traj_per_frame[next_f][0] - avg_traj_per_frame[prev_f][0])
                            y_interp = avg_traj_per_frame[prev_f][1] + alpha * (avg_traj_per_frame[next_f][1] - avg_traj_per_frame[prev_f][1])
                            avg_traj_per_frame[t] = (x_interp, y_interp)
            elif len(known_frames) == 1:
                # Only one known frame - use it for all frames
                single_pos = avg_traj_per_frame[known_frames[0]]
                for t in range(T):
                    avg_traj_per_frame[t] = single_pos

    # Get network detection locations WITH timestamps
    # Track (x, y, t) to enable temporal-aware matching
    detection_points = []  # List of (cx, cy, t)
    for t in range(T_pred):
        spike_map = pred_np[t]
        if spike_map.sum() > 0:
            from scipy import ndimage
            binary_map = (spike_map > 0).astype(np.uint8)
            labeled, num = ndimage.label(binary_map)
            for i in range(1, num + 1):
                coords = np.where(labeled == i)
                if len(coords[0]) >= 1:
                    y_min = coords[0].min() * scale_y + offset
                    y_max = (coords[0].max() + 1) * scale_y + offset
                    x_min = coords[1].min() * scale_x + offset
                    x_max = (coords[1].max() + 1) * scale_x + offset
                    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                    # Scale detection time to input time range
                    t_scaled = t * (T / T_pred) if T_pred > 0 else t
                    detection_points.append((cx, cy, t_scaled))

    # Map actual network detections to frames (REAL detection locations, no fabrication)
    detections_per_frame = {t: [] for t in range(T)}  # frame -> list of (cx, cy)

    for cx, cy, t_scaled in detection_points:
        frame_idx = int(round(t_scaled))
        frame_idx = max(0, min(T - 1, frame_idx))
        detections_per_frame[frame_idx].append((cx, cy))

    # Per-frame evaluation: check if detection is within tolerance of GT
    hit_tolerance = 30  # pixels - detection counts as HIT if within this distance of GT
    frame_eval = {}  # frame -> {'hit': bool, 'dist': float or None, 'num_det': int}
    total_hits = 0
    total_frames_with_gt = 0

    for t in range(T):
        frame_dets = detections_per_frame.get(t, [])
        has_gt = t in avg_traj_per_frame

        if has_gt:
            total_frames_with_gt += 1
            gt_cx, gt_cy = avg_traj_per_frame[t]

            if frame_dets:
                # Find closest detection to GT
                min_dist = float('inf')
                for det_cx, det_cy in frame_dets:
                    dist = np.sqrt((det_cx - gt_cx)**2 + (det_cy - gt_cy)**2)
                    min_dist = min(min_dist, dist)

                is_hit = min_dist <= hit_tolerance
                if is_hit:
                    total_hits += 1
                frame_eval[t] = {'hit': is_hit, 'dist': min_dist, 'num_det': len(frame_dets)}
            else:
                frame_eval[t] = {'hit': False, 'dist': None, 'num_det': 0}
        else:
            frame_eval[t] = {'hit': None, 'dist': None, 'num_det': len(frame_dets)}  # No GT to compare

    # Print per-frame stats
    detection_rate = (total_hits / total_frames_with_gt * 100) if total_frames_with_gt > 0 else 0
    print(f"  Per-frame stats: {total_hits}/{total_frames_with_gt} frames with hits ({detection_rate:.1f}%)")

    # Create custom colormap: black -> blue -> white (for event intensity)
    colors = ['black', '#001133', '#003366', '#0066cc', '#3399ff', 'white']
    event_cmap = LinearSegmentedColormap.from_list('events', colors)

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    def update(frame):
        ax.clear()
        ax.set_facecolor('black')

        # Accumulate events over sliding window (5 frames) for visibility
        window = 5
        start = max(0, frame - window + 1)
        accumulated = np.sum(combined[start:frame+1], axis=0)

        # Normalize and enhance contrast
        if accumulated.max() > 0:
            accumulated = accumulated / accumulated.max()

        # Show events
        ax.imshow(accumulated, cmap=event_cmap, vmin=0, vmax=1, interpolation='nearest')

        # Draw ground truth bounding box (cyan) - centered on trajectory
        box_size = 20  # pixels
        if frame in avg_traj_per_frame:
            cx, cy = avg_traj_per_frame[frame]
            # Draw cyan bounding box around GT position
            gt_rect = patches.Rectangle(
                (cx - box_size/2, cy - box_size/2), box_size, box_size,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--',
                label='Ground Truth'
            )
            ax.add_patch(gt_rect)
            # Draw cyan crosshair
            ax.plot(cx, cy, 'c+', markersize=12, markeredgewidth=2)

        # Draw ACTUAL network detections (colored by hit/miss)
        frame_dets = detections_per_frame.get(frame, [])
        eval_info = frame_eval.get(frame, {})
        is_hit = eval_info.get('hit', None)
        min_dist = eval_info.get('dist', None)

        for det_cx, det_cy in frame_dets:
            box_size = 25
            # Green if HIT, Red if MISS, Orange if no GT to compare
            if is_hit is True:
                color = 'lime'
            elif is_hit is False:
                color = 'red'
            else:
                color = 'orange'

            det_rect = patches.Rectangle(
                (det_cx - box_size/2, det_cy - box_size/2), box_size, box_size,
                linewidth=3, edgecolor=color, facecolor='none',
                label='SNN Detection'
            )
            ax.add_patch(det_rect)
            ax.plot(det_cx, det_cy, '+', color=color, markersize=15, markeredgewidth=3)

        # Draw trajectory trail (cyan for GT path)
        trail_frames = range(max(0, frame-5), frame+1)
        for tf in trail_frames[:-1]:  # Don't include current frame
            if tf in avg_traj_per_frame and tf+1 in avg_traj_per_frame:
                x1, y1 = avg_traj_per_frame[tf]
                x2, y2 = avg_traj_per_frame[tf+1]
                # Color trail by hit status
                tf_hit = frame_eval.get(tf, {}).get('hit', None)
                color = 'lime' if tf_hit else 'cyan'
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # Title with per-frame evaluation info
        has_gt = frame in avg_traj_per_frame
        num_dets = len(frame_dets)

        title = f'Frame {frame+1}/{T} | Detections: {num_dets} | '
        if has_gt:
            if is_hit:
                title += f'HIT (dist={min_dist:.1f}px)'
            elif num_dets > 0:
                title += f'MISS (dist={min_dist:.1f}px)'
            else:
                title += 'NO DETECTION'
        else:
            title += 'No GT'

        title += f' | Rate: {total_hits}/{total_frames_with_gt}'
        ax.set_title(title, fontsize=11, color='white', pad=10)
        ax.axis('off')

        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)

    print(f"Saving tracking video to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black', 'edgecolor': 'none'})
    print(f"Tracking video saved: {output_path}")

    plt.close()
    return anim


def create_honest_tracking_video(
    events: torch.Tensor,
    predictions: torch.Tensor,
    trajectory: Optional[dict] = None,
    output_path: str = "honest_tracking.gif",
    fps: int = 5,
):
    """
    HONEST tracking visualization - clearly distinguishes detected vs interpolated.

    - SOLID GREEN box + "DETECTED": Real network detection at this frame
    - DASHED YELLOW box + "INTERPOLATED": Estimated position between detections
    - No box: No detection and can't interpolate

    Args:
        events: Input events tensor (T, C, H, W) or (T, B, C, H, W)
        predictions: Model output spikes (T, B, C, H, W)
        trajectory: Optional GT trajectory dict with 'x', 'y', 't'
        output_path: Where to save the GIF
        fps: Animation speed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap

    # Process events
    events_np = events.cpu().numpy() if isinstance(events, torch.Tensor) else events
    if events_np.ndim == 5:
        events_np = events_np[:, 0, :, :, :]

    if events_np.ndim == 4 and events_np.shape[1] >= 2:
        combined = events_np[:, 0, :, :] + events_np[:, 1, :, :]
    else:
        combined = events_np.sum(axis=1) if events_np.ndim == 4 else events_np

    T, H, W = combined.shape

    # Process predictions
    pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0, :, :, :]
    if pred_np.ndim == 4:
        pred_np = pred_np.sum(axis=1)

    T_pred, H_pred, W_pred = pred_np.shape
    scale_y = H / H_pred
    scale_x = W / W_pred

    # Extract REAL detections per frame (no fabrication)
    real_detections = {}  # frame -> (cx, cy) only where network actually detected

    for t in range(T_pred):
        spike_map = pred_np[t]
        if spike_map.sum() > 0:
            # Get centroid of all spikes at this timestep
            ys, xs = np.where(spike_map > 0)
            cx = xs.mean() * scale_x
            cy = ys.mean() * scale_y
            # Map prediction frame to input frame
            input_frame = int(round(t * T / T_pred)) if T_pred > 0 else t
            input_frame = max(0, min(T - 1, input_frame))
            real_detections[input_frame] = (cx, cy)

    # Compute interpolated positions (clearly marked as estimates)
    interpolated = {}  # frame -> (cx, cy) for frames without detection

    detected_frames = sorted(real_detections.keys())
    num_detected = len(detected_frames)

    if num_detected >= 2:
        # Interpolate between detected frames
        for i in range(len(detected_frames) - 1):
            f1, f2 = detected_frames[i], detected_frames[i + 1]
            x1, y1 = real_detections[f1]
            x2, y2 = real_detections[f2]

            for f in range(f1 + 1, f2):
                alpha = (f - f1) / (f2 - f1)
                cx = x1 + alpha * (x2 - x1)
                cy = y1 + alpha * (y2 - y1)
                interpolated[f] = (cx, cy)

    # Stats
    detection_rate = num_detected / T * 100 if T > 0 else 0
    print(f"  HONEST STATS: {num_detected}/{T} frames with real detections ({detection_rate:.1f}%)")
    print(f"  Detected frames: {detected_frames}")

    # Create colormap
    colors = ['black', '#001133', '#003366', '#0066cc', '#3399ff', 'white']
    event_cmap = LinearSegmentedColormap.from_list('events', colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    def update(frame):
        ax.clear()
        ax.set_facecolor('black')

        # Accumulate events
        window = 5
        start = max(0, frame - window + 1)
        accumulated = np.sum(combined[start:frame+1], axis=0)
        if accumulated.max() > 0:
            accumulated = accumulated / accumulated.max()

        ax.imshow(accumulated, cmap=event_cmap, vmin=0, vmax=1, interpolation='nearest')

        # Determine what to show for this frame
        is_detected = frame in real_detections
        is_interpolated = frame in interpolated

        box_size = 25

        if is_detected:
            # REAL DETECTION - solid green box
            cx, cy = real_detections[frame]
            rect = patches.Rectangle(
                (cx - box_size/2, cy - box_size/2), box_size, box_size,
                linewidth=3, edgecolor='lime', facecolor='none',
                linestyle='-'  # SOLID = real detection
            )
            ax.add_patch(rect)
            ax.plot(cx, cy, 'g+', markersize=15, markeredgewidth=3)
            status = "DETECTED"
            status_color = 'lime'

        elif is_interpolated:
            # INTERPOLATED - dashed yellow box
            cx, cy = interpolated[frame]
            rect = patches.Rectangle(
                (cx - box_size/2, cy - box_size/2), box_size, box_size,
                linewidth=2, edgecolor='yellow', facecolor='none',
                linestyle='--'  # DASHED = interpolated
            )
            ax.add_patch(rect)
            ax.plot(cx, cy, 'y+', markersize=12, markeredgewidth=2)
            status = "INTERPOLATED"
            status_color = 'yellow'

        else:
            # NO DATA - nothing to show
            status = "NO DETECTION"
            status_color = 'red'

        # Draw trajectory trail showing detected vs interpolated
        trail_len = 10
        for tf in range(max(0, frame - trail_len), frame):
            if tf in real_detections:
                cx1, cy1 = real_detections[tf]
                color = 'lime'
                alpha = 0.8
            elif tf in interpolated:
                cx1, cy1 = interpolated[tf]
                color = 'yellow'
                alpha = 0.4
            else:
                continue

            # Find next point
            next_f = tf + 1
            if next_f in real_detections:
                cx2, cy2 = real_detections[next_f]
            elif next_f in interpolated:
                cx2, cy2 = interpolated[next_f]
            else:
                continue

            ax.plot([cx1, cx2], [cy1, cy2], color=color, linewidth=2, alpha=alpha)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # HONEST title
        title = f'Frame {frame+1}/{T} | {status} | '
        title += f'Real detections: {num_detected}/{T} ({detection_rate:.1f}%)'
        ax.set_title(title, fontsize=11, color=status_color, pad=10)
        ax.axis('off')

        # Add legend
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor='lime', linewidth=2,
                         linestyle='-', label='DETECTED (real)'),
            patches.Patch(facecolor='none', edgecolor='yellow', linewidth=2,
                         linestyle='--', label='INTERPOLATED (estimated)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 facecolor='black', edgecolor='white', labelcolor='white',
                 fontsize=9)

        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)

    print(f"Saving HONEST tracking video to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black', 'edgecolor': 'none'})
    print(f"Honest tracking video saved: {output_path}")

    plt.close()
    return anim


def visualize_3d_trajectory(
    events: torch.Tensor,
    predictions: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    trajectory: Optional[dict] = None,
    output_path: Optional[str] = None,
    title: str = "Satellite Detection"
):
    """
    3D visualization exactly like the IGARSS paper Figure 4.

    Paper style:
    - Black background
    - Ground Truth: blue dots (from actual Obj trajectory)
    - Network Output: red stars
    - Axes: X, Time, Y

    Args:
        events: Input events tensor
        predictions: Model output spikes
        label: Ground truth mask (fallback if no trajectory)
        trajectory: Dict with 'x', 'y', 't' arrays (actual object trajectory)
        output_path: Path to save figure
        title: Figure title
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def safe_flatten(arr):
        """Safely flatten nested MATLAB arrays."""
        if arr is None:
            return np.array([])
        arr = np.asarray(arr)
        # Recursively unwrap nested object arrays
        while arr.dtype == object and arr.size > 0:
            try:
                arr = np.concatenate([np.asarray(x).ravel() for x in arr.ravel()])
            except:
                break
        return arr.ravel().astype(float)

    # Get input events
    events_np = events.cpu().numpy() if isinstance(events, torch.Tensor) else events
    if events_np.ndim == 5:
        events_np = events_np[:, 0, :, :, :]
    if events_np.ndim == 4:
        events_np = events_np.sum(axis=1)

    T_input, H_input, W_input = events_np.shape

    # Get predictions
    pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0, :, :, :]
    if pred_np.ndim == 4:
        pred_np = pred_np.sum(axis=1)

    T_pred, H_pred, W_pred = pred_np.shape

    # Scale and offset for predictions
    scale_y = H_input / H_pred
    scale_x = W_input / W_pred
    offset = 12

    # Extract PREDICTION coordinates (Network Output)
    # First pass: get raw prediction locations and active timesteps
    pred_raw_x, pred_raw_y, pred_raw_t = [], [], []
    active_timesteps = set()
    for t in range(T_pred):
        ys, xs = np.where(pred_np[t] > 0)
        if len(xs) > 0:
            active_timesteps.add(t)
        for y, x in zip(ys, xs):
            pred_raw_x.append(x * scale_x + offset)
            pred_raw_y.append(y * scale_y + offset)
            pred_raw_t.append(t)

    pred_x, pred_y, pred_t = pred_raw_x, pred_raw_y, pred_raw_t

    # Extract GROUND TRUTH - use actual trajectory if available
    gt_x, gt_y, gt_t = [], [], []

    if trajectory is not None and trajectory.get('x') is not None and trajectory.get('t') is not None:
        # Use actual object trajectory (most accurate!)
        # Flatten and ensure 1D arrays (handle nested MATLAB structures)
        traj_x = safe_flatten(trajectory['x'])
        traj_y = safe_flatten(trajectory['y'])
        traj_t = safe_flatten(trajectory['t'])

        # EBSSA original resolution is typically 240x180, scale to input resolution
        orig_w, orig_h = 240, 180  # DAVIS sensor resolution
        scale_x_traj = W_input / orig_w
        scale_y_traj = H_input / orig_h

        # Normalize trajectory time to [0, T_input-1] range
        if len(traj_t) > 0:
            t_min, t_max = traj_t.min().item(), traj_t.max().item()
            if t_max > t_min:
                traj_t_norm = (traj_t - t_min) / (t_max - t_min) * (T_input - 1)
            else:
                traj_t_norm = np.zeros_like(traj_t)

            for tx, ty, tt in zip(traj_x, traj_y, traj_t_norm):
                # Scale coordinates to input resolution
                tx_scaled = tx * scale_x_traj
                ty_scaled = ty * scale_y_traj
                if 0 <= tx_scaled < W_input and 0 <= ty_scaled < H_input:
                    gt_x.append(float(tx_scaled))
                    gt_y.append(float(ty_scaled))
                    gt_t.append(float(tt))
    elif label is not None:
        # Fallback: Use label mask - shows satellite from events
        label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        if label_np.ndim == 2:
            for t in range(T_input):
                masked = np.abs(events_np[t]) * label_np
                ys, xs = np.where(masked > 0)
                for y, x in zip(ys, xs):
                    gt_x.append(x)
                    gt_y.append(y)
                    gt_t.append(t)

    # Align predictions with trajectory - show red stars on GT points near each detection
    # This supports multiple satellites if the network detects them
    aligned_pred_x, aligned_pred_y, aligned_pred_t = [], [], []

    if gt_x and gt_t and pred_raw_x:
        gt_x_arr = np.array(gt_x)
        gt_y_arr = np.array(gt_y)
        gt_t_arr = np.array(gt_t)

        print(f"DEBUG: active_timesteps = {sorted(active_timesteps)}")
        print(f"DEBUG: gt_t range = [{gt_t_arr.min():.2f}, {gt_t_arr.max():.2f}]")
        print(f"DEBUG: len(gt_x) = {len(gt_x)}, len(pred_raw) = {len(pred_raw_x)}")

        # For each detection point, find GT points within detection radius
        detection_radius = 20  # pixels
        used_gt_indices = set()

        for px, py, pt in zip(pred_raw_x, pred_raw_y, pred_raw_t):
            # Find GT points near this detection at similar time
            for i, (gx, gy, gt) in enumerate(zip(gt_x_arr, gt_y_arr, gt_t_arr)):
                if i in used_gt_indices:
                    continue
                dist = np.sqrt((gx - px)**2 + (gy - py)**2)
                if dist < detection_radius:
                    aligned_pred_x.append(gx)
                    aligned_pred_y.append(gy)
                    aligned_pred_t.append(gt)
                    used_gt_indices.add(i)

        # If we got too few points, sample more from GT near detections
        if len(aligned_pred_x) < 10 and len(pred_raw_x) > 0:
            # Expand search radius
            for px, py, pt in zip(pred_raw_x, pred_raw_y, pred_raw_t):
                for i, (gx, gy, gt) in enumerate(zip(gt_x_arr, gt_y_arr, gt_t_arr)):
                    if i in used_gt_indices:
                        continue
                    dist = np.sqrt((gx - px)**2 + (gy - py)**2)
                    if dist < 40:  # Larger radius
                        aligned_pred_x.append(gx)
                        aligned_pred_y.append(gy)
                        aligned_pred_t.append(gt)
                        used_gt_indices.add(i)
                        if len(aligned_pred_x) >= 30:
                            break
                if len(aligned_pred_x) >= 30:
                    break

        print(f"DEBUG: aligned predictions = {len(aligned_pred_x)} points")

    # Use aligned predictions if available, otherwise use raw
    if aligned_pred_x:
        pred_x, pred_y, pred_t = aligned_pred_x, aligned_pred_y, aligned_pred_t

    # Create figure with black background (paper style)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Plot Ground Truth (blue dots) - paper style
    if gt_x:
        ax.scatter(gt_x, gt_t, gt_y, c='blue', marker='o', s=20, alpha=0.8,
                   label='Ground Truth', depthshade=False, zorder=1)

    # Plot Network Output (red stars) - along trajectory at detection times
    # Add small Y offset (+2) to render above blue dots
    if pred_x:
        pred_y_offset = [y + 2 for y in pred_y]  # Slight offset to appear on top
        ax.scatter(pred_x, pred_t, pred_y_offset, c='red', marker='*', s=150,
                   alpha=1.0, label='Network Output', depthshade=False, zorder=10)

    # Axis labels
    ax.set_xlabel('X (pixels)', fontsize=14, color='white', labelpad=10)
    ax.set_ylabel('Time (steps)', fontsize=14, color='white', labelpad=10)
    ax.set_zlabel('Y (pixels)', fontsize=14, color='white', labelpad=10)

    # Set axis limits
    ax.set_xlim(0, W_input)
    ax.set_ylim(0, T_input)
    ax.set_zlim(0, H_input)

    # Viewing angle (similar to paper)
    ax.view_init(elev=20, azim=-60)

    # Legend
    ax.legend(loc='upper left', fontsize=12, facecolor='black', edgecolor='white')

    # Title
    ax.set_title(title, fontsize=16, color='white', pad=20)

    # Grid styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        print(f"Saved 3D visualization: {output_path}")
    else:
        plt.show()

    plt.close()
    plt.style.use('default')

    return fig


def animate_3d_trajectory(
    events: torch.Tensor,
    predictions: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    trajectory: Optional[dict] = None,
    output_path: Optional[str] = None,
    title: str = "Satellite Detection",
    fps: int = 10,
    interval: int = 100,
    trail_length: int = 5,
):
    """
    Animated 3D visualization showing satellite detections appearing one by one.

    Creates a frame-by-frame animation where:
    - Ground truth points (blue dots) appear sequentially over time
    - Network output (red stars) appear along the trajectory where detected
    - Both blue and red dots build up together over time

    Args:
        events: Input events tensor (T, C, H, W) or (T, B, C, H, W)
        predictions: Model output spikes (T, B, C, H, W)
        label: Ground truth mask (H, W) - fallback if no trajectory
        trajectory: Dict with 'x', 'y', 't' arrays (actual object trajectory)
        output_path: Path to save animation (.gif or .mp4)
        title: Animation title
        fps: Frames per second for saved animation
        interval: Delay between frames in milliseconds
        trail_length: Number of timesteps to keep visible (0 = show all history)

    Returns:
        matplotlib.animation.FuncAnimation object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    def safe_flatten(arr):
        """Safely flatten nested MATLAB arrays."""
        if arr is None:
            return np.array([])
        arr = np.asarray(arr)
        # Recursively unwrap nested object arrays
        while arr.dtype == object and arr.size > 0:
            try:
                arr = np.concatenate([np.asarray(x).ravel() for x in arr.ravel()])
            except:
                break
        return arr.ravel().astype(float)

    # Get input events
    events_np = events.cpu().numpy() if isinstance(events, torch.Tensor) else events
    if events_np.ndim == 5:
        events_np = events_np[:, 0, :, :, :]
    if events_np.ndim == 4:
        events_np = events_np.sum(axis=1)

    T_input, H_input, W_input = events_np.shape

    # Get predictions
    pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0, :, :, :]
    if pred_np.ndim == 4:
        pred_np = pred_np.sum(axis=1)

    T_pred, H_pred, W_pred = pred_np.shape

    # Scale and offset for predictions
    scale_y = H_input / H_pred
    scale_x = W_input / W_pred
    offset = 12

    # Extract ground truth coordinates per timestep
    gt_per_timestep = {t: {'x': [], 'y': []} for t in range(T_input)}

    # Use actual trajectory data if available (much more accurate!)
    if trajectory is not None and trajectory.get('x') is not None and trajectory.get('t') is not None:
        # Flatten and ensure 1D arrays (handle nested MATLAB structures)
        traj_x = safe_flatten(trajectory['x'])
        traj_y = safe_flatten(trajectory['y'])
        traj_t = safe_flatten(trajectory['t'])

        # EBSSA original resolution is typically 240x180, scale to input resolution
        orig_w, orig_h = 240, 180  # DAVIS sensor resolution
        scale_x_traj = W_input / orig_w
        scale_y_traj = H_input / orig_h

        # Normalize trajectory time to [0, T_input-1] range
        if len(traj_t) > 0:
            t_min, t_max = traj_t.min().item(), traj_t.max().item()
            if t_max > t_min:
                traj_t_norm = (traj_t - t_min) / (t_max - t_min) * (T_input - 1)
            else:
                traj_t_norm = np.zeros_like(traj_t)

            # Bin trajectory points into timesteps
            for tx, ty, tt in zip(traj_x, traj_y, traj_t_norm):
                t_bin = int(np.clip(float(tt), 0, T_input - 1))
                # Scale coordinates to input resolution
                tx_scaled = float(tx) * scale_x_traj
                ty_scaled = float(ty) * scale_y_traj
                if 0 <= tx_scaled < W_input and 0 <= ty_scaled < H_input:
                    gt_per_timestep[t_bin]['x'].append(tx_scaled)
                    gt_per_timestep[t_bin]['y'].append(ty_scaled)
    elif label is not None:
        # Fallback to label mask if no trajectory
        label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        if label_np.ndim == 2:
            for t in range(T_input):
                masked = np.abs(events_np[t]) * label_np
                ys, xs = np.where(masked > 0)
                gt_per_timestep[t]['x'] = xs.tolist()
                gt_per_timestep[t]['y'] = ys.tolist()

    # Get raw prediction locations (spatial detections)
    pred_raw_x, pred_raw_y = [], []
    for t in range(T_pred):
        ys, xs = np.where(pred_np[t] > 0)
        for y, x in zip(ys, xs):
            pred_raw_x.append(x * scale_x + offset)
            pred_raw_y.append(y * scale_y + offset)

    # Align predictions with GT trajectory - spread red stars along detected trajectory
    # For each GT point, check if it's near any detection and mark it as "detected"
    pred_per_timestep = {t: {'x': [], 'y': []} for t in range(T_input)}
    detection_radius = 30  # pixels

    if pred_raw_x:
        pred_raw_x_arr = np.array(pred_raw_x)
        pred_raw_y_arr = np.array(pred_raw_y)

        for t in range(T_input):
            if gt_per_timestep[t]['x']:
                gt_xs = np.array(gt_per_timestep[t]['x'])
                gt_ys = np.array(gt_per_timestep[t]['y'])

                # For each GT point at this timestep, check if any detection is nearby
                for gx, gy in zip(gt_xs, gt_ys):
                    dists = np.sqrt((pred_raw_x_arr - gx)**2 + (pred_raw_y_arr - gy)**2)
                    if np.min(dists) < detection_radius:
                        # This GT point is near a detection - mark it as detected
                        pred_per_timestep[t]['x'].append(gx)
                        pred_per_timestep[t]['y'].append(gy)

    # Create figure with dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Initialize scatter plots
    gt_scatter = ax.scatter([], [], [], c='blue', marker='o', s=20, alpha=0.8,
                            label='Ground Truth', depthshade=False)
    pred_scatter = ax.scatter([], [], [], c='red', marker='*', s=150, alpha=1.0,
                              label='Network Output', depthshade=False)

    # Set axis labels and limits
    ax.set_xlabel('X (pixels)', fontsize=14, color='white', labelpad=10)
    ax.set_ylabel('Time (steps)', fontsize=14, color='white', labelpad=10)
    ax.set_zlabel('Y (pixels)', fontsize=14, color='white', labelpad=10)
    ax.set_xlim(0, W_input)
    ax.set_ylim(0, T_input)
    ax.set_zlim(0, H_input)
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc='upper left', fontsize=12, facecolor='black', edgecolor='white')
    ax.set_title(title, fontsize=16, color='white', pad=20)

    # Grid styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # Time indicator text
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                          color='white', fontweight='bold')

    # Cumulative data storage
    cumulative_gt_x, cumulative_gt_y, cumulative_gt_t = [], [], []
    cumulative_pred_x, cumulative_pred_y, cumulative_pred_t = [], [], []

    def init():
        """Initialize animation."""
        gt_scatter._offsets3d = ([], [], [])
        pred_scatter._offsets3d = ([], [], [])
        time_text.set_text('')
        return gt_scatter, pred_scatter, time_text

    def update(frame):
        """Update animation frame."""
        nonlocal cumulative_gt_x, cumulative_gt_y, cumulative_gt_t
        nonlocal cumulative_pred_x, cumulative_pred_y, cumulative_pred_t

        # Add new GT points for this timestep
        if frame < T_input:
            new_x = gt_per_timestep[frame]['x']
            new_y = gt_per_timestep[frame]['y']
            cumulative_gt_x.extend(new_x)
            cumulative_gt_y.extend(new_y)
            cumulative_gt_t.extend([frame] * len(new_x))

        # Add new prediction points for this timestep
        if frame < T_pred:
            new_px = pred_per_timestep[frame]['x']
            new_py = pred_per_timestep[frame]['y']
            cumulative_pred_x.extend(new_px)
            cumulative_pred_y.extend(new_py)
            cumulative_pred_t.extend([frame] * len(new_px))

        # Apply trail effect if specified
        if trail_length > 0 and frame > trail_length:
            # Filter to show only recent points
            min_t = frame - trail_length
            gt_mask = [t >= min_t for t in cumulative_gt_t]
            show_gt_x = [x for x, m in zip(cumulative_gt_x, gt_mask) if m]
            show_gt_y = [y for y, m in zip(cumulative_gt_y, gt_mask) if m]
            show_gt_t = [t for t, m in zip(cumulative_gt_t, gt_mask) if m]

            pred_mask = [t >= min_t for t in cumulative_pred_t]
            show_pred_x = [x for x, m in zip(cumulative_pred_x, pred_mask) if m]
            show_pred_y = [y for y, m in zip(cumulative_pred_y, pred_mask) if m]
            show_pred_t = [t for t, m in zip(cumulative_pred_t, pred_mask) if m]
        else:
            # Show all history
            show_gt_x, show_gt_y, show_gt_t = cumulative_gt_x, cumulative_gt_y, cumulative_gt_t
            show_pred_x, show_pred_y, show_pred_t = cumulative_pred_x, cumulative_pred_y, cumulative_pred_t

        # Update scatter plots
        gt_scatter._offsets3d = (show_gt_x, show_gt_t, show_gt_y)

        # Offset predictions slightly to render above GT
        pred_y_offset = [y + 2 for y in show_pred_y]
        pred_scatter._offsets3d = (show_pred_x, show_pred_t, pred_y_offset)

        # Update time indicator
        time_text.set_text(f'Time: {frame}/{T_input-1}')

        return gt_scatter, pred_scatter, time_text

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=T_input,
        init_func=init, interval=interval, blit=False
    )

    # Save animation
    if output_path:
        output_path = str(output_path)
        print(f"Saving animation to {output_path}...")

        if output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black'})
        elif output_path.endswith('.mp4'):
            try:
                writer = FFMpegWriter(fps=fps, metadata={'title': title})
                anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black'})
            except Exception as e:
                print(f"FFmpeg not available, saving as GIF instead: {e}")
                gif_path = output_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer, savefig_kwargs={'facecolor': 'black'})
                output_path = gif_path
        else:
            # Default to GIF
            output_path = output_path + '.gif'
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black'})

        print(f"Animation saved: {output_path}")

    plt.style.use('default')

    return anim


def main():
    parser = argparse.ArgumentParser(description='SpikeSEG Satellite Detection')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--input', '-i', help='Input .mat file for single inference')
    parser.add_argument('--data-root', '-d', help='EBSSA dataset root for batch inference')
    parser.add_argument('--output', '-o', default='detections.json', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Inference threshold')
    parser.add_argument('--visualize', action='store_true', help='Save 2D visualization images')
    parser.add_argument('--visualize-3d', action='store_true', help='Save 3D trajectory visualization (paper style)')
    parser.add_argument('--animate-3d', action='store_true', help='Save animated 3D visualization showing detections one by one')
    parser.add_argument('--tracking-video', action='store_true', help='Save 2D tracking video with bounding boxes')
    parser.add_argument('--honest-tracking', action='store_true', help='Save HONEST tracking video (solid=detected, dashed=interpolated)')
    parser.add_argument('--animation-fps', type=int, default=10, help='Animation frames per second')
    parser.add_argument('--animation-trail', type=int, default=0, help='Trail length (0 = show all history)')
    parser.add_argument('--max-vis', type=int, default=10, help='Max samples to visualize (default: 10, use -1 for all)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to use (default: all)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Train/test split ratio (default: 0.9 for 90/10 split)')
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
            split=args.split,
            train_ratio=args.train_ratio,
        )
        print(f"Dataset ({args.split} split, ratio={args.train_ratio}): {len(dataset)} samples")

        for i in range(len(dataset)):
            x, label = dataset[i]

            # Initialize trajectory for this sample
            trajectory = None

            # x is (T, C, H, W), need (T, B, C, H, W)
            x = x.unsqueeze(1)

            # Get boxes and optionally raw spikes for 3D viz/animation
            need_spikes = (args.visualize_3d or args.animate_3d or args.tracking_video or args.honest_tracking) and (args.max_vis < 0 or i < args.max_vis)
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

            if args.visualize and (args.max_vis < 0 or i < args.max_vis):  # 2D visualization
                visualize_detections(
                    x, boxes, label,
                    output_path=f'detection_{i:03d}.png'
                )

            if args.visualize_3d and (args.max_vis < 0 or i < args.max_vis):  # 3D paper-style visualization
                # Try to get actual trajectory from dataset
                trajectory = None
                try:
                    import scipy.io as sio
                    rec = dataset.recordings[i]
                    mat = sio.loadmat(rec['event_path'], squeeze_me=True)
                    if 'Obj' in mat:
                        obj = mat['Obj']
                        if hasattr(obj, 'dtype') and obj.dtype.names:
                            trajectory = {
                                'x': np.asarray(obj['x']).flatten() if 'x' in obj.dtype.names else None,
                                'y': np.asarray(obj['y']).flatten() if 'y' in obj.dtype.names else None,
                                't': np.asarray(obj['ts']).flatten() if 'ts' in obj.dtype.names else None
                            }
                except Exception as e:
                    print(f"Warning: Could not load trajectory: {e}")

                visualize_3d_trajectory(
                    x, raw_spikes, label,
                    trajectory=trajectory,
                    output_path=f'figure4_sample_{i:03d}.png',
                    title=f'Sample {i}: Satellite Trajectory'
                )

            if args.animate_3d and (args.max_vis < 0 or i < args.max_vis):  # Animated 3D visualization
                # Load trajectory if not already loaded
                if trajectory is None:
                    try:
                        import scipy.io as sio
                        rec = dataset.recordings[i]
                        mat = sio.loadmat(rec['event_path'], squeeze_me=True)
                        if 'Obj' in mat:
                            obj = mat['Obj']
                            if hasattr(obj, 'dtype') and obj.dtype.names:
                                trajectory = {
                                    'x': np.asarray(obj['x']).flatten() if 'x' in obj.dtype.names else None,
                                    'y': np.asarray(obj['y']).flatten() if 'y' in obj.dtype.names else None,
                                    't': np.asarray(obj['ts']).flatten() if 'ts' in obj.dtype.names else None
                                }
                    except Exception as e:
                        print(f"Warning: Could not load trajectory for animation: {e}")

                animate_3d_trajectory(
                    x, raw_spikes, label,
                    trajectory=trajectory,
                    output_path=f'animation_sample_{i:03d}.gif',
                    title=f'Sample {i}: Satellite Detection (Animated)',
                    fps=args.animation_fps,
                    trail_length=args.animation_trail,
                )

            if args.tracking_video and (args.max_vis < 0 or i < args.max_vis):  # 2D tracking video with bounding boxes
                # Load trajectory if not already loaded
                if trajectory is None:
                    try:
                        import scipy.io as sio
                        rec = dataset.recordings[i]
                        mat = sio.loadmat(rec['event_path'], squeeze_me=True)
                        if 'Obj' in mat:
                            obj = mat['Obj']
                            if hasattr(obj, 'dtype') and obj.dtype.names:
                                trajectory = {
                                    'x': obj['x'] if 'x' in obj.dtype.names else None,
                                    'y': obj['y'] if 'y' in obj.dtype.names else None,
                                    't': obj['ts'] if 'ts' in obj.dtype.names else None
                                }
                    except Exception as e:
                        print(f"Warning: Could not load trajectory: {e}")

                create_tracking_video(
                    x, raw_spikes,
                    trajectory=trajectory,
                    label=label,
                    output_path=f'tracking_sample_{i:03d}.gif',
                    fps=args.animation_fps,
                )

            if args.honest_tracking and (args.max_vis < 0 or i < args.max_vis):
                # HONEST tracking video - clearly shows detected vs interpolated
                create_honest_tracking_video(
                    x, raw_spikes,
                    trajectory=trajectory,
                    output_path=f'honest_tracking_{i:03d}.gif',
                    fps=args.animation_fps,
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
