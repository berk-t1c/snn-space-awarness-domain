#!/usr/bin/env python3
"""Diagnostic script to visualize what's wrong with model predictions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig
from scipy.ndimage import distance_transform_edt

def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = ckpt.get('config', {})
    model_cfg = config_dict.get('model', {})

    n_classes = model_cfg.get('n_classes', 1)
    thresholds = model_cfg.get('thresholds', [0.1, 0.1, 0.1])
    leaks = model_cfg.get('leaks', [0.09, 0.01, 0.0])
    conv1_channels = model_cfg.get('conv1_channels', 4)
    conv2_channels = model_cfg.get('conv2_channels', 36)
    kernels = model_cfg.get('kernel_sizes', [5, 5, 7])
    input_channels = config_dict.get('data', {}).get('input_channels', 2)

    enc_config = EncoderConfig(
        input_channels=input_channels,
        conv1=LayerConfig(out_channels=conv1_channels, kernel_size=kernels[0], threshold=thresholds[0], leak=leaks[0]),
        conv2=LayerConfig(out_channels=conv2_channels, kernel_size=kernels[1], threshold=thresholds[1], leak=leaks[1]),
        conv3=LayerConfig(out_channels=n_classes, kernel_size=kernels[2], threshold=thresholds[2], leak=leaks[2]),
    )

    model = SpikeSEGEncoder(enc_config).to(device)

    # Handle state dict with 'encoder.' prefix from training
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[len('encoder.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Filter to only include model parameters
    param_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in param_keys}

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    return model

def diagnose(checkpoint_path: str, data_root: str, n_samples: int = 5):
    """Run diagnostic on model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)
    print(f"Model loaded")

    # Load dataset
    dataset = EBSSADataset(root=data_root, sensor='all', n_timesteps=10)
    print(f"Dataset: {len(dataset)} samples")

    # Create figure
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    print("\n" + "="*80)
    print("  DIAGNOSTIC ANALYSIS")
    print("="*80)

    for i in range(min(n_samples, len(dataset))):
        x, label = dataset[i]

        # Ensure correct shape
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dim
        x = x.permute(1, 0, 2, 3, 4).to(device)  # (T, B, C, H, W)
        label = label.to(device)

        with torch.no_grad():
            output = model(x)

        # Get spike data
        class_spikes = output.classification_spikes[:, 0].cpu().numpy()  # (T, C, H, W)
        spike_map = class_spikes.sum(axis=(0, 1))  # Sum over time and channels -> (H, W)

        # Get input events
        input_events = x[:, 0].sum(dim=(0, 1)).cpu().numpy()  # (H, W)

        # Get label
        label_np = label.cpu().numpy() if label.dim() == 2 else label[0].cpu().numpy()

        # Create GT mask and distance map
        gt_mask = (label_np > 0).astype(np.float32)
        if gt_mask.sum() > 0:
            distance_map = distance_transform_edt(1 - gt_mask)
            true_mask = distance_map <= 1.0
            false_mask = ~true_mask
        else:
            true_mask = np.zeros_like(gt_mask, dtype=bool)
            false_mask = np.ones_like(gt_mask, dtype=bool)

        # Scale spike map to label size if needed
        if spike_map.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            scale_y = gt_mask.shape[0] / spike_map.shape[0]
            scale_x = gt_mask.shape[1] / spike_map.shape[1]
            spike_map_scaled = zoom(spike_map, (scale_y, scale_x), order=0)
        else:
            spike_map_scaled = spike_map

        # Compute densities
        total_spikes = spike_map_scaled.sum()
        total_pixels = spike_map_scaled.size
        global_density = total_spikes / total_pixels if total_pixels > 0 else 0

        true_spikes = spike_map_scaled[true_mask].sum() if true_mask.any() else 0
        true_pixels = true_mask.sum()
        true_density = true_spikes / true_pixels if true_pixels > 0 else 0

        false_spikes = spike_map_scaled[false_mask].sum() if false_mask.any() else 0
        false_pixels = false_mask.sum()
        false_density = false_spikes / false_pixels if false_pixels > 0 else 0

        # Classification
        is_tp = true_density > global_density
        is_fp = false_density > global_density

        print(f"\nSample {i}:")
        print(f"  Input events: {input_events.sum():.0f}")
        print(f"  Output spikes: {total_spikes:.0f}")
        print(f"  Spike map shape: {spike_map.shape} -> scaled to {spike_map_scaled.shape}")
        print(f"  GT pixels: {gt_mask.sum():.0f}, TRUE region: {true_mask.sum():.0f} px, FALSE region: {false_mask.sum():.0f} px")
        print(f"  Densities: global={global_density:.4f}, true={true_density:.4f}, false={false_density:.4f}")
        print(f"  Result: {'TP' if is_tp else 'FN'} + {'FP' if is_fp else 'TN'}")
        print(f"  Spikes in TRUE region: {true_spikes:.0f}, in FALSE region: {false_spikes:.0f}")

        # Plot
        ax = axes[i]

        # 1. Input events
        ax[0].imshow(input_events, cmap='hot')
        ax[0].set_title(f'Sample {i}: Input Events\n({input_events.sum():.0f} total)')
        ax[0].axis('off')

        # 2. Ground truth
        ax[1].imshow(label_np, cmap='gray')
        ax[1].set_title(f'Ground Truth\n({gt_mask.sum():.0f} px)')
        ax[1].axis('off')

        # 3. Output spikes (raw)
        ax[2].imshow(spike_map, cmap='hot')
        ax[2].set_title(f'Output Spikes (raw)\n({total_spikes:.0f} total, shape={spike_map.shape})')
        ax[2].axis('off')

        # 4. Output spikes scaled with GT overlay
        ax[3].imshow(spike_map_scaled, cmap='hot')
        # Overlay GT contour
        if gt_mask.sum() > 0:
            ax[3].contour(gt_mask, colors='cyan', linewidths=1)
        ax[3].set_title(f'Spikes + GT\ntrue_d={true_density:.3f}, false_d={false_density:.3f}')
        ax[3].axis('off')

        # 5. Classification result
        result_img = np.zeros((*spike_map_scaled.shape, 3))
        # Green = TP region with spikes, Red = FP region with spikes
        spike_binary = spike_map_scaled > 0
        result_img[true_mask & spike_binary] = [0, 1, 0]  # Green = good (TP area)
        result_img[false_mask & spike_binary] = [1, 0, 0]  # Red = bad (FP area)
        result_img[true_mask & ~spike_binary] = [0, 0.3, 0]  # Dark green = missed

        ax[4].imshow(result_img)
        status = f"{'TP' if is_tp else 'FN'}+{'FP' if is_fp else 'TN'}"
        ax[4].set_title(f'Result: {status}\nGreen=good, Red=FP')
        ax[4].axis('off')

    plt.tight_layout()
    plt.savefig('diagnostic_output.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic_output.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='runs/spikeseg_20260122_080644/checkpoints/checkpoint_best.pt')
    parser.add_argument('--data-root', default='../ebssa-data-utah/ebssa')
    parser.add_argument('--n-samples', type=int, default=5)
    args = parser.parse_args()

    diagnose(args.checkpoint, args.data_root, args.n_samples)
