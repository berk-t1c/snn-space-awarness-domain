#!/usr/bin/env python3
"""Sweep inference thresholds to find optimal value."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig
from scipy.ndimage import distance_transform_edt

def evaluate_threshold(model_weights, threshold, dataset, device):
    """Evaluate model with given threshold."""
    # Create model with specified threshold
    leak_ratios = [0.9, 0.1, 0.0]  # 90%, 10%, 0% of threshold
    leaks = [threshold * r for r in leak_ratios]
    
    enc_config = EncoderConfig(
        input_channels=2,
        conv1=LayerConfig(out_channels=4, kernel_size=5, threshold=threshold, leak=leaks[0]),
        conv2=LayerConfig(out_channels=36, kernel_size=5, threshold=threshold, leak=leaks[1]),
        conv3=LayerConfig(out_channels=1, kernel_size=7, threshold=threshold, leak=leaks[2]),
    )
    
    model = SpikeSEGEncoder(enc_config).to(device)
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(len(dataset)):
        x, label = dataset[i]
        if x.dim() == 4:
            x = x.unsqueeze(0)
        x = x.permute(1, 0, 2, 3, 4).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        class_spikes = output.classification_spikes[:, 0].cpu().numpy()
        spike_map = class_spikes.sum(axis=(0, 1))
        
        label_np = label.cpu().numpy() if label.dim() == 2 else label[0].cpu().numpy()
        gt_mask = (label_np > 0).astype(np.float32)
        
        if spike_map.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            scale_y = gt_mask.shape[0] / spike_map.shape[0]
            scale_x = gt_mask.shape[1] / spike_map.shape[1]
            spike_map = zoom(spike_map, (scale_y, scale_x), order=0)
        
        if gt_mask.sum() > 0:
            dist_map = distance_transform_edt(1 - gt_mask)
            true_region = dist_map <= 1.0
        else:
            true_region = np.zeros_like(gt_mask, dtype=bool)
        false_region = ~true_region
        
        total_spikes = spike_map.sum()
        total_pixels = spike_map.size
        global_density = total_spikes / total_pixels if total_pixels > 0 else 0
        
        true_spikes = spike_map[true_region].sum() if true_region.any() else 0
        true_pixels = true_region.sum()
        true_density = true_spikes / true_pixels if true_pixels > 0 else 0
        
        false_spikes = spike_map[false_region].sum() if false_region.any() else 0
        false_pixels = false_region.sum()
        false_density = false_spikes / false_pixels if false_pixels > 0 else 0
        
        if gt_mask.sum() == 0:
            if total_spikes > 0:
                fp += 1
            else:
                tn += 1
        else:
            if true_density > global_density:
                tp += 1
            else:
                fn += 1
            if false_density > global_density:
                fp += 1
            else:
                tn += 1
    
    eps = 1e-7
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    informedness = sensitivity + specificity - 1.0
    
    return {
        'threshold': threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'informedness': informedness
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint weights
    ckpt = torch.load('runs/spikeseg_20260122_080644/checkpoints/checkpoint_best.pt', 
                      map_location=device, weights_only=False)
    
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_state_dict[key[8:]] = value
        else:
            new_state_dict[key] = value
    
    # Load dataset
    dataset = EBSSADataset(root='../ebssa-data-utah/ebssa', sensor='all', n_timesteps=10)
    print(f"Dataset: {len(dataset)} samples\n")
    
    # Sweep thresholds
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    print(f"{'Threshold':>10} | {'Sens':>6} | {'Spec':>6} | {'Info':>6} | TP/TN/FP/FN")
    print("-" * 60)
    
    best_info = -1
    best_thresh = 0.1
    
    for thresh in thresholds:
        result = evaluate_threshold(new_state_dict, thresh, dataset, device)
        print(f"{thresh:>10.2f} | {result['sensitivity']*100:>5.1f}% | {result['specificity']*100:>5.1f}% | "
              f"{result['informedness']*100:>5.1f}% | {result['tp']}/{result['tn']}/{result['fp']}/{result['fn']}")
        
        if result['informedness'] > best_info:
            best_info = result['informedness']
            best_thresh = thresh
    
    print("-" * 60)
    print(f"Best threshold: {best_thresh} with informedness: {best_info*100:.1f}%")
    print(f"Target (IGARSS 2023): 89.1%")

if __name__ == '__main__':
    main()
