#!/usr/bin/env python3
"""Simple test: does fire_once=False produce more spikes?"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from spikeseg.data.datasets import EBSSADataset
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig, LayerConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load checkpoint - try multiple paths
for ckpt_path in [
    os.path.expanduser('~/checkpoint_best.pt'),
    './checkpoint_best.pt',
    '../checkpoint_best.pt',
]:
    if os.path.exists(ckpt_path):
        break
else:
    print("ERROR: checkpoint_best.pt not found")
    sys.exit(1)

print(f"Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg = ckpt.get('config', {})
m = cfg.get('model', {})

# Build model
enc_config = EncoderConfig(
    input_channels=cfg.get('data', {}).get('input_channels', 2),
    conv1=LayerConfig(out_channels=m.get('conv1_channels', 4), kernel_size=5, threshold=0.05, leak=0.045),
    conv2=LayerConfig(out_channels=m.get('conv2_channels', 36), kernel_size=5, threshold=0.05, leak=0.005),
    conv3=LayerConfig(out_channels=m.get('n_classes', 1), kernel_size=7, threshold=0.05, leak=0.0),
)
model = SpikeSEGEncoder(enc_config).to(device)

# Load weights
sd = {k.replace('encoder.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()}, strict=False)
model.eval()
print("Model loaded")

# Load one sample - try multiple data paths
for data_path in [
    os.path.expanduser('~/ebssa-data'),
    os.path.expanduser('~/ebssa-data-utah/ebssa'),
    '../ebssa-data',
    './data/EBSSA',
]:
    if os.path.exists(data_path):
        break
else:
    print("ERROR: EBSSA data not found")
    sys.exit(1)

print(f"Data: {data_path}")
ds = EBSSADataset(root=data_path, n_timesteps=20, height=128, width=128, split='all')
x, _ = ds[0]
x = x.unsqueeze(1).to(device)
print(f"Input: {x.shape}")

# Test 1: fire_once=True
print("\n=== fire_once=True ===")
model.reset_state()
with torch.no_grad():
    out1 = model(x, fire_once=True)
s1 = out1.classification_spikes
per_t1 = [s1[t].sum().item() for t in range(s1.shape[0])]
print(f"Spikes per timestep: {per_t1}")
print(f"TOTAL: {sum(per_t1):.0f}")

# Test 2: fire_once=False
print("\n=== fire_once=False ===")
model.reset_state()
with torch.no_grad():
    out2 = model(x, fire_once=False)
s2 = out2.classification_spikes
per_t2 = [s2[t].sum().item() for t in range(s2.shape[0])]
print(f"Spikes per timestep: {per_t2}")
print(f"TOTAL: {sum(per_t2):.0f}")

# Result
print("\n" + "="*50)
if sum(per_t2) > sum(per_t1):
    print(f"SUCCESS: {sum(per_t2):.0f} > {sum(per_t1):.0f} spikes")
else:
    print(f"SAME: {sum(per_t2):.0f} vs {sum(per_t1):.0f}")
