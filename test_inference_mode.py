"""Quick test to compare spike counts with/without inference mode"""
import sys
sys.path.insert(0, '/home/user/snn-space-awarness-domain')

import torch
from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig
from spikeseg.data.ebssa import EBSSADataset

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

checkpoint = torch.load('/home/user/checkpoint_best.pt', map_location=device, weights_only=False)
config = EncoderConfig.from_paper("igarss2023", n_classes=2)
model = SpikeSEGEncoder(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load one sample
dataset = EBSSADataset(
    root_dir='/home/user/ebssa-data',
    n_time_bins=10,
    height=128, width=128,
    normalize=True, use_labels=True,
    windows_per_recording=1, split='all'
)
x, label = dataset[0]
x = x.unsqueeze(1).to(device)  # (T, 1, C, H, W)

print(f"\nInput shape: {x.shape}")
print(f"Timesteps: {x.shape[0]}")

# Test WITH fire-once (default)
print("\n=== WITH fire_once=True (default) ===")
with torch.no_grad():
    output1 = model(x, fire_once=True)
    spikes1 = output1.classification_spikes  # (T, B, C, H, W)
    
    total_spikes1 = spikes1.sum().item()
    spikes_per_t1 = [spikes1[t].sum().item() for t in range(spikes1.shape[0])]
    
    print(f"Total classification spikes: {total_spikes1}")
    print(f"Spikes per timestep: {spikes_per_t1}")

# Test WITHOUT fire-once (inference mode)
print("\n=== WITH fire_once=False (inference mode) ===")
with torch.no_grad():
    output2 = model(x, fire_once=False)
    spikes2 = output2.classification_spikes  # (T, B, C, H, W)
    
    total_spikes2 = spikes2.sum().item()
    spikes_per_t2 = [spikes2[t].sum().item() for t in range(spikes2.shape[0])]
    
    print(f"Total classification spikes: {total_spikes2}")
    print(f"Spikes per timestep: {spikes_per_t2}")

print(f"\n=== COMPARISON ===")
print(f"fire_once=True:  {total_spikes1} total spikes")
print(f"fire_once=False: {total_spikes2} total spikes")
print(f"Increase: {total_spikes2 - total_spikes1} more spikes ({(total_spikes2/max(total_spikes1,1)-1)*100:.1f}% increase)")
