#!/usr/bin/env python3
"""
Plot STDP Training Progress from logs.

Generates publication-quality visualization of convergence and spike activity.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for dark theme (matching your presentation)
plt.style.use('dark_background')

# Data from training log
conv2_epochs = list(range(1, 21))
conv2_convergence = [2.8, 19.4, 38.9, 72.2, 72.2, 72.2, 75.0, 75.0, 75.0, 77.8,
                     80.6, 80.6, 80.6, 80.6, 83.3, 83.3, 83.3, 86.1, 86.1, 91.7]
conv2_spikes = [82165, 24349, 20738, 21219, 20202, 20523, 19749, 20672, 20978, 20109,
                19787, 19597, 19396, 19574, 19120, 19009, 18559, 17210, 18151, 16996]

conv3_epochs = [21]
conv3_convergence = [100.0]
conv3_spikes = [1252]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
fig.suptitle('SpikeSEG STDP Training Progress', fontsize=16, fontweight='bold', color='white')

# Color scheme
conv2_color = '#4fd1c5'  # Teal
conv3_color = '#f6ad55'  # Orange
threshold_color = '#ecc94b'  # Yellow

# ============================================================
# Plot 1: Convergence Progress
# ============================================================
ax1.set_facecolor('#1a202c')

# Conv2 convergence
ax1.plot(conv2_epochs, conv2_convergence, 'o-', color=conv2_color,
         linewidth=2, markersize=6, label='Conv2 (36 features)')

# Conv3 convergence (continues from epoch 20)
all_epochs = conv2_epochs + conv3_epochs
all_convergence = conv2_convergence + conv3_convergence
ax1.plot([20, 21], [91.7, 100.0], 'o-', color=conv3_color,
         linewidth=2, markersize=6, label='Conv3 (1 feature)')

# Convergence threshold line
ax1.axhline(y=90, color=threshold_color, linestyle='--', linewidth=1.5,
            label='Target (90%)', alpha=0.7)

# Annotations
ax1.annotate('Conv2 Converged\n91.7%', xy=(20, 91.7), xytext=(17, 80),
            fontsize=10, color=conv2_color,
            arrowprops=dict(arrowstyle='->', color=conv2_color, lw=1.5))
ax1.annotate('Conv3 Converged\n100%', xy=(21, 100), xytext=(19, 95),
            fontsize=10, color=conv3_color,
            arrowprops=dict(arrowstyle='->', color=conv3_color, lw=1.5))

ax1.set_ylabel('Convergence (%)', fontsize=12, color='white')
ax1.set_ylim(0, 105)
ax1.set_xlim(0, 22)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Weight Convergence (C_l < 0.01)', fontsize=12, color='#a0aec0')

# Add layer transition marker
ax1.axvline(x=20, color='white', linestyle=':', alpha=0.5)
ax1.text(20.2, 50, 'Layer\nTransition', fontsize=9, color='white', alpha=0.7)

# ============================================================
# Plot 2: Spike Activity
# ============================================================
ax2.set_facecolor('#1a202c')

# Conv2 spikes (bar chart)
bars1 = ax2.bar(conv2_epochs, np.array(conv2_spikes)/1000, color=conv2_color,
                alpha=0.8, label='Conv2', width=0.8)

# Conv3 spikes
bars2 = ax2.bar(conv3_epochs, np.array(conv3_spikes)/1000, color=conv3_color,
                alpha=0.8, label='Conv3', width=0.8)

# Highlight first epoch spike explosion
ax2.annotate('Initial spike\nexplosion', xy=(1, 82), xytext=(4, 70),
            fontsize=10, color='white',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

# Highlight stabilization
ax2.annotate('Stabilized\n~20k spikes', xy=(10, 20), xytext=(13, 40),
            fontsize=10, color='white',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

ax2.set_xlabel('Epoch', fontsize=12, color='white')
ax2.set_ylabel('Spikes (thousands)', fontsize=12, color='white')
ax2.set_xlim(0, 22)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('Spike Activity per Epoch', fontsize=12, color='#a0aec0')

# Add layer transition marker
ax2.axvline(x=20, color='white', linestyle=':', alpha=0.5)

# ============================================================
# Add summary text box
# ============================================================
summary_text = """Training Summary:
• Total time: 3h 8m (11,288s)
• Conv2: 20 epochs → 91.7% converged
• Conv3: 1 epoch → 100% converged
• Hardware: NVIDIA H100 (85GB)
• Samples: 84 recordings"""

props = dict(boxstyle='round', facecolor='#2d3748', edgecolor='#4a5568', alpha=0.9)
fig.text(0.02, 0.02, summary_text, fontsize=9, color='white',
         verticalalignment='bottom', bbox=props, family='monospace')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save figure
plt.savefig('docs/training_progress.png', dpi=150, facecolor='#1a202c',
            edgecolor='none', bbox_inches='tight')
plt.savefig('docs/training_progress.svg', facecolor='#1a202c',
            edgecolor='none', bbox_inches='tight')

print("Saved: docs/training_progress.png")
print("Saved: docs/training_progress.svg")

# Also show
plt.show()
