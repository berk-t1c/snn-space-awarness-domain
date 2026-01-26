# SpikeSEG: Spiking Neural Network for Event-Based Space Domain Awareness

A PyTorch implementation of biologically-inspired spiking neural networks for satellite detection and instance segmentation using event camera data. This implementation combines methodologies from multiple peer-reviewed publications to provide a complete pipeline for neuromorphic space situational awareness.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
   - [Spiking Neural Networks](#spiking-neural-networks)
   - [STDP Learning Rule](#stdp-learning-rule)
   - [Winner-Take-All Competition](#winner-take-all-competition)
3. [Architecture](#architecture)
   - [Encoder Network](#encoder-network)
   - [Decoder Network](#decoder-network)
   - [HULK-SMASH Instance Segmentation](#hulk-smash-instance-segmentation)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage](#usage)
   - [Quick Start](#quick-start)
   - [Training](#training)
   - [Inference](#inference)
   - [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Datasets](#datasets)
10. [Current Capabilities](#current-capabilities)
11. [Known Limitations](#known-limitations)
12. [Roadmap](#roadmap)
13. [Citation](#citation)
14. [References](#references)
15. [License](#license)

---

## Overview

SpikeSEG implements a three-layer spiking convolutional neural network trained using Spike-Timing Dependent Plasticity (STDP), an unsupervised learning rule inspired by biological synaptic plasticity. The system processes asynchronous event streams from neuromorphic vision sensors to detect and segment satellites in space domain awareness applications.

### Key Features

- **Biologically Plausible Learning**: Unsupervised STDP-based training without backpropagation
- **Event-Driven Processing**: Native support for asynchronous event camera data
- **Instance Segmentation**: HULK-SMASH algorithm for separating multiple objects
- **Temporal Dynamics**: Leaky Integrate-and-Fire (LIF) neurons with configurable leak rates
- **Competitive Learning**: Winner-Take-All lateral inhibition with homeostatic plasticity
- **Saliency Mapping**: Decoder network traces classification spikes back to pixel space

### Implemented Papers

This implementation synthesizes methods from the following publications:

| Paper | Year | Contribution |
|-------|------|--------------|
| Kheradpisheh et al. [1] | 2018 | STDP learning rule, network architecture |
| Kirkland et al. [2] | 2020 | SpikeSEG encoder-decoder, saliency mapping |
| Kirkland et al. [3] | 2022 | HULK-SMASH instance segmentation |
| Kirkland et al. [4] | 2023 | Space domain awareness, layer-wise leak |

---

## Theoretical Background

### Spiking Neural Networks

Spiking Neural Networks (SNNs) represent the third generation of neural network models, incorporating temporal dynamics that more closely approximate biological neural computation [5]. Unlike rate-coded artificial neurons, spiking neurons communicate through discrete events (spikes) whose precise timing carries information.

#### Neuron Models

This implementation provides two neuron models:

**Integrate-and-Fire (IF) Neuron**

The simplest spiking neuron model accumulates input current until reaching a threshold:

```
V(t) = V(t-1) + I(t)

if V(t) >= theta:
    spike = 1
    V(t) = 0  (reset)
```

**Leaky Integrate-and-Fire (LIF) Neuron**

The LIF neuron adds membrane potential decay, providing temporal filtering. This implementation supports two leak modes as described in the literature:

*Subtractive Mode* (IGARSS 2023 [4]):
```
V(t) = V(t-1) + I(t) - lambda
```

*Multiplicative Mode* (Kheradpisheh 2018 [1]):
```
V(t) = beta * V(t-1) + I(t)
```

Where:
- `V(t)` is the membrane potential at time t
- `I(t)` is the input current (weighted sum of incoming spikes)
- `lambda` is the subtractive leak term
- `beta` is the multiplicative decay factor (0 < beta < 1)
- `theta` is the firing threshold

The IGARSS 2023 paper specifies layer-wise leak configuration:
> "lambda is set to 90% and 10% of the neuron threshold in layers 1 and 2 respectively, using a 5x5 convolution kernel and a 7x7 final classification kernel with no leakage." [4]

### STDP Learning Rule

Spike-Timing Dependent Plasticity is a biologically-observed phenomenon where synaptic strength changes based on the relative timing of pre- and post-synaptic spikes [6]. This implementation follows the simplified STDP rule from Kheradpisheh et al. [1]:

**Long-Term Potentiation (LTP)** - Pre-synaptic spike precedes post-synaptic spike:
```
Delta_w = a_plus * w * (1 - w)    if t_pre <= t_post
```

**Long-Term Depression (LTD)** - Pre-synaptic spike follows post-synaptic spike:
```
Delta_w = -a_minus * w * (1 - w)   if t_pre > t_post
```

Where:
- `a_plus` and `a_minus` are learning rates
- `w` is the current synaptic weight
- The multiplicative term `w * (1 - w)` provides soft bounds in [0, 1]

From Kheradpisheh et al. [1]:
> "The exact time difference between two spikes does not affect the weight change, but only its sign is considered. Also, it is assumed that if a presynaptic neuron does not fire before the postsynaptic one, it will fire later."

#### Learning Rate Parameters

| Source | a_plus | a_minus | Application |
|--------|--------|---------|-------------|
| Kheradpisheh 2018 [1] | 0.004 | 0.003 | Original paper |
| IGARSS 2023 [4] | 0.04 | 0.03 | Faster convergence |

#### Convergence Criterion

Training continues until weights converge to binary values (0 or 1). The convergence metric from Equation 4 of [1]:

```
C_l = (1/n_w) * sum_f sum_i w_{f,i} * (1 - w_{f,i})
```

Training stops when `C_l < 0.01`, indicating weights have polarized.

### Winner-Take-All Competition

Winner-Take-All (WTA) lateral inhibition enforces competition between neurons, ensuring feature diversity [1]. Two complementary inhibition mechanisms operate simultaneously:

**Global Intra-Map Competition**: Within each feature map, the first neuron to fire inhibits all others in the same map. This ensures each feature is learned by exactly one spatial location per stimulus.

**Local Inter-Map Competition**: At each spatial location, firing neurons inhibit nearby neurons in other feature maps within a local radius. This encourages different features to activate at different locations.

From Kheradpisheh et al. [1]:
> "We use a winner-take-all (WTA) mechanism to enforce competition among neurons. The first neuron that fires inhibits the others, preventing them from firing and receiving plasticity updates."

#### Homeostatic Plasticity

To prevent dead neurons (neurons that never fire) and overactive neurons (neurons that dominate), adaptive threshold mechanisms maintain balanced activity:

```
theta_new = theta_old + theta_plus    (after spike)
theta_new = theta_old - (theta_old - theta_rest) / tau_theta    (decay toward rest)
```

---

## Architecture

### Network Overview

```
Input Events --> [Encoder] --> Classification Spikes --> [Decoder] --> Saliency Map
                                      |
                                      v
                              [HULK] --> Instance Masks --> [SMASH] --> Object Groups
```

### Encoder Network

The encoder extracts hierarchical spatio-temporal features through three spiking convolutional layers with interleaved max pooling:

```
Input (1, H, W)
    |
    v
Conv1: 5x5, 4 features, LIF (leak=90%)
    |
    v
Pool1: 2x2 max pooling (indices saved)
    |
    v
Conv2: 5x5, 36 features, LIF (leak=10%)
    |
    v
Pool2: 2x2 max pooling (indices saved)
    |
    v
Conv3: 7x7, n_classes features, IF (leak=0%)
    |
    v
Classification Spikes (n_classes, H', W')
```

#### Layer Specifications (IGARSS 2023)

| Layer | Kernel | Channels | Threshold | Leak | Neuron |
|-------|--------|----------|-----------|------|--------|
| Conv1 | 5x5 | 4 | 10.0 | 9.0 (90%) | LIF |
| Conv2 | 5x5 | 36 | 10.0 | 1.0 (10%) | LIF |
| Conv3 | 7x7 | n_classes | 10.0 | 0.0 (0%) | IF |

#### Weight Initialization

Weights are initialized from a normal distribution centered near 1 to accelerate STDP convergence:
- Mean: 0.8
- Standard deviation: 0.01

This initialization is based on the observation that STDP with soft bounds drives weights toward 0 or 1, so starting near 1 reduces training time for features that should be potentiated.

### Decoder Network

The decoder maps classification spikes back to pixel space using transposed convolutions with tied weights from the encoder:

```
Classification Spikes (n_classes, H', W')
    |
    v
TransConv3: 7x7 (tied to Conv3)
    |
    v
Unpool2: 2x2 (using saved indices)
    |
    v
TransConv2: 5x5 (tied to Conv2)
    |
    v
Unpool1: 2x2 (using saved indices)
    |
    v
TransConv1: 5x5 (tied to Conv1)
    |
    v
Saliency Map (1, H, W)
```

From Kirkland et al. [2]:
> "The decoder uses tied weights from the encoder, transposed convolutions, and max unpooling with saved indices to trace the causal pathway from classification spikes back to the input pixels that contributed to activation."

### HULK-SMASH Instance Segmentation

The HULK-SMASH algorithm [3] extends semantic segmentation to instance segmentation by processing each classification spike individually.

#### HULK: Hierarchical Unravelling of Linked Kernels

Instead of decoding all classification spikes together, HULK decodes each spike separately to reveal which input pixels contributed to that specific detection:

```python
for each classification_spike in encoder_output:
    pixel_mask = decoder.unravel(spike_location, pooling_indices)
    spike_activity = record_intermediate_spikes()
    instances.append(Instance(mask, spike_activity))
```

From Kirkland et al. [3]:
> "The Hierarchical Unravelling of Linked Kernels (HULK) process permits spiking activity from the classification convolution layer, to be tracked as it propagates through the decoding layers of HULK."

#### ASH: Active Spike Hashing

ASH compresses the 4D spike activity (x, y, feature, time) into a 2D binary matrix (feature, time) for efficient similarity comparison:

```
hash_matrix[feature_id, timestep] = 1  if feature fired at timestep
                                   = 0  otherwise
```

From Kirkland et al. [3]:
> "The ASH process then results in a 2D featural-temporal hashing of the spiking activity. This essentially works as a spike train with each feature neuron activity over time being mapped out."

#### SMASH Score

The SMASH score combines featural-temporal similarity with spatial proximity:

```
SMASH(i, j) = Jaccard(ASH_i, ASH_j) * IoU(BBox_i, BBox_j)
```

Where:
- `Jaccard(ASH_i, ASH_j)` measures similarity of spike patterns
- `IoU(BBox_i, BBox_j)` measures spatial overlap of bounding boxes

Instances with SMASH scores above a threshold are grouped into the same object.

---

## Mathematical Foundations

### Membrane Potential Dynamics

For a LIF neuron with subtractive leak:

```
dV/dt = I(t) - lambda

V[n+1] = V[n] + I[n] - lambda  (discrete time)
```

### Spike Generation

The spike output at time t is determined by:

```
S(t) = H(V(t) - theta)
```

Where H is the Heaviside step function and theta is the firing threshold.

### STDP Weight Update

For a synapse connecting pre-synaptic neuron j to post-synaptic neuron i:

```
Delta_w_ij = {
    a_plus * w_ij * (1 - w_ij)    if t_j - t_i <= 0  (causal)
    -a_minus * w_ij * (1 - w_ij)  if t_j - t_i > 0   (anti-causal)
}
```

### Convolution Operation

The spiking convolution at layer l:

```
I_l[n, c, h, w] = sum_{c'} sum_{kh} sum_{kw} W_l[c, c', kh, kw] * S_{l-1}[n, c', h+kh, w+kw]
```

Where S is the binary spike tensor from the previous layer.

### Pooling with Index Preservation

Max pooling preserves indices for unpooling:

```
y[n, c, h, w], idx[n, c, h, w] = max_{(kh, kw) in kernel} x[n, c, h*s+kh, w*s+kw]
```

### Jaccard Similarity for ASH

For binary ASH matrices:

```
J(A, B) = |A AND B| / |A OR B|
```

Computed efficiently using logical operations on binary tensors.

---

## Installation

### Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended)

### From Source

```bash
git clone https://github.com/berk-t1c/snn-space-awareness-domain.git
cd snn-space-awareness-domain
pip install -e .
```

### Dependencies

Core dependencies are installed automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| numpy | >=1.24.0 | Numerical computing |
| scipy | >=1.10.0 | Scientific computing |
| h5py | >=3.8.0 | HDF5 file handling |
| opencv-python | >=4.7.0 | Image processing |
| matplotlib | >=3.7.0 | Visualization |
| pyyaml | >=6.0 | Configuration parsing |
| tqdm | >=4.65.0 | Progress bars |
| scikit-learn | >=1.2.0 | Machine learning utilities |

### Optional Dependencies

For development and visualization:

```bash
pip install -e ".[dev]"  # Testing and linting tools
pip install -e ".[vis]"  # TensorBoard and plotting
pip install -e ".[all]"  # Everything
```

---

## Project Structure

```
snn-space-awareness-domain/
|-- spikeseg/                    # Main package
|   |-- __init__.py
|   |-- config.py                # Configuration management
|   |
|   |-- core/                    # Core neural network components
|   |   |-- neurons.py           # IF and LIF neuron implementations
|   |   |-- layers.py            # Spiking Conv2d, Pool2d, TransposedConv2d
|   |   +-- functional.py        # Stateless operations (spike_fn, DoG, Gabor)
|   |
|   |-- models/                  # Network architectures
|   |   |-- encoder.py           # SpikeSEG encoder (Conv1->Pool1->Conv2->Pool2->Conv3)
|   |   |-- decoder.py           # SpikeSEG decoder (tied weights, unpooling)
|   |   +-- spikeseg.py          # Complete SpikeSEG model
|   |
|   |-- learning/                # Learning algorithms
|   |   |-- stdp.py              # STDP learning rule implementation
|   |   +-- wta.py               # Winner-Take-All lateral inhibition
|   |
|   |-- algorithms/              # Instance segmentation
|   |   |-- hulk.py              # Hierarchical Unravelling of Linked Kernels
|   |   +-- smash.py             # Similarity Matching through Active Spike Hashing
|   |
|   |-- data/                    # Data handling
|   |   |-- datasets.py          # EBSSA, N-MNIST dataset classes
|   |   |-- events.py            # Event processing and encoding
|   |   +-- preprocessing.py     # DoG filters, Gabor banks, temporal encoding
|   |
|   +-- utils/                   # Utilities
|       |-- logging.py           # Logging and TensorBoard integration
|       +-- visualization.py     # Plotting utilities
|
|-- scripts/                     # Executable scripts
|   |-- train.py                 # Training script
|   |-- train_cv.py              # Cross-validation training
|   |-- evaluate.py              # Model evaluation
|   |-- inference.py             # Run inference on new data
|   +-- demo.py                  # Quick demonstration
|
|-- configs/                     # Configuration files
|   +-- config.yaml              # Default training configuration
|
|-- tests/                       # Unit tests
|   |-- test_neurons.py
|   |-- test_stdp.py
|   +-- test_model.py
|
|-- pyproject.toml               # Package configuration
|-- requirements.txt             # Dependencies
+-- README.md                    # This file
```

---

## Usage

### Quick Start

```python
import torch
from spikeseg.models import SpikeSEG

# Create model with IGARSS 2023 parameters
model = SpikeSEG.from_paper("igarss2023", n_classes=1)

# Process event sequence
model.reset_state()
input_events = torch.randn(1, 1, 128, 128)  # (batch, channels, height, width)

for t in range(10):  # 10 timesteps
    segmentation, encoder_output = model(input_events)

    if encoder_output.has_spikes:
        print(f"Timestep {t}: {encoder_output.n_classification_spikes} detections")
```

### Training

#### Using Configuration File

```bash
python scripts/train.py --config configs/config.yaml
```

#### Programmatic Training

```python
from spikeseg.models import SpikeSEG
from spikeseg.learning import STDPLearner, STDPConfig, WTAInhibition
from spikeseg.data import EBSSADataset

# Initialize model
model = SpikeSEG.from_paper("igarss2023", n_classes=1)

# Initialize STDP learner
stdp_config = STDPConfig.from_paper("igarss2023")
learner = STDPLearner(stdp_config)

# Initialize WTA
wta = WTAInhibition(mode="both", n_channels=36, spatial_shape=(32, 32))

# Load dataset
dataset = EBSSADataset(
    root="./data/EBSSA",
    split="train",
    n_timesteps=10
)

# Training loop
for epoch in range(50):
    for events, labels in dataset:
        model.reset_state()

        for t in range(10):
            output = model.encode(events[t])

            # Apply WTA and get winner
            filtered_spikes, winner_mask = wta(output.layer_spikes['conv2'])

            # Update weights via STDP
            if winner_mask.any():
                learner.update_weights(
                    model.encoder.conv2.conv.weight,
                    pre_spike_times=output.layer_spike_times['conv1'],
                    post_spike_times=output.layer_spike_times['conv2'],
                    winner_mask=winner_mask
                )

        # Check convergence
        if learner.has_converged():
            print("Training converged")
            break
```

### Inference

```python
from spikeseg.models import SpikeSEG
from spikeseg.algorithms import HULKDecoder, group_instances_to_objects

# Load trained model
model = SpikeSEG.from_paper("igarss2023", n_classes=1)
model.load_state_dict(torch.load("checkpoint.pth"))

# Create HULK decoder for instance segmentation
hulk = HULKDecoder.from_encoder(model.encoder)

# Process input
model.reset_state()
encoder_output = model.encode(input_events)

# Decode each classification spike individually (instance segmentation)
instances = []
for spike_location in encoder_output.get_spike_locations():
    instance = hulk.unravel_spike(
        spike_location=spike_location,
        pool1_indices=encoder_output.pooling_indices.pool1_indices,
        pool2_indices=encoder_output.pooling_indices.pool2_indices
    )
    instances.append(instance)

# Group instances into objects using SMASH scores
objects = group_instances_to_objects(instances, smash_threshold=0.1)

print(f"Detected {len(objects)} objects from {len(instances)} instances")
```

### Configuration

Configuration is managed through YAML files. Key parameters:

```yaml
# Model Architecture
model:
  n_classes: 1              # Number of output classes
  conv1_channels: 4         # First layer features
  conv2_channels: 36        # Second layer features (STDP-learned)
  kernel_sizes: [5, 5, 7]   # Kernel sizes per layer
  thresholds: [10.0, 10.0, 10.0]  # Firing thresholds
  leaks: [9.0, 1.0, 0.0]    # Leak values (90%, 10%, 0% of threshold)

# STDP Learning
stdp:
  lr_plus: 0.04             # LTP learning rate
  lr_minus: 0.03            # LTD learning rate
  weight_init_mean: 0.8     # Initial weight mean
  weight_init_std: 0.01     # Initial weight std

# Winner-Take-All
wta:
  mode: "both"              # "global", "local", or "both"
  local_radius: 2           # Radius for local inhibition
  enable_homeostasis: true  # Adaptive thresholds

# Data
data:
  dataset: "ebssa"          # Dataset name
  n_timesteps: 10           # Timesteps per sample
  input_height: 128         # Input dimensions
  input_width: 128
```

---

## API Reference

### Core Modules

#### `spikeseg.core.neurons`

| Class | Description |
|-------|-------------|
| `IFNeuron` | Integrate-and-Fire neuron without leak |
| `LIFNeuron` | Leaky Integrate-and-Fire neuron with configurable leak mode |
| `create_neuron` | Factory function for neuron creation |

```python
from spikeseg.core.neurons import LIFNeuron

neuron = LIFNeuron(
    threshold=10.0,
    leak_factor=0.9,
    leak_mode="subtractive"  # or "multiplicative"
)

spikes, membrane, pre_reset = neuron(input_current, membrane)
```

#### `spikeseg.core.layers`

| Class | Description |
|-------|-------------|
| `SpikingConv2d` | Spiking convolutional layer with integrated neuron |
| `SpikingPool2d` | Max pooling with index preservation |
| `SpikingUnpool2d` | Max unpooling using saved indices |
| `SpikingTransposedConv2d` | Transposed convolution for decoder |

#### `spikeseg.models`

| Class | Description |
|-------|-------------|
| `SpikeSEG` | Complete model with encoder and decoder |
| `SpikeSEGEncoder` | Feature extraction encoder |
| `SpikeSEGDecoder` | Saliency mapping decoder |
| `EncoderConfig` | Encoder configuration dataclass |
| `EncoderOutput` | Encoder output container |

```python
from spikeseg.models import SpikeSEG, EncoderConfig

# From paper preset
model = SpikeSEG.from_paper("igarss2023", n_classes=1)

# From custom config
config = EncoderConfig(
    input_channels=1,
    conv1=LayerConfig(out_channels=4, kernel_size=5, threshold=10.0, leak=9.0),
    conv2=LayerConfig(out_channels=36, kernel_size=5, threshold=10.0, leak=1.0),
    conv3=LayerConfig(out_channels=1, kernel_size=7, threshold=10.0, leak=0.0)
)
model = SpikeSEG(config)
```

#### `spikeseg.learning`

| Class | Description |
|-------|-------------|
| `STDPLearner` | STDP weight update manager |
| `STDPConfig` | STDP configuration with paper presets |
| `WTAInhibition` | Winner-Take-All lateral inhibition |
| `AdaptiveThreshold` | Homeostatic threshold adaptation |
| `ConvergenceTracker` | Monitor training convergence |

```python
from spikeseg.learning import STDPLearner, STDPConfig

config = STDPConfig.from_paper("kheradpisheh2018")
learner = STDPLearner(config)

# Update weights
learner.update_weights(
    weights=layer.conv.weight,
    pre_spike_times=pre_times,
    post_spike_times=post_times,
    winner_mask=winner
)

# Check convergence
if learner.get_convergence_metric() < 0.01:
    print("Converged")
```

#### `spikeseg.algorithms`

| Class/Function | Description |
|----------------|-------------|
| `HULKDecoder` | Instance-wise spike unraveling |
| `ActiveSpikeHash` | 2D featural-temporal hashing |
| `BoundingBox` | Axis-aligned bounding box |
| `Instance` | Single detection instance |
| `Object` | Grouped object from instances |
| `compute_smash_score` | SMASH similarity score |
| `group_instances_to_objects` | Instance clustering |

#### `spikeseg.data`

| Class | Description |
|-------|-------------|
| `EBSSADataset` | Event-Based Space Situational Awareness dataset |
| `NMNISTDataset` | Neuromorphic MNIST for benchmarking |
| `EventData` | Event container dataclass |
| `SpykeTorchPreprocessor` | Image-to-spike preprocessing |
| `SpikeSEGPreprocessor` | Event stream preprocessing |

---

## Datasets

### EBSSA: Event-Based Space Situational Awareness

The EBSSA dataset [7] contains event camera recordings of satellites:

- **Sensors**: ATIS (304x240) and DAVIS240C (240x180)
- **Recordings**: 84 labelled, 153 unlabelled
- **Format**: MATLAB .mat files with (x, y, polarity, timestamp) events
- **Labels**: Bounding box annotations

```python
from spikeseg.data import EBSSADataset

dataset = EBSSADataset(
    root="/path/to/EBSSA",
    split="train",
    sensor="all",  # "atis", "davis", or "all"
    n_timesteps=10,
    height=128,
    width=128
)
```

### N-MNIST: Neuromorphic MNIST

The N-MNIST dataset [8] provides spiking digit recognition for benchmarking:

- **Resolution**: 34x34 pixels
- **Classes**: 10 digits (0-9)
- **Format**: Binary event files

---

## Current Capabilities

This implementation provides a complete pipeline for neuromorphic satellite detection:

### Detection and Tracking

| Feature | Status | Description |
|---------|--------|-------------|
| Satellite Detection | Working | Detects moving satellites in event camera streams |
| Multi-Object Tracking | Working | Tracks multiple satellites simultaneously via HULK-SMASH |
| Bounding Box Output | Working | Returns bounding boxes with confidence scores |
| Real-Time Processing | Working | Processes event streams at sensor frame rate |

### Visualization

| Feature | Status | Description |
|---------|--------|-------------|
| 2D Detection Overlay | Working | Bounding boxes overlaid on event frames |
| 3D Trajectory Plot | Working | X-Y-Time visualization (IGARSS paper style) |
| 3D Animated Visualization | Working | Frame-by-frame animation showing detections over time |
| Saliency Maps | Working | Pixel-level attribution from decoder |
| STDP Filter Visualization | Working | Learned Gabor-like features |

### Training

| Feature | Status | Description |
|---------|--------|-------------|
| Unsupervised STDP | Working | No labels required for feature learning |
| Layer-wise Training | Working | Train Conv1, Conv2 sequentially |
| Convergence Monitoring | Working | Automatic stop when weights converge |
| Cross-Validation | Working | K-fold CV with proper train/test splits |

### Data Formats

| Format | Status | Description |
|--------|--------|-------------|
| EBSSA (.mat) | Working | Native support for EBSSA dataset |
| N-MNIST (.bin) | Working | Neuromorphic MNIST for benchmarking |
| NumPy (.npy) | Working | Generic event arrays |
| HDF5 (.h5) | Working | Large-scale event storage |

---

## Known Limitations

### Training Data Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Positive samples only** | Model trained exclusively on images containing satellites | May produce false positives on empty sky or other moving objects |
| **No negative mining** | No deliberate training on background-only samples | Consider post-processing with trajectory validation |
| **Single domain** | Trained on EBSSA dataset sensors only | May require fine-tuning for different event cameras |

### Algorithmic Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Motion saliency detector** | STDP learns general motion patterns, not satellite-specific features | Suitable for space domain where most moving points are satellites |
| **Requires temporal coherence** | Objects must persist across multiple timesteps | Not suitable for single-frame detection |
| **Fixed receptive field** | Kernel sizes optimized for satellite scale in EBSSA | May need adjustment for different satellite distances/sizes |

### Performance Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No GPU-optimized STDP** | Training uses CPU-bound weight updates | Training is unsupervised and relatively fast regardless |
| **Sequential timestep processing** | Cannot parallelize across time dimension | Inherent to recurrent SNN architecture |
| **Memory for pooling indices** | Must store indices for unpooling in decoder | Memory scales with spatial resolution |

### Evaluation Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No false positive rate metric** | Cannot quantify spurious detections | Recommend testing on background-only sequences |
| **Limited cross-dataset validation** | Performance on other event camera datasets unknown | Community contributions welcome |

---

## Roadmap

### Near-Term Improvements

- [ ] **Negative sample augmentation**: Generate synthetic background-only samples for false positive testing
- [ ] **Confidence calibration**: Improve confidence scores based on spike density and temporal consistency
- [ ] **Multi-scale detection**: Support satellites at varying distances/apparent sizes
- [ ] **Online learning**: Adapt STDP weights during inference for domain shift

### Medium-Term Goals

- [ ] **GPU-accelerated STDP**: CUDA kernels for faster training
- [ ] **Streaming inference API**: Process continuous event streams without batching
- [ ] **ROS2 integration**: Publish detections as ROS messages for robotics applications
- [ ] **ONNX export**: Deploy to edge devices and neuromorphic hardware

### Long-Term Vision

- [ ] **Neuromorphic hardware deployment**: Intel Loihi, IBM TrueNorth, SynSense Speck
- [ ] **Multi-sensor fusion**: Combine event cameras with traditional sensors
- [ ] **Orbital mechanics integration**: Use Keplerian elements for trajectory prediction
- [ ] **Comprehensive benchmark suite**: Standardized evaluation across multiple datasets

### Community Contributions Welcome

We welcome contributions in the following areas:

1. **Additional datasets**: Event camera recordings from other sources
2. **Performance benchmarks**: FPS/latency measurements on various hardware
3. **Bug reports**: Issues with specific edge cases or data formats
4. **Documentation**: Tutorials, examples, and translations

---

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{kheradpisheh2018stdp,
  title={STDP-based spiking deep convolutional neural networks for object recognition},
  author={Kheradpisheh, Saeed Reza and Ganjtabesh, Mohammad and Thorpe, Simon J and Masquelier, Timoth{\'e}e},
  journal={Neural Networks},
  volume={99},
  pages={56--67},
  year={2018},
  publisher={Elsevier}
}

@inproceedings{kirkland2020spikeseg,
  title={SpikeSEG: Spiking segmentation via STDP saliency mapping},
  author={Kirkland, Paul and Di Caterina, Gaetano and Soraghan, John and Andreopoulos, Yiannis and Matich, George},
  booktitle={2020 IEEE International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2020},
  organization={IEEE}
}

@article{kirkland2022hulksmash,
  title={Unsupervised spiking instance segmentation on event data using STDP features},
  author={Kirkland, Paul and Di Caterina, Gaetano and Soraghan, John and Andreopoulos, Yiannis and Matich, George},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}

@inproceedings{kirkland2023igarss,
  title={Neuromorphic sensing and processing for space domain awareness},
  author={Kirkland, Paul and others},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  year={2023},
  organization={IEEE}
}
```

---

## References

[1] S. R. Kheradpisheh, M. Ganjtabesh, S. J. Thorpe, and T. Masquelier, "STDP-based spiking deep convolutional neural networks for object recognition," *Neural Networks*, vol. 99, pp. 56-67, 2018.

[2] P. Kirkland, G. Di Caterina, J. Soraghan, Y. Andreopoulos, and G. Matich, "SpikeSEG: Spiking segmentation via STDP saliency mapping," in *2020 IEEE International Joint Conference on Neural Networks (IJCNN)*, 2020, pp. 1-8.

[3] P. Kirkland, G. Di Caterina, J. Soraghan, Y. Andreopoulos, and G. Matich, "Unsupervised spiking instance segmentation on event data using STDP features," *IEEE Transactions on Neural Networks and Learning Systems*, 2022.

[4] P. Kirkland et al., "Neuromorphic sensing and processing for space domain awareness," in *IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium*, 2023.

[5] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659-1671, 1997.

[6] G.-Q. Bi and M.-M. Poo, "Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type," *Journal of Neuroscience*, vol. 18, no. 24, pp. 10464-10472, 1998.

[7] S. Afshar et al., "Event-based object detection and tracking for space situational awareness," *IEEE Sensors Journal*, vol. 20, no. 24, pp. 15117-15132, 2020.

[8] G. Orchard, A. Jayawant, G. K. Cohen, and N. Thakor, "Converting static image datasets to spiking neuromorphic datasets using saccades," *Frontiers in Neuroscience*, vol. 9, p. 437, 2015.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

This implementation builds upon the foundational work of Kheradpisheh et al. and Kirkland et al. The EBSSA dataset is provided by the Western Sydney University Neuromorphic Systems Lab.

For questions or issues, please open a GitHub issue or contact the maintainers.
