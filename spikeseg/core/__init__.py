"""
Core building blocks for Spiking Neural Networks.

This module provides the fundamental components for building SNNs:

- **Neurons**: Spiking neuron models (IF, LIF) with configurable dynamics
- **Layers**: Spiking convolution, pooling, unpooling, and transposed convolution
- **Functional**: Stateless operations for spikes, filters, encoding, and WTA

Architecture (from Kirkland et al. papers):

    ENCODER:
        Input Events → Conv1 → Pool1 → Conv2 → Pool2 → Conv3 (Classification)
                       (5×5)   (2×2)   (5×5)   (2×2)   (7×7)
                       4 feat  store   36 feat store   C classes
                               indices         indices
    
    DECODER:
        Conv3 → Unpool2 → TransConv2 → Unpool1 → TransConv1 → Pixel Mask
                (use idx)  (tied wts)  (use idx)  (tied wts)

Paper References:
    - Kheradpisheh et al. 2018: STDP-based deep convolutional SNN
    - Kirkland et al. 2020 (SpikeSEG): Encoder-decoder for event segmentation
    - Kirkland et al. 2022 (HULK-SMASH): Instance segmentation via spike decoding  
    - Kirkland et al. 2023 (IGARSS): Modified leak for space domain awareness

Example:
    >>> from spikeseg.core import LIFNeuron, SpikingConv2d, create_dog_filters
    >>> 
    >>> # Create a spiking convolutional layer
    >>> conv = SpikingConv2d(
    ...     in_channels=2, out_channels=4,
    ...     kernel_size=5, neuron_type="lif",
    ...     threshold=15.0, leak_factor=0.5
    ... )
    >>> 
    >>> # Create DoG filters for preprocessing
    >>> dog_on, dog_off = create_dog_filters(size=7, sigma_center=1.0, sigma_surround=2.0)
"""

# =============================================================================
# NEURONS: Spiking Neuron Models
# =============================================================================

from .neurons import (
    # Type alias
    LeakMode,
    # Spike function
    spike_function,
    # Neuron classes
    BaseNeuron,
    IFNeuron,
    LIFNeuron,
    # Factory
    create_neuron,
)

# =============================================================================
# LAYERS: Spiking Neural Network Layers
# =============================================================================

from .layers import (
    SpikingConv2d,
    SpikingPool2d,
    SpikingUnpool2d,
    SpikingTransposedConv2d,
)

# =============================================================================
# FUNCTIONAL: Stateless Operations
# =============================================================================

from .functional import (
    # Spike functions
    spike_fn,
    soft_spike_fn,
    # Membrane dynamics
    if_step,
    lif_step,
    lif_step_subtractive,
    lif_step_multiplicative,
    # Filters (preprocessing)
    create_gaussian_kernel,
    create_dog_filters,
    create_gabor_filters,
    # Temporal encoding
    intensity_to_latency,
    latency_to_spikes,
    encode_image_to_spikes,
    # Winner-Take-All
    wta_global,
    wta_local,
    # Utilities
    compute_output_size,
    count_spikes,
)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # =========================================================================
    # Neurons
    # =========================================================================
    "LeakMode",
    "spike_function",
    "BaseNeuron",
    "IFNeuron",
    "LIFNeuron",
    "create_neuron",
    
    # =========================================================================
    # Layers
    # =========================================================================
    "SpikingConv2d",
    "SpikingPool2d",
    "SpikingUnpool2d",
    "SpikingTransposedConv2d",
    
    # =========================================================================
    # Functional - Spike Functions
    # =========================================================================
    "spike_fn",
    "soft_spike_fn",
    
    # =========================================================================
    # Functional - Membrane Dynamics
    # =========================================================================
    "if_step",
    "lif_step",
    "lif_step_subtractive",
    "lif_step_multiplicative",
    
    # =========================================================================
    # Functional - Filters
    # =========================================================================
    "create_gaussian_kernel",
    "create_dog_filters",
    "create_gabor_filters",
    
    # =========================================================================
    # Functional - Temporal Encoding
    # =========================================================================
    "intensity_to_latency",
    "latency_to_spikes",
    "encode_image_to_spikes",
    
    # =========================================================================
    # Functional - Winner-Take-All
    # =========================================================================
    "wta_global",
    "wta_local",
    
    # =========================================================================
    # Functional - Utilities
    # =========================================================================
    "compute_output_size",
    "count_spikes",
]

