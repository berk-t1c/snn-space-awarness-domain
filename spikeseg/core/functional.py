"""
Functional Operations for Spiking Neural Networks.

This module contains stateless functions that implement core SNN operations.
These are the mathematical building blocks used by the neuron and layer classes.

Follows PyTorch's pattern: torch.nn.Module uses torch.nn.functional

Contents:
    - Spike generation functions
    - Membrane potential dynamics
    - Filter creation (DoG, Gabor)
    - Temporal encoding (rank-order coding)
    - Winner-Take-All operations

Paper References:
    - Kheradpisheh et al. 2018: DoG filters, rank-order coding, membrane dynamics
    - Kirkland et al. 2020 (SpikeSEG): Temporal encoding for event data
    - Kirkland et al. 2023 (IGARSS): Modified leak dynamics

Example:
    >>> import torch
    >>> from spikeseg.core.functional import spike_fn, lif_step, create_dog_filters
    >>> 
    >>> # Generate spikes from membrane potential
    >>> membrane = torch.tensor([0.5, 1.2, 0.8])
    >>> spikes = spike_fn(membrane, threshold=1.0)
    >>> spikes
    tensor([0., 1., 0.])
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate that input is a torch.Tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")


def _validate_same_shape(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                         name1: str, name2: str) -> None:
    """Validate that two tensors have the same shape."""
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Shape mismatch: {name1} {tensor1.shape} != {name2} {tensor2.shape}"
        )


def _validate_same_device(tensor1: torch.Tensor, tensor2: torch.Tensor,
                          name1: str, name2: str) -> None:
    """Validate that two tensors are on the same device."""
    if tensor1.device != tensor2.device:
        raise ValueError(
            f"Device mismatch: {name1} on {tensor1.device}, {name2} on {tensor2.device}"
        )


def _validate_positive_float(value: float, name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_float(value: float, name: str) -> None:
    """Validate that a value is a non-negative number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_odd_int(value: int, name: str) -> None:
    """Validate that a value is an odd positive integer."""
    _validate_positive_int(value, name)
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd for symmetric kernel, got {value}")


def _validate_4d_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate that a tensor is 4-dimensional."""
    _validate_tensor(tensor, name)
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D (batch, channels, height, width), "
            f"got {tensor.dim()}D with shape {tensor.shape}"
        )


def _validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within a range."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")


# =============================================================================
# SPIKE GENERATION FUNCTIONS
# =============================================================================


def spike_fn(membrane: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Generate binary spikes from membrane potential.
    
    This is a Heaviside step function:
        spike = 1 if membrane >= threshold else 0
    
    Note: This function is non-differentiable. For gradient-based learning,
    you would need a surrogate gradient. However, STDP doesn't require gradients.
    
    Args:
        membrane: Membrane potential tensor, any shape.
        threshold: Firing threshold (must be positive).
    
    Returns:
        Binary spike tensor (same shape as membrane).
        Values are 0.0 or 1.0 as float for compatibility with subsequent ops.
    
    Raises:
        TypeError: If membrane is not a tensor or threshold is not a number.
        ValueError: If threshold is not positive.
    
    Example:
        >>> membrane = torch.tensor([5.0, 10.0, 15.0])
        >>> spikes = spike_fn(membrane, threshold=10.0)
        >>> spikes
        tensor([0., 1., 1.])
    """
    _validate_tensor(membrane, "membrane")
    _validate_positive_float(threshold, "threshold")
    
    return (membrane >= threshold).float()


def soft_spike_fn(
    membrane: torch.Tensor, 
    threshold: float, 
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Generate soft (probabilistic) spikes using sigmoid.
    
    This is a differentiable approximation of the spike function.
    Useful for hybrid approaches or surrogate gradient methods.
    
    As temperature → 0, this approaches the hard spike function.
    As temperature → ∞, this approaches 0.5 everywhere.
    
    Args:
        membrane: Membrane potential tensor.
        threshold: Firing threshold (must be positive).
        temperature: Controls sharpness of transition (must be positive).
                    Lower = sharper.
    
    Returns:
        Spike probabilities in [0, 1].
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If threshold or temperature is not positive.
    
    Example:
        >>> membrane = torch.tensor([9.0, 10.0, 11.0])
        >>> soft_spikes = soft_spike_fn(membrane, threshold=10.0, temperature=0.5)
        >>> soft_spikes  # Values close to [0, 0.5, 1]
    """
    _validate_tensor(membrane, "membrane")
    _validate_positive_float(threshold, "threshold")
    _validate_positive_float(temperature, "temperature")
    
    return torch.sigmoid((membrane - threshold) / temperature)


# =============================================================================
# MEMBRANE POTENTIAL DYNAMICS
# =============================================================================


def if_step(
    membrane: torch.Tensor,
    input_current: torch.Tensor,
    threshold: float,
    reset_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timestep of Integrate-and-Fire neuron dynamics.
    
    Dynamics:
        V(t) = V(t-1) + I(t)
        if V(t) >= threshold: spike, V(t) = reset_value
    
    Args:
        membrane: Current membrane potential. Shape: any.
        input_current: Input current (weighted sum of input spikes). Same shape.
        threshold: Firing threshold (must be positive).
        reset_value: Value to reset membrane to after spike. Default 0.
    
    Returns:
        Tuple of (spikes, new_membrane):
            - spikes: Binary tensor indicating which neurons fired.
            - new_membrane: Updated membrane potential after reset.
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If shapes don't match or threshold is not positive.
    
    Example:
        >>> membrane = torch.tensor([0.5, 0.8])
        >>> current = torch.tensor([0.3, 0.5])
        >>> spikes, new_mem = if_step(membrane, current, threshold=1.0)
        >>> spikes   # tensor([0., 1.]) - second neuron fires (0.8 + 0.5 >= 1.0)
    """
    _validate_tensor(membrane, "membrane")
    _validate_tensor(input_current, "input_current")
    _validate_same_shape(membrane, input_current, "membrane", "input_current")
    _validate_same_device(membrane, input_current, "membrane", "input_current")
    _validate_positive_float(threshold, "threshold")
    
    if not isinstance(reset_value, (int, float)):
        raise TypeError(f"reset_value must be a number, got {type(reset_value).__name__}")
    
    # Integrate
    membrane = membrane + input_current
    
    # Generate spikes
    spikes = spike_fn(membrane, threshold)
    
    # Reset fired neurons
    membrane = membrane * (1.0 - spikes) + reset_value * spikes
    
    return spikes, membrane


def lif_step_subtractive(
    membrane: torch.Tensor,
    input_current: torch.Tensor,
    threshold: float,
    leak: float,
    reset_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timestep of Leaky Integrate-and-Fire with subtractive leak.
    
    Dynamics (from IGARSS 2023):
        V(t) = V(t-1) + I(t) - λ
        V(t) = max(V(t), 0)  # Clamp to non-negative
        if V(t) >= threshold: spike, V(t) = reset_value
    
    This is the leak formulation used in the IGARSS 2023 paper:
        "setting the leak, λ, to 90% and 10% of the neuron threshold"
    
    Args:
        membrane: Current membrane potential.
        input_current: Input current.
        threshold: Firing threshold (must be positive).
        leak: Constant leak amount (λ). Must be non-negative.
             Typically leak_factor * threshold.
        reset_value: Value to reset membrane to after spike.
    
    Returns:
        Tuple of (spikes, new_membrane).
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If shapes don't match, threshold not positive, or leak negative.
    
    Example:
        >>> # IGARSS Layer 1: threshold=10, leak=0.9*10=9
        >>> membrane = torch.tensor([5.0])
        >>> current = torch.tensor([8.0])
        >>> spikes, new_mem = lif_step_subtractive(membrane, current, 10.0, 9.0)
        >>> # membrane = 5 + 8 - 9 = 4, no spike
    """
    _validate_tensor(membrane, "membrane")
    _validate_tensor(input_current, "input_current")
    _validate_same_shape(membrane, input_current, "membrane", "input_current")
    _validate_same_device(membrane, input_current, "membrane", "input_current")
    _validate_positive_float(threshold, "threshold")
    _validate_non_negative_float(leak, "leak")
    
    if not isinstance(reset_value, (int, float)):
        raise TypeError(f"reset_value must be a number, got {type(reset_value).__name__}")
    
    # Integrate and leak
    membrane = membrane + input_current - leak
    
    # Clamp to non-negative (biological constraint)
    membrane = torch.clamp(membrane, min=0.0)
    
    # Generate spikes
    spikes = spike_fn(membrane, threshold)
    
    # Reset fired neurons
    membrane = membrane * (1.0 - spikes) + reset_value * spikes
    
    return spikes, membrane


def lif_step_multiplicative(
    membrane: torch.Tensor,
    input_current: torch.Tensor,
    threshold: float,
    beta: float,
    reset_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timestep of Leaky Integrate-and-Fire with multiplicative leak.
    
    Dynamics (from Kheradpisheh 2018, snnTorch):
        V(t) = β * V(t-1) + I(t)
        if V(t) >= threshold: spike, V(t) = reset_value
    
    This creates exponential decay of membrane potential.
    β = 1.0 means no decay (IF neuron).
    β = 0.0 means complete decay (memoryless).
    
    Args:
        membrane: Current membrane potential.
        input_current: Input current.
        threshold: Firing threshold (must be positive).
        beta: Decay factor (must be in [0, 1]). Typically 1 - leak_factor.
        reset_value: Value to reset membrane to after spike.
    
    Returns:
        Tuple of (spikes, new_membrane).
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If shapes don't match, threshold not positive, or beta not in [0,1].
    
    Example:
        >>> # beta=0.9 means 10% decay per timestep
        >>> membrane = torch.tensor([10.0])
        >>> current = torch.tensor([0.0])
        >>> spikes, new_mem = lif_step_multiplicative(membrane, current, 15.0, 0.9)
        >>> new_mem  # tensor([9.0]) - decayed by 10%
    """
    _validate_tensor(membrane, "membrane")
    _validate_tensor(input_current, "input_current")
    _validate_same_shape(membrane, input_current, "membrane", "input_current")
    _validate_same_device(membrane, input_current, "membrane", "input_current")
    _validate_positive_float(threshold, "threshold")
    _validate_range(beta, 0.0, 1.0, "beta")
    
    if not isinstance(reset_value, (int, float)):
        raise TypeError(f"reset_value must be a number, got {type(reset_value).__name__}")
    
    # Decay and integrate
    membrane = beta * membrane + input_current
    
    # Clamp to non-negative
    membrane = torch.clamp(membrane, min=0.0)
    
    # Generate spikes
    spikes = spike_fn(membrane, threshold)
    
    # Reset fired neurons
    membrane = membrane * (1.0 - spikes) + reset_value * spikes
    
    return spikes, membrane


def lif_step(
    membrane: torch.Tensor,
    input_current: torch.Tensor,
    threshold: float,
    leak_factor: float,
    leak_mode: Literal["subtractive", "multiplicative"] = "subtractive",
    reset_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timestep of Leaky Integrate-and-Fire neuron (unified interface).
    
    Supports both leak formulations:
        - "subtractive": V = V + I - λ, where λ = leak_factor * threshold
        - "multiplicative": V = β * V + I, where β = 1 - leak_factor
    
    Args:
        membrane: Current membrane potential.
        input_current: Input current.
        threshold: Firing threshold (must be positive).
        leak_factor: Leak amount as fraction (must be in [0, 1]).
        leak_mode: "subtractive" or "multiplicative".
        reset_value: Value to reset membrane to after spike.
    
    Returns:
        Tuple of (spikes, new_membrane).
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If values are invalid or leak_mode is unknown.
    
    Example:
        >>> membrane = torch.zeros(1, 4, 32, 32)
        >>> current = torch.randn(1, 4, 32, 32)
        >>> spikes, mem = lif_step(membrane, current, threshold=10.0, leak_factor=0.9)
    """
    _validate_range(leak_factor, 0.0, 1.0, "leak_factor")
    
    if leak_mode == "subtractive":
        leak = leak_factor * threshold
        return lif_step_subtractive(membrane, input_current, threshold, leak, reset_value)
    elif leak_mode == "multiplicative":
        beta = 1.0 - leak_factor
        return lif_step_multiplicative(membrane, input_current, threshold, beta, reset_value)
    else:
        raise ValueError(
            f"leak_mode must be 'subtractive' or 'multiplicative', got '{leak_mode}'"
        )


# =============================================================================
# FILTER CREATION (DoG, Gabor)
# =============================================================================


def create_gaussian_kernel(
    size: int,
    sigma: float,
    normalize: bool = True
) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (must be positive, typically odd for symmetry).
        sigma: Standard deviation of Gaussian (must be positive).
        normalize: If True, kernel sums to 1.
    
    Returns:
        Gaussian kernel. Shape: (size, size)
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If size or sigma is not positive.
    
    Example:
        >>> kernel = create_gaussian_kernel(7, sigma=1.0)
        >>> kernel.shape
        torch.Size([7, 7])
        >>> kernel.sum()  # Approximately 1.0
    """
    _validate_positive_int(size, "size")
    _validate_positive_float(sigma, "sigma")
    
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a boolean, got {type(normalize).__name__}")
    
    # Create coordinate grids
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Compute Gaussian
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    if normalize:
        kernel = kernel / kernel.sum()
    
    return kernel


def create_dog_filters(
    size: int = 7,
    sigma_center: float = 1.0,
    sigma_surround: float = 2.0
) -> torch.Tensor:
    """
    Create Difference of Gaussians (DoG) filters for ON/OFF channels.
    
    DoG approximates the center-surround receptive fields found in
    biological retinal ganglion cells and LGN neurons.
    
    ON-center: Responds to bright center, dark surround (center - surround)
    OFF-center: Responds to dark center, bright surround (surround - center)
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "The first layer performs DoG filtering... with σ1 = 1 and σ2 = 2"
    
    Args:
        size: Kernel size (must be positive, typically 7).
        sigma_center: Std of center Gaussian (must be positive, typically 1.0).
        sigma_surround: Std of surround Gaussian (must be positive, typically 2.0).
    
    Returns:
        DoG filters. Shape: (2, 1, size, size)
        - Channel 0: ON-center (center - surround)
        - Channel 1: OFF-center (surround - center)
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If size or sigma values are not positive.
    
    Example:
        >>> dog = create_dog_filters(size=7, sigma_center=1.0, sigma_surround=2.0)
        >>> dog.shape
        torch.Size([2, 1, 7, 7])
    """
    _validate_positive_int(size, "size")
    _validate_positive_float(sigma_center, "sigma_center")
    _validate_positive_float(sigma_surround, "sigma_surround")
    
    # Create center and surround Gaussians
    center = create_gaussian_kernel(size, sigma_center, normalize=True)
    surround = create_gaussian_kernel(size, sigma_surround, normalize=True)
    
    # ON-center: bright center excites, dark surround inhibits
    on_center = center - surround
    
    # OFF-center: dark center excites, bright surround inhibits
    off_center = surround - center
    
    # Stack into filter tensor: (2, 1, size, size)
    filters = torch.stack([on_center, off_center], dim=0).unsqueeze(1)
    
    return filters


def create_gabor_filters(
    size: int = 5,
    sigma: float = 1.0,
    frequency: float = 0.5,
    n_orientations: int = 4
) -> torch.Tensor:
    """
    Create Gabor filters at multiple orientations.
    
    Gabor filters detect edges at specific orientations.
    Used in Conv1 when fixed (non-learned) filters are desired.
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "The 4 feature maps of C1 converged to 4 different Gabor-like edge
         detectors with approximately the same orientations of π/4, π/2, 3π/4, π"
    
    Args:
        size: Kernel size (must be positive, typically odd).
        sigma: Gaussian envelope standard deviation (must be positive).
        frequency: Spatial frequency of sinusoid (must be positive).
        n_orientations: Number of orientations (must be positive, typically 4).
    
    Returns:
        Gabor filters. Shape: (n_orientations, 1, size, size)
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If values are not positive.
    
    Example:
        >>> gabors = create_gabor_filters(size=5, n_orientations=4)
        >>> gabors.shape
        torch.Size([4, 1, 5, 5])
    """
    _validate_positive_int(size, "size")
    _validate_positive_float(sigma, "sigma")
    _validate_positive_float(frequency, "frequency")
    _validate_positive_int(n_orientations, "n_orientations")
    
    filters = []
    
    # Create coordinate grids
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Create Gabor at each orientation
    for i in range(n_orientations):
        # Orientation angle (π/4, π/2, 3π/4, π for n=4)
        theta = math.pi * (i + 1) / n_orientations
        
        # Rotate coordinates
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        
        # Gabor = Gaussian * Sinusoid
        gaussian = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * frequency * x_theta)
        
        gabor = gaussian * sinusoid
        
        # Normalize to zero mean (important for edge detection)
        gabor = gabor - gabor.mean()
        
        filters.append(gabor)
    
    # Stack: (n_orientations, 1, size, size)
    return torch.stack(filters, dim=0).unsqueeze(1)


# =============================================================================
# TEMPORAL ENCODING (Rank-Order Coding)
# =============================================================================


def intensity_to_latency(
    intensity: torch.Tensor,
    max_time: float = 1.0,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Convert intensity values to spike latencies (rank-order coding).
    
    Higher intensity → earlier spike (shorter latency).
    This is the fundamental encoding used in STDP-based SNNs.
    
    Formula: latency = max_time / (intensity + epsilon)
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "The most activated neurons fire first"
        "The firing times are set to the inverse of their value"
    
    Args:
        intensity: Input intensity values (typically 0 to 1).
        max_time: Maximum latency for zero intensity (must be positive).
        epsilon: Small value to prevent division by zero (must be positive).
    
    Returns:
        Spike latencies. Same shape as input.
        Higher intensity → smaller latency.
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If max_time or epsilon is not positive.
    
    Example:
        >>> intensity = torch.tensor([0.1, 0.5, 1.0])
        >>> latency = intensity_to_latency(intensity, max_time=1.0)
        >>> latency  # Higher intensity = shorter latency
    """
    _validate_tensor(intensity, "intensity")
    _validate_positive_float(max_time, "max_time")
    _validate_positive_float(epsilon, "epsilon")
    
    # Clamp intensity to valid range
    intensity = torch.clamp(intensity, min=0.0)
    
    # Convert to latency: higher intensity = shorter latency
    latency = max_time / (intensity + epsilon)
    
    # Clamp latency to [0, max_time]
    latency = torch.clamp(latency, min=0.0, max=max_time)
    
    return latency


def latency_to_spikes(
    latency: torch.Tensor,
    n_timesteps: int,
    max_time: float = 1.0
) -> torch.Tensor:
    """
    Convert spike latencies to a spike train tensor.
    
    Creates a tensor where each timestep is 1 if that neuron
    fires at that time, 0 otherwise.
    
    Args:
        latency: Spike latencies. Shape: (batch, channels, height, width)
        n_timesteps: Number of discrete timesteps (must be positive).
        max_time: Time corresponding to max latency (must be positive).
    
    Returns:
        Spike train. Shape: (batch, n_timesteps, channels, height, width)
        Binary values indicating spike times.
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If latency is not 4D, or n_timesteps/max_time not positive.
    
    Example:
        >>> latency = torch.tensor([[[[0.1, 0.5], [0.3, 0.9]]]])  # (1,1,2,2)
        >>> spikes = latency_to_spikes(latency, n_timesteps=10, max_time=1.0)
        >>> spikes.shape
        torch.Size([1, 10, 1, 2, 2])
    """
    _validate_4d_tensor(latency, "latency")
    _validate_positive_int(n_timesteps, "n_timesteps")
    _validate_positive_float(max_time, "max_time")
    
    batch, channels, height, width = latency.shape
    
    # Convert latency to timestep index
    timestep_idx = (latency / max_time * n_timesteps).long()
    timestep_idx = torch.clamp(timestep_idx, 0, n_timesteps - 1)
    
    # Create spike train tensor
    spikes = torch.zeros(batch, n_timesteps, channels, height, width, 
                         device=latency.device, dtype=latency.dtype)
    
    # Create indices for scatter
    batch_idx = torch.arange(batch, device=latency.device).view(-1, 1, 1, 1)
    batch_idx = batch_idx.expand(-1, channels, height, width)
    
    channel_idx = torch.arange(channels, device=latency.device).view(1, -1, 1, 1)
    channel_idx = channel_idx.expand(batch, -1, height, width)
    
    height_idx = torch.arange(height, device=latency.device).view(1, 1, -1, 1)
    height_idx = height_idx.expand(batch, channels, -1, width)
    
    width_idx = torch.arange(width, device=latency.device).view(1, 1, 1, -1)
    width_idx = width_idx.expand(batch, channels, height, -1)
    
    # Place spikes at their firing times
    spikes[batch_idx, timestep_idx, channel_idx, height_idx, width_idx] = 1.0
    
    return spikes


def encode_image_to_spikes(
    image: torch.Tensor,
    n_timesteps: int,
    dog_size: int = 7,
    sigma_center: float = 1.0,
    sigma_surround: float = 2.0
) -> torch.Tensor:
    """
    Encode a grayscale image into spike trains using DoG + rank-order coding.
    
    Pipeline:
        1. Apply DoG filters → ON/OFF channels
        2. Rectify (keep positive values)
        3. Convert intensity to latency
        4. Convert latency to spike train
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "Images are first converted to on-center and off-center filtered images
         using a DoG filter... Then the first layer performs rank-order coding"
    
    Args:
        image: Grayscale image. Shape: (batch, 1, height, width) or (batch, height, width)
        n_timesteps: Number of simulation timesteps (must be positive).
        dog_size: Size of DoG filter kernel (must be positive).
        sigma_center: DoG center sigma (must be positive).
        sigma_surround: DoG surround sigma (must be positive).
    
    Returns:
        Spike train. Shape: (batch, n_timesteps, 2, height, width)
        2 channels for ON and OFF.
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If image has wrong dimensions or other params invalid.
    
    Example:
        >>> image = torch.rand(1, 1, 64, 64)  # Grayscale image
        >>> spikes = encode_image_to_spikes(image, n_timesteps=20)
        >>> spikes.shape
        torch.Size([1, 20, 2, 64, 64])
    """
    _validate_tensor(image, "image")
    _validate_positive_int(n_timesteps, "n_timesteps")
    _validate_positive_int(dog_size, "dog_size")
    _validate_positive_float(sigma_center, "sigma_center")
    _validate_positive_float(sigma_surround, "sigma_surround")
    
    # Ensure 4D input
    if image.dim() == 3:
        image = image.unsqueeze(1)  # Add channel dim
    elif image.dim() != 4:
        raise ValueError(
            f"image must be 3D (batch, H, W) or 4D (batch, 1, H, W), "
            f"got {image.dim()}D"
        )
    
    if image.shape[1] != 1:
        raise ValueError(
            f"image must have 1 channel (grayscale), got {image.shape[1]}"
        )
    
    # Create DoG filters
    dog = create_dog_filters(dog_size, sigma_center, sigma_surround)
    dog = dog.to(image.device)
    
    # Apply DoG (ON/OFF channels)
    padding = dog_size // 2
    on_off = F.conv2d(image, dog, padding=padding)  # (batch, 2, H, W)
    
    # Rectify: keep positive responses
    on_off = F.relu(on_off)
    
    # Normalize to [0, 1]
    on_off_max = on_off.amax(dim=(2, 3), keepdim=True) + 1e-6
    on_off_norm = on_off / on_off_max
    
    # Convert to latency (higher intensity = earlier spike)
    latency = intensity_to_latency(on_off_norm, max_time=1.0)
    
    # Convert to spike train
    spikes = latency_to_spikes(latency, n_timesteps, max_time=1.0)
    
    return spikes


# =============================================================================
# WINNER-TAKE-ALL (WTA) FUNCTIONS
# =============================================================================


def wta_global(
    spikes: torch.Tensor,
    membrane: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Global Winner-Take-All: Only the first neuron to spike wins.
    
    Within each feature map, when a neuron fires, all other neurons
    in the same feature map are inhibited (membrane reset to 0).
    
    This enforces that each feature is detected at most once per sample.
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "Each neuron can fire at most once per stimulus"
        "global intra-map competition"
    
    Args:
        spikes: Current timestep spikes. Shape: (batch, channels, height, width)
        membrane: Current membrane potential. Same shape.
    
    Returns:
        Tuple of (filtered_spikes, new_membrane):
            - filtered_spikes: Spikes after WTA (at most one per feature map)
            - new_membrane: Reset membrane for inhibited neurons
    
    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes don't match or tensors are not 4D.
    
    Example:
        >>> spikes = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)
        >>> membrane = torch.tensor([[[[0.5, 0.8], [0.5, 0.2]]]])
        >>> new_spikes, new_mem = wta_global(spikes, membrane)
        >>> new_spikes.sum()  # Only 1 spike survives
    """
    _validate_4d_tensor(spikes, "spikes")
    _validate_4d_tensor(membrane, "membrane")
    _validate_same_shape(spikes, membrane, "spikes", "membrane")
    _validate_same_device(spikes, membrane, "spikes", "membrane")
    
    batch, channels, height, width = spikes.shape
    
    # For each batch and channel, find if any neuron spiked
    spikes_flat = spikes.view(batch, channels, -1)  # (B, C, H*W)
    
    # Find first spike location in each feature map
    has_spike = spikes_flat.sum(dim=2) > 0  # (B, C)
    
    # Get index of first spike (if any)
    first_spike_idx = spikes_flat.argmax(dim=2)  # (B, C)
    
    # Create new spike tensor with only first spike
    new_spikes_flat = torch.zeros_like(spikes_flat)
    
    # Set the first spike location to 1 where there was a spike
    batch_idx = torch.arange(batch, device=spikes.device).unsqueeze(1).expand(-1, channels)
    channel_idx = torch.arange(channels, device=spikes.device).unsqueeze(0).expand(batch, -1)
    
    # Only set spike if there was at least one spike in this feature map
    new_spikes_flat[batch_idx[has_spike], channel_idx[has_spike], first_spike_idx[has_spike]] = 1.0
    
    # Reshape back
    new_spikes = new_spikes_flat.view_as(spikes)
    
    # Reset membrane for all neurons in feature maps that had a spike
    inhibition_mask = has_spike.unsqueeze(-1).unsqueeze(-1).expand_as(membrane)
    new_membrane = torch.where(inhibition_mask, torch.zeros_like(membrane), membrane)
    
    return new_spikes, new_membrane


def wta_local(
    spikes: torch.Tensor,
    membrane: torch.Tensor,
    radius: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Local Winner-Take-All: Spatial competition within a neighborhood.
    
    When a neuron fires, neurons within a local radius are inhibited
    across ALL feature maps at that spatial location.
    
    This enforces that nearby locations don't fire for the same feature.
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "local inter-map competition: neurons in a local neighborhood
         across all feature maps compete"
    
    Args:
        spikes: Current timestep spikes. Shape: (batch, channels, height, width)
        membrane: Current membrane potential. Same shape.
        radius: Inhibition radius (must be non-negative). 
               Neighborhood size = 2*radius + 1.
    
    Returns:
        Tuple of (filtered_spikes, new_membrane).
    
    Raises:
        TypeError: If inputs have wrong types.
        ValueError: If shapes don't match, tensors not 4D, or radius negative.
    
    Example:
        >>> spikes = torch.zeros(1, 4, 8, 8)
        >>> spikes[0, 0, 4, 4] = 1  # One spike at center
        >>> membrane = torch.ones(1, 4, 8, 8) * 0.5
        >>> new_spikes, new_mem = wta_local(spikes, membrane, radius=2)
    """
    _validate_4d_tensor(spikes, "spikes")
    _validate_4d_tensor(membrane, "membrane")
    _validate_same_shape(spikes, membrane, "spikes", "membrane")
    _validate_same_device(spikes, membrane, "spikes", "membrane")
    _validate_non_negative_int(radius, "radius")
    
    batch, channels, height, width = spikes.shape
    
    # Find locations where any channel spiked
    any_spike = spikes.sum(dim=1, keepdim=True) > 0  # (B, 1, H, W)
    
    # Dilate to create inhibition mask
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=spikes.device)
    
    # Apply dilation (max pooling equivalent)
    padded = F.pad(any_spike.float(), (radius, radius, radius, radius))
    inhibition_zone = F.conv2d(padded, kernel) > 0  # (B, 1, H, W)
    
    # Expand to all channels
    inhibition_zone = inhibition_zone.expand(-1, channels, -1, -1)  # (B, C, H, W)
    
    # Reset membrane in inhibition zones
    new_membrane = torch.where(inhibition_zone, torch.zeros_like(membrane), membrane)
    
    # Spikes themselves are preserved (they already fired)
    return spikes, new_membrane


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """
    Compute output size of a convolution/pooling operation.
    
    Formula: output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
    
    Args:
        input_size: Input spatial dimension (must be positive).
        kernel_size: Kernel size (must be positive).
        stride: Stride (must be positive).
        padding: Padding (must be non-negative).
        dilation: Dilation (must be positive).
    
    Returns:
        Output spatial dimension.
    
    Raises:
        TypeError: If inputs are not integers.
        ValueError: If values are invalid.
    
    Example:
        >>> compute_output_size(64, kernel_size=5, stride=1, padding=2)
        64  # Same padding
        >>> compute_output_size(64, kernel_size=2, stride=2, padding=0)
        32  # Halved by pooling
    """
    _validate_positive_int(input_size, "input_size")
    _validate_positive_int(kernel_size, "kernel_size")
    _validate_positive_int(stride, "stride")
    _validate_non_negative_int(padding, "padding")
    _validate_positive_int(dilation, "dilation")
    
    output = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    if output <= 0:
        raise ValueError(
            f"Computed output size is {output} (non-positive). "
            f"Check input_size={input_size}, kernel_size={kernel_size}, "
            f"stride={stride}, padding={padding}, dilation={dilation}"
        )
    
    return output


def count_spikes(spikes: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Count total number of spikes.
    
    Args:
        spikes: Spike tensor (binary values).
        dim: Dimension(s) to sum over. None = all dimensions.
    
    Returns:
        Spike count.
    
    Raises:
        TypeError: If spikes is not a tensor.
    
    Example:
        >>> spikes = torch.tensor([[[1, 0, 1], [0, 1, 0]]], dtype=torch.float32)
        >>> count_spikes(spikes)  # Total
        tensor(3.)
        >>> count_spikes(spikes, dim=2)  # Per channel
        tensor([[2., 1.]])
    """
    _validate_tensor(spikes, "spikes")
    
    if dim is None:
        return spikes.sum()
    return spikes.sum(dim=dim)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Spike functions
    "spike_fn",
    "soft_spike_fn",
    # Membrane dynamics
    "if_step",
    "lif_step",
    "lif_step_subtractive",
    "lif_step_multiplicative",
    # Filters
    "create_gaussian_kernel",
    "create_dog_filters",
    "create_gabor_filters",
    # Temporal encoding
    "intensity_to_latency",
    "latency_to_spikes",
    "encode_image_to_spikes",
    # WTA
    "wta_global",
    "wta_local",
    # Utilities
    "compute_output_size",
    "count_spikes",
]
