"""
Event Processing and Encoding for SpikeSEG.

This module implements the exact preprocessing pipeline from the papers:
    - Kheradpisheh et al. 2018: DoG filtering, intensity-to-latency coding
    - SpykeTorch (Mozafari et al. 2019): Local normalization, rank-order coding
    - SpikeSEG (Kirkland et al. 2020): Temporal buffering for event cameras
    - IGARSS 2023: Event camera processing for satellite detection

Pipeline Overview:
    1. Input: Image (H, W) or Event stream (x, y, p, t)
    2. DoG Filtering: Extract ON/OFF center responses (4-6 channels)
    3. Local Normalization: Divide by regional mean
    4. Intensity-to-Latency: Higher intensity → earlier spike time
    5. Spike Wave: Discrete timesteps with binary spikes

Paper References:
    Kheradpisheh et al. 2018 (Section 2.1):
        "DoG cells are retinotopically organized in two ON-center and 
        OFF-center maps which are respectively sensitive to positive and 
        negative contrasts. A DoG cell is allowed to fire if its activation 
        is above a certain threshold. The firing time of a DoG cell is 
        inversely proportional to its activation value."
    
    Mozafari et al. 2019 (SpykeTorch):
        "Each image is convolved by six DoG filters, locally normalized, 
        and transformed into spike-wave."
    
    Kirkland et al. 2020 (SpikeSEG):
        "The input events are fed into the network via a temporal buffering 
        stage... the buffered data is parsed into 20 steps."

Example:
    >>> from spikeseg.data.events import (
    ...     DoGFilter, IntensityToLatency, LocalNormalization,
    ...     ImageToSpikeWave, EventStreamProcessor
    ... )
    >>> 
    >>> # For images (Kheradpisheh style)
    >>> transform = ImageToSpikeWave(
    ...     dog_sizes=[3, 5, 7],
    ...     n_timesteps=15,
    ...     threshold=0.01
    ... )
    >>> spike_wave = transform(image)  # (T, C, H, W)
    >>> 
    >>> # For event streams (EBSSA style)
    >>> processor = EventStreamProcessor(n_timesteps=20)
    >>> voxel = processor(events)  # (T, C, H, W)

Author: SpikeSEG Team
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DIFFERENCE OF GAUSSIANS (DoG) FILTERING
# =============================================================================


def create_dog_kernel(
    size: int,
    sigma_center: float,
    sigma_surround: float,
    on_center: bool = True
) -> torch.Tensor:
    """
    Create a Difference of Gaussians (DoG) kernel.
    
    DoG approximates the center-surround receptive fields found in
    retinal ganglion cells and LGN neurons.
    
    Paper Reference (Kheradpisheh 2018):
        "DoG cells are retinotopically organized in two ON-center and 
        OFF-center maps which are respectively sensitive to positive and 
        negative contrasts."
    
    Args:
        size: Kernel size (should be odd).
        sigma_center: Standard deviation of center Gaussian.
        sigma_surround: Standard deviation of surround Gaussian.
        on_center: If True, center is positive (ON-center cell).
                   If False, center is negative (OFF-center cell).
    
    Returns:
        DoG kernel of shape (size, size).
    
    Example:
        >>> kernel = create_dog_kernel(7, sigma_center=1.0, sigma_surround=2.0)
    """
    if size % 2 == 0:
        size += 1
    
    # Create coordinate grid
    half = size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing='ij'
    )
    
    # Squared distance from center
    d2 = x**2 + y**2
    
    # Center and surround Gaussians
    center = torch.exp(-d2 / (2 * sigma_center**2))
    surround = torch.exp(-d2 / (2 * sigma_surround**2))
    
    # Normalize each Gaussian to sum to 1
    center = center / center.sum()
    surround = surround / surround.sum()
    
    # Difference of Gaussians
    if on_center:
        dog = center - surround  # ON-center: positive center, negative surround
    else:
        dog = surround - center  # OFF-center: negative center, positive surround
    
    return dog


def create_dog_filterbank(
    sizes: Optional[List[int]] = None,
    sigma_ratio: float = 1.6
) -> torch.Tensor:
    """
    Create a bank of DoG filters at multiple scales.
    
    Paper Reference (Mozafari et al. 2019):
        "Each image is convolved by six DoG filters (on- and off-center 
        in three different scales)"
    
    The standard configuration uses 3 scales with ON and OFF variants,
    resulting in 6 filters total.
    
    Args:
        sizes: List of kernel sizes for different scales.
        sigma_ratio: Ratio between surround and center sigma.
                    Default 1.6 approximates Laplacian of Gaussian.
    
    Returns:
        Filter bank of shape (n_filters, 1, max_size, max_size).
        Smaller filters are zero-padded to max_size.
    
    Example:
        >>> filters = create_dog_filterbank([3, 5, 7])
        >>> filters.shape
        torch.Size([6, 1, 7, 7])
    """
    if sizes is None:
        sizes = [3, 5, 7]
    
    filters = []
    max_size = max(sizes)
    
    for size in sizes:
        # Sigma values based on kernel size
        sigma_center = size / 4.0
        sigma_surround = sigma_center * sigma_ratio
        
        # ON-center and OFF-center filters
        on_kernel = create_dog_kernel(size, sigma_center, sigma_surround, on_center=True)
        off_kernel = create_dog_kernel(size, sigma_center, sigma_surround, on_center=False)
        
        # Pad smaller kernels to max_size
        if size < max_size:
            pad = (max_size - size) // 2
            on_kernel = F.pad(on_kernel, (pad, pad, pad, pad), value=0)
            off_kernel = F.pad(off_kernel, (pad, pad, pad, pad), value=0)
        
        filters.append(on_kernel)
        filters.append(off_kernel)
    
    # Stack into (n_filters, 1, H, W) for conv2d
    filterbank = torch.stack(filters).unsqueeze(1)
    
    return filterbank


class DoGFilter(nn.Module):
    """
    Difference of Gaussians filtering layer.
    
    Applies DoG filters to extract ON-center and OFF-center responses,
    mimicking retinal ganglion cells.
    
    Paper Reference (Kheradpisheh 2018, Section 2.1):
        "The first layer applies ON- and OFF-center DoG filters of size wD
        on the input image and encode the image contrasts in the timing 
        of the output spikes."
    
    Attributes:
        filters: DoG filter bank (n_filters, 1, kH, kW).
        n_filters: Number of filters (typically 4 or 6).
        threshold: Minimum response threshold.
    
    Example:
        >>> dog = DoGFilter(sizes=[3, 5, 7])
        >>> image = torch.randn(1, 1, 28, 28)
        >>> response = dog(image)  # (1, 6, H, W)
    """
    
    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        sigma_ratio: float = 1.6,
        threshold: float = 0.0,
        padding: str = 'same'
    ):
        """
        Initialize DoG filter layer.
        
        Args:
            sizes: Kernel sizes for different scales. Default: [3, 5, 7].
            sigma_ratio: Ratio σ_surround / σ_center.
            threshold: Minimum activation threshold.
            padding: Padding mode ('same', 'valid', or int).
        """
        super().__init__()
        
        if sizes is None:
            sizes = [3, 5, 7]
        self.sizes = sizes
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold
        self.padding = padding
        
        # Create filter bank
        filterbank = create_dog_filterbank(sizes, sigma_ratio)
        self.register_buffer('filters', filterbank)
        
        self.n_filters = filterbank.shape[0]
        self.kernel_size = filterbank.shape[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DoG filtering.
        
        Args:
            x: Input image, shape (B, 1, H, W) or (B, H, W).
        
        Returns:
            DoG responses, shape (B, n_filters, H', W').
        """
        # Handle input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        
        # Calculate padding
        if self.padding == 'same':
            pad = self.kernel_size // 2
        elif self.padding == 'valid':
            pad = 0
        else:
            pad = int(self.padding)
        
        # Apply filters
        response = F.conv2d(x, self.filters, padding=pad)
        
        # Apply threshold (only keep strong responses)
        if self.threshold > 0:
            response = torch.where(
                response.abs() >= self.threshold,
                response,
                torch.zeros_like(response)
            )
        
        return response
    
    def extra_repr(self) -> str:
        return f'sizes={self.sizes}, n_filters={self.n_filters}, threshold={self.threshold}'


# =============================================================================
# LOCAL NORMALIZATION
# =============================================================================


def local_normalization(
    x: torch.Tensor,
    radius: int = 4,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Apply local normalization using regional mean.
    
    Paper Reference (SpykeTorch):
        "sf.local_normalization which uses regional mean for normalizing 
        intensity values."
    
    For each pixel, divides by the mean of surrounding pixels within
    the given radius. This provides local contrast normalization.
    
    Args:
        x: Input tensor, shape (..., H, W).
        radius: Normalization radius. Region size = 2*radius + 1.
        eps: Small constant to avoid division by zero.
    
    Returns:
        Locally normalized tensor with same shape.
    
    Example:
        >>> x = torch.randn(1, 4, 28, 28)
        >>> x_norm = local_normalization(x, radius=4)
    """
    # Create averaging kernel
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    kernel = kernel / (kernel_size ** 2)
    
    # Handle different input dimensions
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    
    B, C, H, W = x.shape
    
    # Compute local mean for each channel
    x_flat = x.reshape(B * C, 1, H, W)
    local_mean = F.conv2d(x_flat, kernel, padding=radius)
    local_mean = local_mean.reshape(B, C, H, W)
    
    # Normalize
    x_norm = x / (local_mean + eps)
    
    # Restore original shape
    if len(original_shape) == 2:
        x_norm = x_norm.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        x_norm = x_norm.squeeze(0)
    
    return x_norm


class LocalNormalization(nn.Module):
    """
    Local normalization layer.
    
    Applies local contrast normalization by dividing each pixel
    by the mean of its surrounding region.
    """
    
    def __init__(self, radius: int = 4, eps: float = 1e-12):
        super().__init__()
        self.radius = radius
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return local_normalization(x, self.radius, self.eps)


# =============================================================================
# INTENSITY TO LATENCY ENCODING
# =============================================================================


def intensity_to_latency(
    intensities: torch.Tensor,
    n_timesteps: int = 15,
    to_spike: bool = True
) -> torch.Tensor:
    """
    Convert intensities to spike latencies (rank-order coding).
    
    Paper Reference (Kheradpisheh 2018, Section 2.1):
        "The firing time of a DoG cell is inversely proportional to its 
        activation value. In other words, the order of the spikes depends 
        on the order of the contrasts."
    
    Higher intensity → earlier spike (lower latency).
    This implements rank-order coding where spike order encodes information.
    
    Args:
        intensities: Input intensities, any shape (...).
        n_timesteps: Number of temporal bins.
        to_spike: If True, return binary spike tensor (T, ...).
                  If False, return latency values (...).
    
    Returns:
        If to_spike: Spike tensor of shape (n_timesteps, *intensities.shape).
        If not to_spike: Latency values with same shape as input.
    
    Example:
        >>> intensities = torch.tensor([0.9, 0.5, 0.1])
        >>> spikes = intensity_to_latency(intensities, n_timesteps=10, to_spike=True)
        >>> spikes.shape
        torch.Size([10, 3])
    """
    # Flatten for sorting
    original_shape = intensities.shape
    intensities_flat = intensities.reshape(-1)
    n_elements = intensities_flat.numel()
    
    # Rank intensities (higher intensity = earlier rank = lower latency)
    # argsort gives indices that would sort in ascending order
    # We want descending, so highest intensity gets rank 0
    sorted_indices = torch.argsort(intensities_flat, descending=True)
    ranks = torch.zeros_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(n_elements, device=intensities.device)
    
    # Convert ranks to timesteps
    # Divide into bins: rank 0 -> timestep 0, rank n-1 -> timestep n_timesteps-1
    latencies = (ranks.float() / n_elements * n_timesteps).long()
    latencies = torch.clamp(latencies, 0, n_timesteps - 1)
    
    # Reshape back
    latencies = latencies.reshape(original_shape)
    
    if not to_spike:
        return latencies
    
    # Convert to spike tensor
    spike_wave = torch.zeros(n_timesteps, *original_shape, 
                             device=intensities.device, dtype=torch.float32)
    
    # Create indices for scatter
    for t in range(n_timesteps):
        spike_wave[t] = (latencies == t).float()
    
    return spike_wave


def intensity_to_latency_linear(
    intensities: torch.Tensor,
    n_timesteps: int = 15,
    i_min: float = 0.0,
    i_max: float = 1.0,
    to_spike: bool = True
) -> torch.Tensor:
    """
    Convert intensities to latencies using linear mapping.
    
    Alternative to rank-order: directly maps intensity to latency.
    latency = (1 - normalized_intensity) * (n_timesteps - 1)
    
    Args:
        intensities: Input intensities.
        n_timesteps: Number of temporal bins.
        i_min: Minimum intensity (maps to last timestep).
        i_max: Maximum intensity (maps to first timestep).
        to_spike: Return spike tensor if True.
    
    Returns:
        Latencies or spike tensor.
    """
    # Normalize to [0, 1]
    intensities_norm = (intensities - i_min) / (i_max - i_min + 1e-8)
    intensities_norm = torch.clamp(intensities_norm, 0, 1)
    
    # High intensity -> low latency (early spike)
    latencies = ((1 - intensities_norm) * (n_timesteps - 1)).long()
    latencies = torch.clamp(latencies, 0, n_timesteps - 1)
    
    if not to_spike:
        return latencies
    
    # Convert to spike tensor
    original_shape = intensities.shape
    spike_wave = torch.zeros(n_timesteps, *original_shape,
                             device=intensities.device, dtype=torch.float32)
    
    for t in range(n_timesteps):
        spike_wave[t] = (latencies == t).float()
    
    return spike_wave


class IntensityToLatency(nn.Module):
    """
    Intensity to latency encoding layer.
    
    Converts intensity values to spike times using rank-order coding.
    
    Paper Reference (SpykeTorch):
        "SpykeTorch.utils.Intensity2Latency object generates the spike-wave 
        tensor based on intensities."
    """
    
    def __init__(
        self,
        n_timesteps: int = 15,
        to_spike: bool = True,
        use_linear: bool = False
    ):
        """
        Initialize encoder.
        
        Args:
            n_timesteps: Number of temporal bins.
            to_spike: Output spike tensor if True, latencies if False.
            use_linear: Use linear mapping instead of rank-order.
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.to_spike = to_spike
        self.use_linear = use_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_linear:
            return intensity_to_latency_linear(x, self.n_timesteps, to_spike=self.to_spike)
        else:
            return intensity_to_latency(x, self.n_timesteps, to_spike=self.to_spike)


# =============================================================================
# LATERAL INHIBITION
# =============================================================================


def lateral_inhibition(
    intensities: torch.Tensor,
    inhibition_radius: int = 3,
    inhibition_decay: float = 0.8
) -> torch.Tensor:
    """
    Apply lateral inhibition between spatial locations.
    
    For each location, inhibits surrounding pixels with lower intensity.
    This sharpens responses and creates winner-take-all competition.
    
    Paper Reference (SpykeTorch):
        "sf.intensity_lateral_inhibition... decreases the intensity of 
        surrounding cells that have lower intensities by a specific factor."
    
    Args:
        intensities: Input intensities, shape (..., H, W).
        inhibition_radius: Radius of inhibition kernel.
        inhibition_decay: Decay factor for inhibition (0-1).
    
    Returns:
        Inhibited intensities with same shape.
    """
    # Create inhibition kernel (stronger at center, weaker at edges)
    size = 2 * inhibition_radius + 1
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij'
    )
    center = inhibition_radius
    distance = torch.sqrt((x - center)**2 + (y - center)**2)
    
    # Inhibition strength decreases with distance
    kernel = inhibition_decay ** distance
    kernel[center, center] = 0  # Don't inhibit self
    kernel = kernel.to(intensities.device)
    
    # Get input shape
    original_shape = intensities.shape
    if intensities.dim() == 2:
        intensities = intensities.unsqueeze(0).unsqueeze(0)
    elif intensities.dim() == 3:
        intensities = intensities.unsqueeze(0)
    
    B, C, H, W = intensities.shape
    
    # Process each location
    result = intensities.clone()
    padded = F.pad(intensities, [inhibition_radius] * 4, mode='constant', value=0)
    
    for dy in range(-inhibition_radius, inhibition_radius + 1):
        for dx in range(-inhibition_radius, inhibition_radius + 1):
            if dy == 0 and dx == 0:
                continue
            
            # Get shifted version
            shifted = padded[:, :, 
                            inhibition_radius + dy:inhibition_radius + dy + H,
                            inhibition_radius + dx:inhibition_radius + dx + W]
            
            # Apply inhibition where shifted value is higher
            inhibit_factor = kernel[inhibition_radius + dy, inhibition_radius + dx]
            mask = (shifted > result).float()
            result = result * (1 - mask * inhibit_factor)
    
    # Restore shape
    if len(original_shape) == 2:
        result = result.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        result = result.squeeze(0)
    
    return result


def pointwise_inhibition(
    potentials: torch.Tensor,
    spikes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pointwise inhibition between feature maps.
    
    At each spatial location, only the feature with highest potential
    (or earliest spike) is allowed to fire.
    
    Paper Reference (SpykeTorch):
        "sf.pointwise_inhibition... at most one neuron is allowed to fire 
        at each position, which is the neuron with the earliest spike time."
    
    Args:
        potentials: Feature map potentials, shape (B, C, H, W).
        spikes: Optional spike times, shape (T, C, H, W).
    
    Returns:
        Inhibited potentials with same shape.
    """
    # Keep only max over channels at each position
    max_vals, _ = potentials.max(dim=1, keepdim=True)
    mask = (potentials == max_vals).float()
    
    # Handle ties by keeping first channel
    # Create cumulative mask to break ties
    cumsum = mask.cumsum(dim=1)
    mask = (cumsum == 1).float() * mask
    
    return potentials * mask


# =============================================================================
# COMPLETE IMAGE TO SPIKE WAVE TRANSFORM
# =============================================================================


class ImageToSpikeWave(nn.Module):
    """
    Complete pipeline: Image → DoG → Normalize → Spike Wave.
    
    This implements the full preprocessing from Kheradpisheh 2018
    and SpykeTorch papers.
    
    Pipeline:
        1. Apply DoG filters (ON/OFF at multiple scales)
        2. Take absolute value (separate pos/neg later)
        3. Apply local normalization
        4. Apply lateral inhibition (optional)
        5. Convert intensity to latency (rank-order coding)
    
    Paper Reference (SpykeTorch Listing 6):
        "InputTransform object converts the input image into a tensor, 
        adds an extra dimension for time, applies provided filters, 
        applies local normalization, and generates spike-wave tensor."
    
    Example:
        >>> transform = ImageToSpikeWave(
        ...     dog_sizes=[3, 5, 7],
        ...     n_timesteps=15,
        ...     normalization_radius=4
        ... )
        >>> spike_wave = transform(image)  # (T, 6, H, W)
    """
    
    def __init__(
        self,
        dog_sizes: Optional[List[int]] = None,
        dog_sigma_ratio: float = 1.6,
        normalization_radius: int = 4,
        n_timesteps: int = 15,
        threshold: float = 0.01,
        use_lateral_inhibition: bool = False,
        inhibition_radius: int = 3
    ):
        """
        Initialize transform.
        
        Args:
            dog_sizes: DoG kernel sizes for different scales. Default: [3, 5, 7].
            dog_sigma_ratio: Ratio σ_surround / σ_center.
            normalization_radius: Radius for local normalization.
            n_timesteps: Number of temporal bins for spike wave.
            threshold: Minimum DoG response threshold.
            use_lateral_inhibition: Apply lateral inhibition.
            inhibition_radius: Radius for lateral inhibition.
        """
        super().__init__()
        
        if dog_sizes is None:
            dog_sizes = [3, 5, 7]
        
        self.dog_filter = DoGFilter(
            sizes=dog_sizes,
            sigma_ratio=dog_sigma_ratio,
            threshold=threshold
        )
        self.local_norm = LocalNormalization(radius=normalization_radius)
        self.encoder = IntensityToLatency(n_timesteps=n_timesteps, to_spike=True)
        
        self.use_lateral_inhibition = use_lateral_inhibition
        self.inhibition_radius = inhibition_radius
        self.n_timesteps = n_timesteps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform image to spike wave.
        
        Args:
            x: Input image, shape (B, 1, H, W) or (H, W).
        
        Returns:
            Spike wave tensor, shape (T, n_filters, H', W') or (B, T, C, H', W').
        """
        # Handle different input shapes
        single_image = x.dim() <= 3
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        
        B = x.shape[0]
        
        # Apply DoG filtering
        dog_response = self.dog_filter(x)  # (B, C, H, W)
        
        # Take absolute value (we care about magnitude, sign encoded in filter type)
        intensities = dog_response.abs()
        
        # Local normalization
        intensities = self.local_norm(intensities)
        
        # Lateral inhibition (optional)
        if self.use_lateral_inhibition:
            intensities = lateral_inhibition(intensities, self.inhibition_radius)
        
        # Convert to spike wave for each sample
        spike_waves = []
        for b in range(B):
            spike_wave = self.encoder(intensities[b])  # (T, C, H, W)
            spike_waves.append(spike_wave)
        
        spike_waves = torch.stack(spike_waves)  # (B, T, C, H, W)
        
        if single_image:
            spike_waves = spike_waves.squeeze(0)  # (T, C, H, W)
        
        return spike_waves


# =============================================================================
# EVENT STREAM PROCESSING (FOR EVENT CAMERAS)
# =============================================================================


class EventBuffer:
    """
    Buffer for accumulating events over time windows.
    
    Paper Reference (SpikeSEG):
        "The input events are fed into the network via a temporal 
        buffering stage, to allow for a more plausible current computing 
        solution... the buffered data is parsed into 20 steps."
    
    Attributes:
        events_x: List of x coordinate arrays per time bin.
        events_y: List of y coordinate arrays per time bin.
        events_p: List of polarity arrays per time bin.
        events_t: List of timestamp arrays per time bin.
        n_bins: Number of temporal bins.
    """
    
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.events_x: List[List] = []
        self.events_y: List[List] = []
        self.events_p: List[List] = []
        self.events_t: List[List] = []
        self.clear()
    
    def clear(self):
        """Clear all buffered events."""
        self.events_x = [[] for _ in range(self.n_bins)]
        self.events_y = [[] for _ in range(self.n_bins)]
        self.events_p = [[] for _ in range(self.n_bins)]
        self.events_t = [[] for _ in range(self.n_bins)]
    
    def add_events(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray,
        t_start: float,
        t_end: float
    ):
        """Add events to appropriate time bins."""
        # Normalize time to [0, n_bins)
        # Guard against division by zero when all events have same timestamp
        duration = t_end - t_start
        if duration > 0:
            t_norm = (t - t_start) / duration * self.n_bins
        else:
            # All events at same time - put in middle bin
            t_norm = np.full_like(t, self.n_bins // 2, dtype=np.float64)
        bin_idx = np.clip(t_norm.astype(int), 0, self.n_bins - 1)
        
        for b in range(self.n_bins):
            mask = bin_idx == b
            self.events_x[b].extend(x[mask])
            self.events_y[b].extend(y[mask])
            self.events_p[b].extend(p[mask])
            self.events_t[b].extend(t[mask])


class EventStreamProcessor(nn.Module):
    """
    Process raw event streams into voxel grids for SNN.
    
    This handles event camera data (ATIS, DAVIS, etc.) by:
        1. Temporal binning into discrete timesteps
        2. Spatial accumulation into voxel grid
        3. Optional polarity separation (ON/OFF channels)
        4. Optional noise filtering
    
    Paper Reference (IGARSS 2023):
        "Temporal buffering into 20 timesteps... 10 parsed event streams 
        per buffer"
    
    Example:
        >>> processor = EventStreamProcessor(
        ...     n_timesteps=20,
        ...     height=180,
        ...     width=240,
        ...     polarity_channels=True
        ... )
        >>> voxel = processor(x, y, p, t)  # (T, 2, H, W)
    """
    
    def __init__(
        self,
        n_timesteps: int = 20,
        height: int = 180,
        width: int = 240,
        polarity_channels: bool = True,
        normalize: bool = True,
        hot_pixel_threshold: Optional[int] = None
    ):
        """
        Initialize processor.
        
        Args:
            n_timesteps: Number of temporal bins.
            height: Sensor height in pixels.
            width: Sensor width in pixels.
            polarity_channels: Separate channels for ON/OFF events.
            normalize: Normalize voxel values to [0, 1].
            hot_pixel_threshold: Max events per pixel (noise filter).
        """
        super().__init__()
        
        self.n_timesteps = n_timesteps
        self.height = height
        self.width = width
        self.polarity_channels = polarity_channels
        self.normalize = normalize
        self.hot_pixel_threshold = hot_pixel_threshold
        
        self.n_channels = 2 if polarity_channels else 1
    
    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        p: Union[np.ndarray, torch.Tensor],
        t: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert event stream to voxel grid.
        
        Args:
            x: X coordinates, shape (N,).
            y: Y coordinates, shape (N,).
            p: Polarities, shape (N,).
            t: Timestamps, shape (N,).
        
        Returns:
            Voxel grid, shape (T, C, H, W).
        """
        # Convert to numpy for processing
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            y = y.numpy()
            p = p.numpy()
            t = t.numpy()
        
        # Initialize voxel grid
        voxel = np.zeros(
            (self.n_timesteps, self.n_channels, self.height, self.width),
            dtype=np.float32
        )
        
        if len(x) == 0:
            return torch.from_numpy(voxel)
        
        # Hot pixel filtering
        if self.hot_pixel_threshold is not None:
            x, y, p, t = self._filter_hot_pixels(x, y, p, t)
            if len(x) == 0:
                return torch.from_numpy(voxel)
        
        # Normalize timestamps to [0, n_timesteps-1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            # Map to [0, n_timesteps-1] range, then clip to handle edge cases
            t_norm = (t - t_min) / (t_max - t_min) * (self.n_timesteps - 1)
        else:
            t_norm = np.zeros_like(t, dtype=np.float32)

        t_idx = np.clip(np.round(t_norm).astype(np.int32), 0, self.n_timesteps - 1)
        
        # Clip spatial coordinates
        x = np.clip(x, 0, self.width - 1).astype(np.int32)
        y = np.clip(y, 0, self.height - 1).astype(np.int32)
        
        # Determine polarity channel
        if self.polarity_channels:
            # Channel 0: ON (positive), Channel 1: OFF (negative)
            if p.min() < 0:
                # +1 (ON) -> Channel 0, -1 (OFF) -> Channel 1
                c_idx = ((1 - p) // 2).astype(np.int32)
            else:
                c_idx = p.astype(np.int32)
        else:
            c_idx = np.zeros_like(x)
        
        # Accumulate events
        np.add.at(voxel, (t_idx, c_idx, y, x), 1.0)
        
        # Normalize
        if self.normalize:
            for ti in range(self.n_timesteps):
                max_val = voxel[ti].max()
                if max_val > 0:
                    voxel[ti] /= max_val
        
        return torch.from_numpy(voxel)
    
    def _filter_hot_pixels(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter out hot pixels (pixels with excessive events)."""
        # Clip coordinates first to avoid index errors
        x_clipped = np.clip(x, 0, self.width - 1).astype(np.int32)
        y_clipped = np.clip(y, 0, self.height - 1).astype(np.int32)
        
        # Count events per pixel
        pixel_counts = np.zeros((self.height, self.width), dtype=np.int32)
        np.add.at(pixel_counts, (y_clipped, x_clipped), 1)
        
        # Identify hot pixels
        hot_mask = pixel_counts > self.hot_pixel_threshold
        
        # Filter events from hot pixels
        event_mask = ~hot_mask[y_clipped, x_clipped]
        
        return x[event_mask], y[event_mask], p[event_mask], t[event_mask]


# =============================================================================
# NOISE FILTERING (EBSSA/SATELLITE SPECIFIC)
# =============================================================================


def filter_refractory_events(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    refractory_period: float = 1000.0  # microseconds
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter events that occur too close in time at the same pixel.
    
    Biological neurons have a refractory period during which they
    cannot fire again. This filter removes rapid-fire noise.
    
    Args:
        x, y, p, t: Event arrays.
        refractory_period: Minimum time between events at same pixel (μs).
    
    Returns:
        Filtered event arrays.
    """
    n_events = len(x)
    if n_events == 0:
        return x, y, p, t
    
    # Track last event time per pixel
    last_time = {}
    keep_mask = np.ones(n_events, dtype=bool)
    
    for i in range(n_events):
        pixel = (x[i], y[i])
        if pixel in last_time:
            if t[i] - last_time[pixel] < refractory_period:
                keep_mask[i] = False
                continue
        last_time[pixel] = t[i]
    
    return x[keep_mask], y[keep_mask], p[keep_mask], t[keep_mask]


def filter_isolated_events(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    spatial_radius: int = 1,
    temporal_window: float = 10000.0,  # microseconds
    min_neighbors: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter spatially and temporally isolated events (noise).
    
    Real events typically occur in clusters (moving edges).
    Isolated events are likely noise.
    
    Args:
        x, y, p, t: Event arrays.
        spatial_radius: Radius for neighbor search.
        temporal_window: Time window for neighbor search (μs).
        min_neighbors: Minimum neighbors to keep event.
    
    Returns:
        Filtered event arrays.
    """
    n_events = len(x)
    if n_events < 2:
        return x, y, p, t
    
    keep_mask = np.zeros(n_events, dtype=bool)
    
    # Sort by time
    sort_idx = np.argsort(t)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    t_sorted = t[sort_idx]
    
    # For each event, count neighbors
    for i in range(n_events):
        # Find temporal neighbors
        t_start = t_sorted[i] - temporal_window
        t_end = t_sorted[i] + temporal_window
        
        # Binary search for time bounds
        j_start = np.searchsorted(t_sorted, t_start)
        j_end = np.searchsorted(t_sorted, t_end)
        
        # Count spatial neighbors within time window
        neighbors = 0
        for j in range(j_start, j_end):
            if j == i:
                continue
            dx = abs(x_sorted[i] - x_sorted[j])
            dy = abs(y_sorted[i] - y_sorted[j])
            if dx <= spatial_radius and dy <= spatial_radius:
                neighbors += 1
                if neighbors >= min_neighbors:
                    break
        
        keep_mask[sort_idx[i]] = neighbors >= min_neighbors
    
    return x[keep_mask], y[keep_mask], p[keep_mask], t[keep_mask]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # DoG filtering
    "create_dog_kernel",
    "create_dog_filterbank",
    "DoGFilter",
    
    # Normalization
    "local_normalization",
    "LocalNormalization",
    
    # Intensity to latency
    "intensity_to_latency",
    "intensity_to_latency_linear",
    "IntensityToLatency",
    
    # Lateral inhibition
    "lateral_inhibition",
    "pointwise_inhibition",
    
    # Complete transforms
    "ImageToSpikeWave",
    
    # Event processing
    "EventBuffer",
    "EventStreamProcessor",
    
    # Noise filtering
    "filter_refractory_events",
    "filter_isolated_events",
]