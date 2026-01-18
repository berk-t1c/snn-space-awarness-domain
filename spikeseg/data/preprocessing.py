"""
Preprocessing Module for SpikeSEG.

Complete preprocessing pipelines matching the papers exactly:
    - Kheradpisheh et al. 2018: Image preprocessing with DoG, rank-order coding
    - SpykeTorch (Mozafari et al. 2019): Complete image-to-spike pipeline
    - SpikeSEG (Kirkland et al. 2020): PENT, temporal buffering, LIF buffer
    - IGARSS 2023: Event camera preprocessing for satellite detection

Pipeline Summary (SpykeTorch Tutorial):
    "image >> grayscale >> tensor >> DoG/Gabor filters >> 
     local normalization >> lateral inhibition >> intensity-to-latency"

Paper References:
    [1] Kheradpisheh et al. 2018 - "STDP-based spiking deep CNNs"
    [2] Mozafari et al. 2019 - "SpykeTorch: Efficient Simulation"
    [3] Kirkland et al. 2020 - "SpikeSEG: Spiking Segmentation via STDP"
    [4] Kirkland et al. 2023 - IGARSS Satellite Detection

Example:
    >>> from spikeseg.data.preprocessing import (
    ...     SpykeTorchPreprocessor, SpikeSEGPreprocessor,
    ...     GaborFilterBank, AdaptiveThreshold
    ... )
    >>> 
    >>> # SpykeTorch-style preprocessing
    >>> preprocessor = SpykeTorchPreprocessor(n_timesteps=15)
    >>> spike_wave = preprocessor(image)
    >>>
    >>> # SpikeSEG temporal buffering
    >>> preprocessor = SpikeSEGPreprocessor(buffer_size=20)
    >>> buffered = preprocessor(events)

Author: SpikeSEG Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Lazy imports for circular dependency avoidance
_DoGFilter = None
_lateral_inhibition = None


def _get_dog_filter():
    """Lazy import DoGFilter to avoid circular imports."""
    global _DoGFilter
    if _DoGFilter is None:
        from .events import DoGFilter
        _DoGFilter = DoGFilter
    return _DoGFilter


def _get_lateral_inhibition():
    """Lazy import lateral_inhibition to avoid circular imports."""
    global _lateral_inhibition
    if _lateral_inhibition is None:
        from .events import lateral_inhibition
        _lateral_inhibition = lateral_inhibition
    return _lateral_inhibition


# =============================================================================
# GABOR FILTER BANK (Alternative to DoG)
# =============================================================================


def create_gabor_kernel(
    size: int = 7,
    sigma: float = 1.0,
    theta: float = 0.0,
    lambd: float = 2.0,
    gamma: float = 0.5,
    psi: float = 0.0
) -> torch.Tensor:
    """
    Create a Gabor filter kernel.
    
    Gabor filters are used in early visual cortex modeling and
    are an alternative to DoG for edge detection.
    
    Paper Reference (Kheradpisheh 2016):
        "Bio-inspired unsupervised learning of visual features leads to 
        robust invariant object recognition... using Gabor filters"
    
    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation of Gaussian envelope.
        theta: Orientation in radians.
        lambd: Wavelength of sinusoidal factor.
        gamma: Spatial aspect ratio.
        psi: Phase offset.
    
    Returns:
        Gabor kernel of shape (size, size).
    """
    if size % 2 == 0:
        size += 1
    
    half = size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing='ij'
    )
    
    # Rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    
    # Gabor function
    gaussian = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    sinusoid = torch.cos(2 * math.pi * x_theta / lambd + psi)
    
    gabor = gaussian * sinusoid
    
    # Normalize
    gabor = gabor - gabor.mean()
    gabor = gabor / (gabor.abs().sum() + 1e-8)
    
    return gabor


def create_gabor_filterbank(
    n_orientations: int = 4,
    n_scales: int = 1,
    size: int = 7,
    base_sigma: float = 1.0,
    base_lambda: float = 2.0
) -> torch.Tensor:
    """
    Create a bank of Gabor filters at multiple orientations and scales.
    
    Paper Reference (SpykeTorch):
        "We create a Filter object by providing a list of 4 Gabor filter 
        kernels... for different orientations"
    
    Args:
        n_orientations: Number of orientation angles (typically 4 or 8).
        n_scales: Number of scales.
        size: Kernel size.
        base_sigma: Base sigma for smallest scale.
        base_lambda: Base wavelength.
    
    Returns:
        Filter bank of shape (n_filters, 1, size, size).
    """
    filters = []
    
    for scale in range(n_scales):
        sigma = base_sigma * (2 ** scale)
        lambd = base_lambda * (2 ** scale)
        
        for i in range(n_orientations):
            theta = i * math.pi / n_orientations
            
            kernel = create_gabor_kernel(
                size=size,
                sigma=sigma,
                theta=theta,
                lambd=lambd
            )
            filters.append(kernel)
    
    filterbank = torch.stack(filters).unsqueeze(1)
    return filterbank


class GaborFilterBank(nn.Module):
    """
    Gabor filter bank layer.
    
    Applies oriented Gabor filters to extract edge information
    at multiple orientations, similar to V1 simple cells.
    
    Paper Reference (SpykeTorch Tutorial):
        "SpykeTorch.utils provides Filter class which is suitable for 
        applying different filter kernels on a 4D input tensor. Here, 
        we create a Filter object by providing a list of 4 Gabor filter 
        kernels."
    """
    
    def __init__(
        self,
        n_orientations: int = 4,
        n_scales: int = 1,
        kernel_size: int = 7,
        use_abs: bool = True,
        threshold: float = 0.0
    ):
        """
        Initialize Gabor filter bank.
        
        Args:
            n_orientations: Number of orientations (default 4: 0°, 45°, 90°, 135°).
            n_scales: Number of scales.
            kernel_size: Size of Gabor kernels.
            use_abs: Take absolute value (combines ON/OFF responses).
            threshold: Minimum response threshold.
        """
        super().__init__()
        
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.use_abs = use_abs
        self.threshold = threshold
        
        filterbank = create_gabor_filterbank(
            n_orientations=n_orientations,
            n_scales=n_scales,
            size=kernel_size
        )
        self.register_buffer('filters', filterbank)
        
        self.n_filters = filterbank.shape[0]
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filtering.
        
        Args:
            x: Input image, shape (B, 1, H, W) or (B, H, W).
        
        Returns:
            Gabor responses, shape (B, n_filters, H', W').
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        pad = self.kernel_size // 2
        response = F.conv2d(x, self.filters, padding=pad)
        
        if self.use_abs:
            response = response.abs()
        
        if self.threshold > 0:
            response = torch.where(
                response >= self.threshold,
                response,
                torch.zeros_like(response)
            )
        
        return response


# =============================================================================
# PRE-EMPTIVE NEURON THRESHOLDING (PENT) - SpikeSEG
# =============================================================================


class AdaptiveThreshold(nn.Module):
    """
    Adaptive Neuron Threshold (PENT) from SpikeSEG.
    
    Paper Reference (SpikeSEG, Section IV.B):
        "A progression of the Pre-Emptive Neuron Thresholding (PENT) 
        processes... with the adaptation now being able to affect all 
        encoding Conv layers within the network. The thresholding is 
        based on the homoeostasis mechanism called synaptic scaling, 
        normally taking effect after hours or even days of high neuronal 
        activity, to try and reduce activity."
    
    The threshold adapts based on input activity to maintain
    stable firing rates (homeostasis).
    
    Attributes:
        base_threshold: Initial threshold value.
        alpha: Adaptation rate.
        target_rate: Target firing rate.
    """
    
    def __init__(
        self,
        base_threshold: float = 1.0,
        alpha: float = 0.1,
        target_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0
    ):
        """
        Initialize adaptive threshold.
        
        Args:
            base_threshold: Starting threshold value.
            alpha: Adaptation rate (how fast threshold changes).
            target_rate: Desired fraction of neurons firing.
            min_threshold: Minimum allowed threshold.
            max_threshold: Maximum allowed threshold.
        """
        super().__init__()
        
        self.register_buffer('threshold', torch.tensor(base_threshold))
        self.alpha = alpha
        self.target_rate = target_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def forward(self, potentials: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive thresholding.
        
        Args:
            potentials: Membrane potentials, shape (...).
        
        Returns:
            Tuple of (spikes, updated_threshold).
        """
        # Generate spikes
        spikes = (potentials >= self.threshold).float()
        
        # Calculate current firing rate
        current_rate = spikes.mean()
        
        # Adapt threshold
        if self.training:
            # If firing too much, increase threshold
            # If firing too little, decrease threshold
            error = current_rate - self.target_rate
            new_threshold = self.threshold + self.alpha * error
            new_threshold = torch.clamp(
                new_threshold, 
                self.min_threshold, 
                self.max_threshold
            )
            self.threshold.copy_(new_threshold)
        
        return spikes, self.threshold
    
    def reset(self, value: Optional[float] = None):
        """Reset threshold to initial or specified value."""
        if value is not None:
            self.threshold.fill_(value)


class LayerwiseAdaptiveThreshold(nn.Module):
    """
    Layer-wise adaptive thresholding for multi-layer networks.
    
    Paper Reference (SpikeSEG):
        "The adaptation now being able to affect all encoding Conv 
        layers within the network."
    
    Each layer has its own adaptive threshold that adjusts
    independently based on that layer's activity.
    """
    
    def __init__(
        self,
        n_layers: int,
        base_threshold: float = 1.0,
        alpha: float = 0.1,
        target_rate: float = 0.1
    ):
        """
        Initialize layer-wise thresholds.
        
        Args:
            n_layers: Number of layers to track.
            base_threshold: Initial threshold for all layers.
            alpha: Adaptation rate.
            target_rate: Target firing rate.
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.thresholds = nn.ModuleList([
            AdaptiveThreshold(base_threshold, alpha, target_rate)
            for _ in range(n_layers)
        ])
    
    def forward(
        self, 
        potentials: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply threshold for specific layer."""
        return self.thresholds[layer_idx](potentials)
    
    def get_thresholds(self) -> List[float]:
        """Get current threshold values for all layers."""
        return [t.threshold.item() for t in self.thresholds]
    
    def reset_all(self, value: Optional[float] = None):
        """Reset all thresholds."""
        for t in self.thresholds:
            t.reset(value)


# =============================================================================
# LEAKY INTEGRATE-AND-FIRE BUFFER LAYER - SpikeSEG
# =============================================================================


class LIFBuffer(nn.Module):
    """
    Leaky Integrate-and-Fire Buffer Layer.
    
    Paper Reference (SpikeSEG, Section IV.A):
        "A number of event based cameras do not directly produce spikes, 
        hence a leaky integrate and fire buffer layer... This adaptive 
        thresholding allows the buffer to have a variable amount of 
        events in contrast to maintaining a fixed number of events 
        with variable buffer rate."
    
    Accumulates incoming events and produces spikes when
    membrane potential exceeds threshold.
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        n_channels: int = 2,
        tau: float = 10.0,
        threshold: float = 1.0,
        reset_mode: str = 'subtract'  # 'subtract' or 'zero'
    ):
        """
        Initialize LIF buffer.
        
        Args:
            height: Sensor height.
            width: Sensor width.
            n_channels: Number of polarity channels.
            tau: Membrane time constant (controls leak).
            threshold: Spike threshold.
            reset_mode: 'subtract' (soft reset) or 'zero' (hard reset).
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.tau = tau
        self.threshold = threshold
        self.reset_mode = reset_mode
        
        # Decay factor
        self.register_buffer('decay', torch.tensor(math.exp(-1.0 / tau)))
        
        # Membrane potential state
        self.register_buffer(
            'membrane',
            torch.zeros(1, n_channels, height, width)
        )
    
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Process events through LIF buffer.
        
        Args:
            events: Event frame, shape (B, C, H, W) or (C, H, W).
        
        Returns:
            Spike output, same shape as input.
        """
        if events.dim() == 3:
            events = events.unsqueeze(0)
        
        B = events.shape[0]
        
        # Expand membrane if batch size changed
        if self.membrane.shape[0] != B:
            self.membrane = self.membrane.expand(B, -1, -1, -1).clone()
        
        # Leak
        self.membrane = self.membrane * self.decay
        
        # Integrate
        self.membrane = self.membrane + events
        
        # Fire
        spikes = (self.membrane >= self.threshold).float()
        
        # Reset
        if self.reset_mode == 'subtract':
            self.membrane = self.membrane - spikes * self.threshold
        else:  # zero
            self.membrane = self.membrane * (1 - spikes)
        
        return spikes.squeeze(0) if B == 1 else spikes
    
    def reset_state(self):
        """Reset membrane potential to zero."""
        self.membrane.zero_()


# =============================================================================
# TEMPORAL BUFFERING - SpikeSEG/IGARSS
# =============================================================================


class TemporalBuffer(nn.Module):
    """
    Temporal Buffering for Event Streams.
    
    Paper Reference (SpikeSEG, Section IV.C):
        "The input events are fed into the network via a temporal 
        buffering stage, to allow for a more plausible current 
        computing solution... To internally mimic the continuous 
        data the buffered data is parsed into 20 steps. 10 of these 
        are parsed event streams dividing the temporal data into 
        equal parts, and the other 10 steps ensure the all parsed 
        event streams have time to fully pass through network."
    
    Attributes:
        n_event_steps: Number of parsed event stream steps.
        n_propagation_steps: Extra steps for network propagation.
        total_steps: Total buffer size (event + propagation).
    """
    
    def __init__(
        self,
        n_event_steps: int = 10,
        n_propagation_steps: int = 10,
        height: int = 180,
        width: int = 240,
        n_channels: int = 2
    ):
        """
        Initialize temporal buffer.
        
        Paper Reference:
            "10 of these are parsed event streams... and the other 
            10 steps ensure the all parsed event streams have time 
            to fully pass through network, since the network has 
            9 computational layers"
        
        Args:
            n_event_steps: Steps for event data (default 10).
            n_propagation_steps: Extra steps for propagation (default 10).
            height: Frame height.
            width: Frame width.
            n_channels: Number of polarity channels.
        """
        super().__init__()
        
        self.n_event_steps = n_event_steps
        self.n_propagation_steps = n_propagation_steps
        self.total_steps = n_event_steps + n_propagation_steps
        self.height = height
        self.width = width
        self.n_channels = n_channels
    
    def forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray
    ) -> torch.Tensor:
        """
        Buffer event stream into temporal bins.
        
        Args:
            x, y, p, t: Event arrays.
        
        Returns:
            Buffered tensor, shape (total_steps, n_channels, H, W).
            First n_event_steps contain events, rest are zeros
            for propagation.
        """
        # Initialize buffer
        buffer = torch.zeros(
            self.total_steps, self.n_channels, self.height, self.width
        )
        
        if len(x) == 0:
            return buffer
        
        # Normalize timestamps to [0, n_event_steps)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.n_event_steps - 1e-6)
        else:
            t_norm = np.zeros_like(t)
        
        t_idx = np.clip(t_norm.astype(np.int32), 0, self.n_event_steps - 1)
        
        # Clip spatial coordinates
        x = np.clip(x, 0, self.width - 1).astype(np.int32)
        y = np.clip(y, 0, self.height - 1).astype(np.int32)
        
        # Handle polarity
        if p.min() < 0:
            c_idx = ((p + 1) // 2).astype(np.int32)
        else:
            c_idx = p.astype(np.int32)
        
        # Accumulate events into buffer
        np.add.at(
            buffer.numpy(),
            (t_idx, c_idx, y, x),
            1.0
        )
        
        return buffer
    
    def get_spike_activity_maps(
        self,
        buffer: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Generate spike activity maps from buffer.
        
        Paper Reference (SpikeSEG):
            "For each time step in the encoding processing a spike 
            activity map S_mt is also produced, where m is the feature 
            map and t is the time step."
        
        Args:
            buffer: Temporal buffer, shape (T, C, H, W).
            threshold: Spike threshold.
        
        Returns:
            Binary spike activity maps, shape (T, C, H, W).
        """
        return (buffer >= threshold).float()


# =============================================================================
# COMPLETE PREPROCESSING PIPELINES
# =============================================================================


class SpykeTorchPreprocessor(nn.Module):
    """
    Complete SpykeTorch-style preprocessing pipeline.
    
    Paper Reference (SpykeTorch Tutorial):
        "The whole pipeline of image transformation is as follows:
        image >> grayscale image >> PyTorch's 3D tensor >> 
        4D tensor (time dimension) >> Gabor filters >> 
        lateral inhibition >> intensity-to-latency encoding."
    
    This is the standard preprocessing for static images
    used in Kheradpisheh 2018 and related works.
    """
    
    def __init__(
        self,
        filter_type: str = 'dog',  # 'dog' or 'gabor'
        dog_sizes: Optional[List[int]] = None,
        n_orientations: int = 4,
        kernel_size: int = 7,
        normalization_radius: int = 4,
        n_timesteps: int = 15,
        threshold: float = 0.01,
        use_lateral_inhibition: bool = False,
        use_pointwise_inhibition: bool = True
    ):
        """
        Initialize SpykeTorch preprocessor.
        
        Args:
            filter_type: 'dog' for DoG filters, 'gabor' for Gabor filters.
            dog_sizes: DoG kernel sizes (for filter_type='dog'). Default: [3, 5, 7].
            n_orientations: Gabor orientations (for filter_type='gabor').
            kernel_size: Gabor kernel size.
            normalization_radius: Local normalization radius.
            n_timesteps: Number of temporal bins for spike wave.
            threshold: Minimum filter response threshold.
            use_lateral_inhibition: Apply spatial lateral inhibition.
            use_pointwise_inhibition: Apply feature-wise inhibition.
        """
        super().__init__()
        
        if dog_sizes is None:
            dog_sizes = [3, 5, 7]
        
        self.filter_type = filter_type
        self.n_timesteps = n_timesteps
        self.use_lateral_inhibition = use_lateral_inhibition
        self.use_pointwise_inhibition = use_pointwise_inhibition
        self.normalization_radius = normalization_radius
        
        # Create filter bank
        if filter_type == 'dog':
            DoGFilter = _get_dog_filter()
            self.filters = DoGFilter(sizes=dog_sizes, threshold=threshold)
        else:
            self.filters = GaborFilterBank(
                n_orientations=n_orientations,
                kernel_size=kernel_size,
                use_abs=True,
                threshold=threshold
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image to spike wave.
        
        Args:
            x: Input image, shape (B, 1, H, W), (B, H, W), or (H, W).
               Values should be in [0, 1].
        
        Returns:
            Spike wave tensor, shape (B, T, C, H, W) or (T, C, H, W).
        """
        # Handle input dimensions
        single_image = x.dim() <= 3
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        
        B = x.shape[0]
        
        # Step 1: Apply filters
        response = self.filters(x)  # (B, C, H, W)
        
        # Step 2: Take absolute value (if not already done)
        intensities = response.abs()
        
        # Step 3: Local normalization
        intensities = self._local_normalize(intensities)
        
        # Step 4: Lateral inhibition (optional)
        if self.use_lateral_inhibition:
            intensities = self._lateral_inhibition(intensities)
        
        # Step 5: Pointwise inhibition (optional)
        if self.use_pointwise_inhibition:
            intensities = self._pointwise_inhibition(intensities)
        
        # Step 6: Intensity to latency encoding
        spike_waves = []
        for b in range(B):
            spike_wave = self._intensity_to_latency(intensities[b])
            spike_waves.append(spike_wave)
        
        spike_waves = torch.stack(spike_waves)  # (B, T, C, H, W)
        
        if single_image:
            spike_waves = spike_waves.squeeze(0)
        
        return spike_waves
    
    def _local_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local normalization using regional mean."""
        kernel_size = 2 * self.normalization_radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
        kernel = kernel / (kernel_size ** 2)
        
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        local_mean = F.conv2d(x_flat, kernel, padding=self.normalization_radius)
        local_mean = local_mean.reshape(B, C, H, W)
        
        return x / (local_mean + 1e-12)
    
    def _lateral_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial lateral inhibition."""
        lateral_inhibition = _get_lateral_inhibition()
        return lateral_inhibition(x)
    
    def _pointwise_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pointwise inhibition between features.
        
        Paper Reference (SpykeTorch):
            "At most one neuron is allowed to fire at each position, 
            which is the neuron with the earliest spike time."
        """
        # Keep only max across channels at each position
        max_vals, _ = x.max(dim=1, keepdim=True)
        mask = (x == max_vals).float()
        
        # Break ties by keeping first channel
        cumsum = mask.cumsum(dim=1)
        mask = (cumsum == 1).float() * mask
        
        return x * mask
    
    def _intensity_to_latency(self, intensities: torch.Tensor) -> torch.Tensor:
        """Convert intensities to spike times using rank-order coding."""
        C, H, W = intensities.shape
        
        # Flatten for ranking
        flat = intensities.reshape(-1)
        n_elements = flat.numel()
        
        # Rank (higher intensity = lower latency)
        sorted_idx = torch.argsort(flat, descending=True)
        ranks = torch.zeros_like(sorted_idx)
        ranks[sorted_idx] = torch.arange(n_elements, device=flat.device)
        
        # Convert to timesteps
        latencies = (ranks.float() / n_elements * self.n_timesteps).long()
        latencies = torch.clamp(latencies, 0, self.n_timesteps - 1)
        latencies = latencies.reshape(C, H, W)
        
        # Create spike tensor
        spike_wave = torch.zeros(self.n_timesteps, C, H, W, device=intensities.device)
        for t in range(self.n_timesteps):
            spike_wave[t] = (latencies == t).float()
        
        return spike_wave


class SpikeSEGPreprocessor(nn.Module):
    """
    Complete SpikeSEG preprocessing pipeline for event cameras.
    
    Paper Reference (SpikeSEG/IGARSS 2023):
        Combines temporal buffering, LIF buffer, and adaptive thresholding
        for processing event camera data.
    
    Pipeline:
        1. Temporal buffering (20 steps: 10 event + 10 propagation)
        2. LIF buffer layer (accumulate and spike)
        3. Adaptive thresholding (PENT)
        4. Spike activity map generation
    """
    
    def __init__(
        self,
        height: int = 180,
        width: int = 240,
        n_channels: int = 2,
        n_event_steps: int = 10,
        n_propagation_steps: int = 10,
        use_lif_buffer: bool = True,
        lif_tau: float = 10.0,
        use_adaptive_threshold: bool = True,
        base_threshold: float = 1.0,
        target_rate: float = 0.1
    ):
        """
        Initialize SpikeSEG preprocessor.
        
        Args:
            height: Sensor height.
            width: Sensor width.
            n_channels: Polarity channels (2 for ON/OFF).
            n_event_steps: Event stream temporal bins.
            n_propagation_steps: Extra propagation steps.
            use_lif_buffer: Use LIF buffer layer.
            lif_tau: LIF time constant.
            use_adaptive_threshold: Use PENT adaptive threshold.
            base_threshold: Initial threshold value.
            target_rate: Target firing rate for adaptation.
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.n_channels = n_channels
        
        # Temporal buffer
        self.temporal_buffer = TemporalBuffer(
            n_event_steps=n_event_steps,
            n_propagation_steps=n_propagation_steps,
            height=height,
            width=width,
            n_channels=n_channels
        )
        
        # LIF buffer (optional)
        self.use_lif_buffer = use_lif_buffer
        if use_lif_buffer:
            self.lif_buffer = LIFBuffer(
                height=height,
                width=width,
                n_channels=n_channels,
                tau=lif_tau,
                threshold=base_threshold
            )
        
        # Adaptive threshold (optional)
        self.use_adaptive_threshold = use_adaptive_threshold
        if use_adaptive_threshold:
            self.adaptive_threshold = AdaptiveThreshold(
                base_threshold=base_threshold,
                target_rate=target_rate
            )
    
    def forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess event stream.
        
        Args:
            x, y, p, t: Event arrays (x-coord, y-coord, polarity, timestamp).
        
        Returns:
            Tuple of:
                - Processed spike tensor, shape (T, C, H, W)
                - Spike activity maps, shape (T, C, H, W)
        """
        # Step 1: Temporal buffering
        buffer = self.temporal_buffer(x, y, p, t)
        
        # Step 2: Process through LIF buffer (optional)
        if self.use_lif_buffer:
            self.lif_buffer.reset_state()
            processed = []
            for t_idx in range(buffer.shape[0]):
                frame = self.lif_buffer(buffer[t_idx])
                processed.append(frame)
            processed = torch.stack(processed)
        else:
            processed = buffer
        
        # Step 3: Apply adaptive threshold (optional)
        if self.use_adaptive_threshold:
            spikes, _ = self.adaptive_threshold(processed)
        else:
            spikes = (processed > 0).float()
        
        # Step 4: Generate spike activity maps
        activity_maps = self.temporal_buffer.get_spike_activity_maps(processed)
        
        return spikes, activity_maps
    
    def reset(self):
        """Reset all stateful components."""
        if self.use_lif_buffer:
            self.lif_buffer.reset_state()
        if self.use_adaptive_threshold:
            self.adaptive_threshold.reset()


# =============================================================================
# DATA AUGMENTATION FOR SPIKE TRAINING
# =============================================================================


class SpikeAugmentation(nn.Module):
    """
    Data augmentation for spike tensors.
    
    Applies augmentations that preserve temporal structure
    and spike properties.
    """
    
    def __init__(
        self,
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        flip_temporal: bool = False,
        jitter_time: int = 0,
        dropout_prob: float = 0.0,
        noise_prob: float = 0.0
    ):
        """
        Initialize spike augmentation.
        
        Args:
            flip_horizontal: Random horizontal flip.
            flip_vertical: Random vertical flip.
            flip_temporal: Random temporal flip.
            jitter_time: Max timestep jitter.
            dropout_prob: Probability of dropping spikes.
            noise_prob: Probability of adding noise spikes.
        """
        super().__init__()
        
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.flip_temporal = flip_temporal
        self.jitter_time = jitter_time
        self.dropout_prob = dropout_prob
        self.noise_prob = noise_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to spike tensor.
        
        Args:
            x: Spike tensor, shape (T, C, H, W) or (B, T, C, H, W).
        
        Returns:
            Augmented spike tensor.
        """
        if not self.training:
            return x
        
        has_batch = x.dim() == 5
        if not has_batch:
            x = x.unsqueeze(0)
        
        # Horizontal flip
        if self.flip_horizontal and torch.rand(1) > 0.5:
            x = torch.flip(x, [-1])
        
        # Vertical flip
        if self.flip_vertical and torch.rand(1) > 0.5:
            x = torch.flip(x, [-2])
        
        # Temporal flip
        if self.flip_temporal and torch.rand(1) > 0.5:
            x = torch.flip(x, [1])
        
        # Spike dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(x) > self.dropout_prob
            x = x * mask.float()
        
        # Noise injection
        if self.noise_prob > 0:
            noise = (torch.rand_like(x) < self.noise_prob).float()
            x = torch.clamp(x + noise, 0, 1)
        
        if not has_batch:
            x = x.squeeze(0)
        
        return x


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================


def normalize_events(
    x: np.ndarray,
    y: np.ndarray,
    target_height: int,
    target_width: int,
    original_height: int,
    original_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize event coordinates to target resolution.
    
    Args:
        x, y: Original coordinates.
        target_height, target_width: Target resolution.
        original_height, original_width: Original resolution.
    
    Returns:
        Normalized (x, y) coordinates.
    """
    x_norm = (x / original_width * target_width).astype(np.int32)
    y_norm = (y / original_height * target_height).astype(np.int32)
    
    x_norm = np.clip(x_norm, 0, target_width - 1)
    y_norm = np.clip(y_norm, 0, target_height - 1)
    
    return x_norm, y_norm


def normalize_timestamps(
    t: np.ndarray,
    n_bins: int
) -> np.ndarray:
    """
    Normalize timestamps to discrete bins.
    
    Args:
        t: Timestamps (any unit).
        n_bins: Number of temporal bins.
    
    Returns:
        Bin indices (0 to n_bins-1).
    """
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (n_bins - 1e-6)
    else:
        t_norm = np.zeros_like(t)
    
    return np.clip(t_norm.astype(np.int32), 0, n_bins - 1)


# =============================================================================
# COMPOSABLE TRANSFORMS
# =============================================================================


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """Convert numpy array to torch tensor."""
    
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x).float()


class Normalize:
    """Normalize tensor to [0, 1] or zero-mean unit-variance."""
    
    def __init__(self, mode: str = 'minmax'):
        """
        Args:
            mode: 'minmax' for [0,1], 'standard' for zero-mean unit-var.
        """
        self.mode = mode
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'minmax':
            x_min = x.min()
            x_max = x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return x
        else:  # standard
            mean = x.mean()
            std = x.std()
            if std > 0:
                return (x - mean) / std
            return x - mean


class Resize:
    """Resize spatial dimensions."""
    
    def __init__(self, size: Tuple[int, int], mode: str = 'bilinear'):
        self.size = size
        self.mode = mode
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, self.size, mode=self.mode, align_corners=False)
            return x.squeeze(0).squeeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
            x = F.interpolate(x, self.size, mode=self.mode, align_corners=False)
            return x.squeeze(0)
        elif x.dim() == 4:
            return F.interpolate(x, self.size, mode=self.mode, align_corners=False)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            x = F.interpolate(x, self.size, mode=self.mode, align_corners=False)
            return x.reshape(B, T, C, *self.size)
        else:
            raise ValueError(f"Unsupported tensor dim: {x.dim()}")


class CenterCrop:
    """Center crop spatial dimensions."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        th, tw = self.size
        
        i = (h - th) // 2
        j = (w - tw) // 2
        
        return x[..., i:i+th, j:j+tw]


class RandomCrop:
    """Random crop spatial dimensions."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        th, tw = self.size
        
        if h < th or w < tw:
            # Pad if needed
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h, w = x.shape[-2:]
        
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()
        
        return x[..., i:i+th, j:j+tw]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Gabor filters
    "create_gabor_kernel",
    "create_gabor_filterbank",
    "GaborFilterBank",
    
    # Adaptive thresholding (PENT)
    "AdaptiveThreshold",
    "LayerwiseAdaptiveThreshold",
    
    # LIF buffer
    "LIFBuffer",
    
    # Temporal buffering
    "TemporalBuffer",
    
    # Complete pipelines
    "SpykeTorchPreprocessor",
    "SpikeSEGPreprocessor",
    
    # Augmentation
    "SpikeAugmentation",
    
    # Normalization utilities
    "normalize_events",
    "normalize_timestamps",
    
    # Composable transforms
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    "CenterCrop",
    "RandomCrop",
]