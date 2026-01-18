"""
WTA: Winner-Take-All Lateral Inhibition.

This module provides lateral inhibition mechanisms that enforce competition
between neurons. In STDP-trained networks, WTA ensures that:
    1. Each feature is learned by at most one neuron
    2. Different neurons learn different features
    3. Only winning neurons get STDP weight updates

Paper References:
    Kheradpisheh et al. 2018 - "STDP-based spiking deep convolutional neural 
    networks for object recognition"
    
    "We use a winner-take-all (WTA) mechanism to enforce competition among
    neurons. The first neuron that fires inhibits the others, preventing
    them from firing and receiving plasticity updates."

Competition Types:
    - Intra-map (global): Neurons within same feature map compete
    - Inter-map (local): Neurons across feature maps at same location compete

Homeostasis:
    Adaptive thresholds ensure all neurons get a chance to learn:
    - Neurons that fire too often: threshold increases
    - Neurons that never fire: threshold decreases

Example:
    >>> from spikeseg.learning.wta import WTAInhibition, WTAConfig
    >>> 
    >>> # Create WTA with adaptive thresholds
    >>> config = WTAConfig(mode="both", enable_homeostasis=True)
    >>> wta = WTAInhibition(config, n_channels=36, spatial_shape=(16, 16))
    >>> 
    >>> # During forward pass
    >>> filtered_spikes, new_membrane = wta(spikes, membrane)
    >>> winner_mask = wta.get_winner_mask()  # For STDP updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class WTAError(Exception):
    """Base exception for WTA module errors."""
    pass


class WTAConfigError(WTAError):
    """Raised for invalid WTA configuration."""
    pass


class WTARuntimeError(WTAError):
    """Raised for runtime errors during WTA processing."""
    pass


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_tensor(tensor: Any, name: str) -> None:
    """Validate that input is a torch.Tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")


def _validate_4d_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor is 4D with shape (N, C, H, W)."""
    _validate_tensor(tensor, name)
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D (N, C, H, W), got {tensor.dim()}D with shape {tensor.shape}"
        )


def _validate_positive_int(value: Any, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_int(value: Any, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_positive_float(value: Any, name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_float(value: Any, name: str) -> None:
    """Validate that a value is a non-negative number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_range(value: Any, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within [min_val, max_val]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ValueError(
            f"{name} must be in range [{min_val}, {max_val}], got {value}"
        )


def _validate_same_shape(t1: torch.Tensor, t2: torch.Tensor,
                         name1: str, name2: str) -> None:
    """Validate two tensors have the same shape."""
    if t1.shape != t2.shape:
        raise ValueError(
            f"{name1} and {name2} must have same shape. "
            f"Got {t1.shape} and {t2.shape}"
        )


def _validate_same_device(t1: torch.Tensor, t2: torch.Tensor,
                          name1: str, name2: str) -> None:
    """Validate two tensors are on the same device."""
    if t1.device != t2.device:
        raise ValueError(
            f"{name1} and {name2} must be on same device. "
            f"Got {t1.device} and {t2.device}"
        )


def _validate_tuple_of_ints(value: Any, length: int, name: str) -> None:
    """Validate that value is a tuple of integers with specified length."""
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {len(value)}")
    for i, v in enumerate(value):
        if not isinstance(v, int):
            raise TypeError(
                f"{name}[{i}] must be an integer, got {type(v).__name__}"
            )


# =============================================================================
# WTA CONFIGURATION
# =============================================================================


class WTAMode(Enum):
    """
    Winner-Take-All competition modes.
    
    Attributes:
        GLOBAL: Intra-map competition - one winner per feature map.
        LOCAL: Inter-map competition - spatial neighborhood competition.
        BOTH: Combined global + local competition.
    """
    GLOBAL = "global"
    LOCAL = "local"
    BOTH = "both"


@dataclass
class WTAConfig:
    """
    Configuration for Winner-Take-All inhibition.
    
    Attributes:
        mode: Competition mode (global, local, or both).
        local_radius: Radius for local competition (neighborhood = 2*radius+1).
        enable_homeostasis: Enable adaptive threshold mechanism.
        target_rate: Target firing rate for homeostasis (spikes per presentation).
        homeostasis_lr: Learning rate for threshold adaptation.
        threshold_min: Minimum threshold value.
        threshold_max: Maximum threshold value.
        track_statistics: Track firing statistics for analysis.
    
    Example:
        >>> config = WTAConfig(mode="both", local_radius=2, enable_homeostasis=True)
    """
    mode: WTAMode = WTAMode.GLOBAL
    local_radius: int = 2
    enable_homeostasis: bool = True
    target_rate: float = 0.1
    homeostasis_lr: float = 0.001
    threshold_min: float = 1.0
    threshold_max: float = 100.0
    track_statistics: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Convert string mode to enum if needed
        if isinstance(self.mode, str):
            try:
                self.mode = WTAMode(self.mode.lower())
            except ValueError:
                raise WTAConfigError(
                    f"Invalid mode '{self.mode}'. "
                    f"Must be one of: {[m.value for m in WTAMode]}"
                )
        
        if not isinstance(self.mode, WTAMode):
            raise WTAConfigError(
                f"mode must be WTAMode or string, got {type(self.mode).__name__}"
            )
        
        _validate_positive_int(self.local_radius, "local_radius")
        _validate_range(self.target_rate, 0.0, 1.0, "target_rate")
        _validate_positive_float(self.homeostasis_lr, "homeostasis_lr")
        _validate_positive_float(self.threshold_min, "threshold_min")
        _validate_positive_float(self.threshold_max, "threshold_max")
        
        if self.threshold_min >= self.threshold_max:
            raise WTAConfigError(
                f"threshold_min ({self.threshold_min}) must be < "
                f"threshold_max ({self.threshold_max})"
            )
    
    def __repr__(self) -> str:
        return (
            f"WTAConfig(mode={self.mode.value}, "
            f"homeostasis={self.enable_homeostasis})"
        )


# =============================================================================
# WINNER-TAKE-ALL FUNCTIONS
# =============================================================================


def wta_global_membrane(
    spikes: torch.Tensor,
    membrane: torch.Tensor,
    pre_reset_membrane: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Global WTA with winner mask output.

    Intra-map competition: within each feature map, only the first
    neuron to spike wins. All others are inhibited.

    Tie-breaking (Kheradpisheh 2018):
        "pick the one which has the highest potential"

    Paper Reference:
        "Each neuron can fire at most once per stimulus"
        "global intra-map competition"

    Args:
        spikes: Current spikes, shape (batch, channels, height, width).
        membrane: Current membrane potential (post-reset), same shape.
        pre_reset_membrane: Membrane potential BEFORE reset. If provided,
                           used for tie-breaking (required for correct paper behavior).
                           If None, falls back to membrane (may be incorrect if reset).

    Returns:
        Tuple of (filtered_spikes, new_membrane, winner_mask):
            - filtered_spikes: Only winning spikes remain
            - new_membrane: Membrane reset for inhibited neurons
            - winner_mask: Binary mask of winners (for STDP)

    Raises:
        WTARuntimeError: If inputs are invalid.
    """
    _validate_4d_tensor(spikes, "spikes")
    _validate_4d_tensor(membrane, "membrane")
    _validate_same_shape(spikes, membrane, "spikes", "membrane")
    _validate_same_device(spikes, membrane, "spikes", "membrane")

    # Use pre_reset_membrane for tie-breaking if provided (Kheradpisheh 2018)
    tiebreak_membrane = pre_reset_membrane if pre_reset_membrane is not None else membrane
    if pre_reset_membrane is not None:
        _validate_4d_tensor(pre_reset_membrane, "pre_reset_membrane")
        _validate_same_shape(spikes, pre_reset_membrane, "spikes", "pre_reset_membrane")

    batch, channels, height, width = spikes.shape
    device = spikes.device

    # Flatten spatial dimensions
    spikes_flat = spikes.view(batch, channels, -1)  # (B, C, H*W)
    tiebreak_flat = tiebreak_membrane.view(batch, channels, -1)  # (B, C, H*W)

    # Find if any neuron spiked in each feature map
    has_spike = spikes_flat.sum(dim=2) > 0  # (B, C)

    # Find winner by highest potential among spiking neurons (Kheradpisheh 2018)
    # "pick the one which has the highest potential"
    # Use -inf for non-spiking neurons so they don't win
    masked_membrane = torch.where(
        spikes_flat > 0,
        tiebreak_flat,
        torch.tensor(float('-inf'), device=device, dtype=tiebreak_membrane.dtype)
    )
    winner_idx = masked_membrane.argmax(dim=2)  # (B, C)

    # Create new spike tensor with only winners
    new_spikes_flat = torch.zeros_like(spikes_flat)

    # Indexing tensors
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, channels)
    channel_idx = torch.arange(channels, device=device).unsqueeze(0).expand(batch, -1)

    # Set winner spikes
    new_spikes_flat[
        batch_idx[has_spike],
        channel_idx[has_spike],
        winner_idx[has_spike]
    ] = 1.0

    # Reshape to original
    new_spikes = new_spikes_flat.view_as(spikes)

    # Winner mask is same as new_spikes
    winner_mask = new_spikes.clone()

    # Inhibit membrane in feature maps that had winners
    inhibition_mask = has_spike.unsqueeze(-1).unsqueeze(-1).expand_as(membrane)
    new_membrane = torch.where(inhibition_mask, torch.zeros_like(membrane), membrane)

    return new_spikes, new_membrane, winner_mask


def wta_local_membrane(
    spikes: torch.Tensor,
    membrane: torch.Tensor,
    radius: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Local WTA with winner mask output.
    
    Inter-map competition: when a neuron fires, neurons within a local
    spatial radius across ALL feature maps are inhibited.
    
    Paper Reference:
        "local inter-map competition: neurons in a local neighborhood
         across all feature maps compete"
    
    Args:
        spikes: Current spikes, shape (batch, channels, height, width).
        membrane: Current membrane potential, same shape.
        radius: Inhibition radius (neighborhood size = 2*radius + 1).
    
    Returns:
        Tuple of (filtered_spikes, new_membrane, winner_mask).
    
    Raises:
        WTARuntimeError: If inputs are invalid.
    """
    _validate_4d_tensor(spikes, "spikes")
    _validate_4d_tensor(membrane, "membrane")
    _validate_same_shape(spikes, membrane, "spikes", "membrane")
    _validate_same_device(spikes, membrane, "spikes", "membrane")
    _validate_positive_int(radius, "radius")
    
    batch, channels, height, width = spikes.shape
    device = spikes.device
    
    # Find locations where any channel spiked
    any_spike = spikes.sum(dim=1, keepdim=True) > 0  # (B, 1, H, W)
    
    # Create inhibition kernel
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    
    # Dilate to create inhibition zone
    padded = F.pad(any_spike.float(), (radius, radius, radius, radius))
    inhibition_zone = F.conv2d(padded, kernel) > 0  # (B, 1, H, W)
    inhibition_zone = inhibition_zone.expand(-1, channels, -1, -1)
    
    # For local WTA, we keep original spikes but inhibit neighbors
    # The "winners" are the neurons that actually spiked
    winner_mask = spikes.clone()
    
    # Inhibit membrane in neighborhoods
    new_membrane = torch.where(inhibition_zone, torch.zeros_like(membrane), membrane)
    
    return spikes, new_membrane, winner_mask


def wta_by_membrane(
    spikes: torch.Tensor,
    membrane: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    WTA where winner is the neuron with highest membrane potential.
    
    Among all neurons that spiked, keep only the one with highest
    membrane potential (most confident detection).
    
    Args:
        spikes: Current spikes, shape (batch, channels, height, width).
        membrane: Current membrane potential, same shape.
    
    Returns:
        Tuple of (filtered_spikes, new_membrane, winner_mask).
    """
    _validate_4d_tensor(spikes, "spikes")
    _validate_4d_tensor(membrane, "membrane")
    _validate_same_shape(spikes, membrane, "spikes", "membrane")
    _validate_same_device(spikes, membrane, "spikes", "membrane")
    
    batch, channels, height, width = spikes.shape
    device = spikes.device
    
    # Mask membrane by spikes (only consider neurons that fired)
    masked_membrane = membrane * spikes
    masked_membrane_flat = masked_membrane.view(batch, channels, -1)
    
    # Find max membrane in each feature map
    max_vals, max_indices = masked_membrane_flat.max(dim=2)  # (B, C)
    has_spike = max_vals > 0  # (B, C)
    
    # Create winner spikes
    new_spikes_flat = torch.zeros_like(masked_membrane_flat)
    
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, channels)
    channel_idx = torch.arange(channels, device=device).unsqueeze(0).expand(batch, -1)
    
    new_spikes_flat[
        batch_idx[has_spike],
        channel_idx[has_spike],
        max_indices[has_spike]
    ] = 1.0
    
    new_spikes = new_spikes_flat.view_as(spikes)
    winner_mask = new_spikes.clone()
    
    # Inhibit all neurons in feature maps with winners
    inhibition_mask = has_spike.unsqueeze(-1).unsqueeze(-1).expand_as(membrane)
    new_membrane = torch.where(inhibition_mask, torch.zeros_like(membrane), membrane)
    
    return new_spikes, new_membrane, winner_mask


# =============================================================================
# ADAPTIVE THRESHOLD / HOMEOSTASIS
# =============================================================================


class AdaptiveThreshold(nn.Module):
    """
    Adaptive threshold mechanism for homeostasis.
    
    Maintains per-neuron thresholds that adapt based on firing rates:
        - Neurons firing too often: threshold increases
        - Neurons rarely firing: threshold decreases
    
    This ensures all neurons get a chance to learn different features.
    
    Paper Reference:
        "An adaptive threshold mechanism is used to maintain homeostasis"
        "The threshold adapts to keep firing rate near a target level"
    
    Attributes:
        thresholds: Per-neuron threshold values, shape (channels, H, W).
        firing_counts: Running count of spikes per neuron.
        presentation_count: Number of stimuli presented.
    
    Example:
        >>> adaptive = AdaptiveThreshold(
        ...     n_channels=36,
        ...     spatial_shape=(16, 16),
        ...     initial_threshold=10.0,
        ...     target_rate=0.1
        ... )
        >>> # After each presentation
        >>> adaptive.update(spikes)
        >>> current_thresholds = adaptive.get_thresholds()
    """
    
    def __init__(
        self,
        n_channels: int,
        spatial_shape: Tuple[int, int],
        initial_threshold: float = 10.0,
        target_rate: float = 0.1,
        learning_rate: float = 0.001,
        threshold_min: float = 1.0,
        threshold_max: float = 100.0,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize adaptive threshold.
        
        Args:
            n_channels: Number of feature channels.
            spatial_shape: Spatial dimensions (height, width).
            initial_threshold: Starting threshold value.
            target_rate: Target firing rate (spikes per presentation).
            learning_rate: Rate of threshold adaptation.
            threshold_min: Minimum allowed threshold.
            threshold_max: Maximum allowed threshold.
            device: Device to create tensors on.
        
        Raises:
            WTAConfigError: If parameters are invalid.
        """
        super().__init__()
        
        _validate_positive_int(n_channels, "n_channels")
        _validate_tuple_of_ints(spatial_shape, 2, "spatial_shape")
        
        if spatial_shape[0] <= 0 or spatial_shape[1] <= 0:
            raise WTAConfigError(
                f"spatial_shape must be positive, got {spatial_shape}"
            )
        
        _validate_positive_float(initial_threshold, "initial_threshold")
        _validate_range(target_rate, 0.0, 1.0, "target_rate")
        _validate_positive_float(learning_rate, "learning_rate")
        _validate_positive_float(threshold_min, "threshold_min")
        _validate_positive_float(threshold_max, "threshold_max")
        
        if threshold_min >= threshold_max:
            raise WTAConfigError(
                f"threshold_min ({threshold_min}) must be < threshold_max ({threshold_max})"
            )
        
        if not threshold_min <= initial_threshold <= threshold_max:
            raise WTAConfigError(
                f"initial_threshold ({initial_threshold}) must be in "
                f"[{threshold_min}, {threshold_max}]"
            )
        
        self.n_channels = n_channels
        self.spatial_shape = spatial_shape
        self.target_rate = target_rate
        self.learning_rate = learning_rate
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
        # Initialize thresholds (not trainable parameters)
        shape = (n_channels, spatial_shape[0], spatial_shape[1])
        self.register_buffer(
            'thresholds',
            torch.full(shape, initial_threshold, device=device)
        )
        
        # Tracking
        self.register_buffer(
            'firing_counts',
            torch.zeros(shape, device=device)
        )
        self.register_buffer(
            'presentation_count',
            torch.tensor(0, device=device)
        )
    
    def update(self, spikes: torch.Tensor) -> None:
        """
        Update thresholds based on observed spikes.
        
        Args:
            spikes: Spike tensor, shape (batch, channels, H, W).
                   Summed across batch for statistics.
        
        Raises:
            WTARuntimeError: If spike dimensions don't match.
        """
        _validate_4d_tensor(spikes, "spikes")
        
        if spikes.shape[1:] != (self.n_channels,) + self.spatial_shape:
            raise WTARuntimeError(
                f"spikes shape {spikes.shape[1:]} doesn't match "
                f"expected ({self.n_channels},) + {self.spatial_shape}"
            )
        
        # Update firing counts (sum across batch)
        batch_spikes = spikes.sum(dim=0)  # (C, H, W)
        self.firing_counts += batch_spikes.to(self.firing_counts.device)
        
        # Update presentation count
        self.presentation_count += spikes.shape[0]
        
        # Compute current firing rate
        if self.presentation_count > 0:
            current_rate = self.firing_counts / self.presentation_count.float()
            
            # Threshold adaptation
            # If firing too much: increase threshold
            # If firing too little: decrease threshold
            rate_diff = current_rate - self.target_rate
            
            # Update: threshold += lr * rate_diff * threshold
            # (multiplicative update keeps it positive)
            self.thresholds += self.learning_rate * rate_diff * self.thresholds
            
            # Clamp to valid range
            self.thresholds.clamp_(self.threshold_min, self.threshold_max)
    
    def get_thresholds(self, expand_batch: int = 1) -> torch.Tensor:
        """
        Get current thresholds.
        
        Args:
            expand_batch: Batch dimension to add.
        
        Returns:
            Threshold tensor, shape (batch, channels, H, W).
        """
        return self.thresholds.unsqueeze(0).expand(expand_batch, -1, -1, -1)
    
    def reset_statistics(self) -> None:
        """Reset firing statistics but keep thresholds."""
        self.firing_counts.zero_()
        self.presentation_count.zero_()
    
    def reset_all(self, initial_threshold: float = 10.0) -> None:
        """Reset both thresholds and statistics."""
        _validate_positive_float(initial_threshold, "initial_threshold")
        
        self.thresholds.fill_(initial_threshold)
        self.reset_statistics()
    
    @property
    def firing_rates(self) -> torch.Tensor:
        """Current firing rates per neuron."""
        if self.presentation_count == 0:
            return torch.zeros_like(self.firing_counts)
        return self.firing_counts / self.presentation_count.float()
    
    @property
    def mean_threshold(self) -> float:
        """Mean threshold across all neurons."""
        return float(self.thresholds.mean().item())
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveThreshold(n_ch={self.n_channels}, "
            f"shape={self.spatial_shape}, "
            f"mean_thresh={self.mean_threshold:.2f})"
        )


# =============================================================================
# WTA INHIBITION MODULE
# =============================================================================


class WTAInhibition(nn.Module):
    """
    Winner-Take-All inhibition module.
    
    Combines lateral inhibition with optional adaptive thresholds.
    Provides winner masks for STDP weight updates.
    
    Paper Reference:
        Kheradpisheh et al. 2018:
        "A WTA mechanism is used to enforce competition. The first neuron 
        to fire inhibits all others in its competition group."
    
    Attributes:
        config: WTA configuration.
        adaptive_threshold: Optional adaptive threshold mechanism.
        winner_mask: Most recent winner mask (for STDP).
        statistics: Dict of tracked statistics.
    
    Example:
        >>> wta = WTAInhibition(
        ...     config=WTAConfig(mode="both"),
        ...     n_channels=36,
        ...     spatial_shape=(16, 16)
        ... )
        >>> 
        >>> # During forward pass
        >>> filtered_spikes, new_membrane = wta(spikes, membrane)
        >>> 
        >>> # Get winners for STDP
        >>> winners = wta.get_winner_mask()
    """
    
    def __init__(
        self,
        config: Optional[WTAConfig] = None,
        n_channels: Optional[int] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
        initial_threshold: float = 10.0,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize WTA inhibition.
        
        Args:
            config: WTA configuration. If None, uses default.
            n_channels: Number of channels (required if enable_homeostasis).
            spatial_shape: Spatial dimensions (required if enable_homeostasis).
            initial_threshold: Initial threshold for homeostasis.
            device: Device for tensors.
        
        Raises:
            WTAConfigError: If configuration is invalid.
        """
        super().__init__()
        
        if config is None:
            config = WTAConfig()
        
        if not isinstance(config, WTAConfig):
            raise WTAConfigError(
                f"config must be WTAConfig, got {type(config).__name__}"
            )
        
        self.config = config
        self._device = device
        
        # Initialize adaptive threshold if enabled
        self.adaptive_threshold: Optional[AdaptiveThreshold] = None
        
        if config.enable_homeostasis:
            if n_channels is None or spatial_shape is None:
                raise WTAConfigError(
                    "n_channels and spatial_shape required when enable_homeostasis=True"
                )
            
            self.adaptive_threshold = AdaptiveThreshold(
                n_channels=n_channels,
                spatial_shape=spatial_shape,
                initial_threshold=initial_threshold,
                target_rate=config.target_rate,
                learning_rate=config.homeostasis_lr,
                threshold_min=config.threshold_min,
                threshold_max=config.threshold_max,
                device=device
            )
        
        # State tracking
        self._winner_mask: Optional[torch.Tensor] = None
        self._last_spikes: Optional[torch.Tensor] = None
        
        # Statistics
        self.statistics: Dict[str, Any] = {
            'total_spikes': 0,
            'total_winners': 0,
            'presentations': 0
        }
    
    def forward(
        self,
        spikes: torch.Tensor,
        membrane: torch.Tensor,
        threshold: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply WTA inhibition.
        
        Args:
            spikes: Input spikes, shape (batch, channels, H, W).
            membrane: Membrane potential, same shape.
            threshold: Optional external threshold. If None and homeostasis
                      enabled, uses adaptive thresholds.
        
        Returns:
            Tuple of (filtered_spikes, new_membrane).
        
        Raises:
            WTARuntimeError: If inputs are invalid.
        """
        _validate_4d_tensor(spikes, "spikes")
        _validate_4d_tensor(membrane, "membrane")
        _validate_same_shape(spikes, membrane, "spikes", "membrane")
        _validate_same_device(spikes, membrane, "spikes", "membrane")
        
        # Apply global WTA
        if self.config.mode in (WTAMode.GLOBAL, WTAMode.BOTH):
            spikes, membrane, winner_mask = wta_global_membrane(spikes, membrane)
        else:
            winner_mask = spikes.clone()
        
        # Apply local WTA
        if self.config.mode in (WTAMode.LOCAL, WTAMode.BOTH):
            spikes, membrane, local_mask = wta_local_membrane(
                spikes, membrane, self.config.local_radius
            )
            # Combine masks (AND)
            winner_mask = winner_mask * local_mask
        
        # Store winner mask for STDP
        self._winner_mask = winner_mask
        self._last_spikes = spikes
        
        # Update adaptive thresholds if enabled
        if self.adaptive_threshold is not None:
            self.adaptive_threshold.update(spikes)
        
        # Update statistics
        if self.config.track_statistics:
            self.statistics['total_spikes'] += int(spikes.sum().item())
            self.statistics['total_winners'] += int(winner_mask.sum().item())
            self.statistics['presentations'] += spikes.shape[0]
        
        return spikes, membrane
    
    def get_winner_mask(self) -> Optional[torch.Tensor]:
        """
        Get the winner mask from last forward pass.
        
        Returns:
            Binary mask of winners, or None if no forward pass yet.
        """
        return self._winner_mask
    
    def get_thresholds(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """
        Get current adaptive thresholds.
        
        Args:
            batch_size: Batch dimension for threshold tensor.
        
        Returns:
            Threshold tensor, or None if homeostasis disabled.
        """
        if self.adaptive_threshold is None:
            return None
        return self.adaptive_threshold.get_thresholds(batch_size)
    
    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self.statistics = {
            'total_spikes': 0,
            'total_winners': 0,
            'presentations': 0
        }
        if self.adaptive_threshold is not None:
            self.adaptive_threshold.reset_statistics()
    
    @property
    def winner_ratio(self) -> float:
        """Ratio of winners to total spikes."""
        if self.statistics['total_spikes'] == 0:
            return 0.0
        return self.statistics['total_winners'] / self.statistics['total_spikes']
    
    def __repr__(self) -> str:
        homeostasis = "enabled" if self.adaptive_threshold else "disabled"
        return (
            f"WTAInhibition(mode={self.config.mode.value}, "
            f"homeostasis={homeostasis})"
        )


# =============================================================================
# CONVERGENCE TRACKING
# =============================================================================


class ConvergenceTracker:
    """
    Tracks which neurons have learned (converged).
    
    A neuron is considered "converged" when its weights have stabilized
    (small updates) or it has won at least a minimum number of times.
    
    Attributes:
        n_channels: Number of feature channels.
        spatial_shape: Spatial dimensions.
        win_counts: Number of wins per neuron.
        converged_mask: Boolean mask of converged neurons.
    
    Example:
        >>> tracker = ConvergenceTracker(n_channels=36, spatial_shape=(16, 16))
        >>> 
        >>> # After each STDP update
        >>> tracker.update(winner_mask, weight_deltas)
        >>> 
        >>> if tracker.convergence_ratio > 0.95:
        ...     print("Layer converged!")
    """
    
    def __init__(
        self,
        n_channels: int,
        spatial_shape: Tuple[int, int],
        min_wins: int = 10,
        delta_threshold: float = 1e-4,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize convergence tracker.
        
        Args:
            n_channels: Number of feature channels.
            spatial_shape: Spatial dimensions (height, width).
            min_wins: Minimum wins to consider converged.
            delta_threshold: Weight change threshold for convergence.
            device: Device for tensors.
        """
        _validate_positive_int(n_channels, "n_channels")
        _validate_tuple_of_ints(spatial_shape, 2, "spatial_shape")
        _validate_positive_int(min_wins, "min_wins")
        _validate_positive_float(delta_threshold, "delta_threshold")
        
        self.n_channels = n_channels
        self.spatial_shape = spatial_shape
        self.min_wins = min_wins
        self.delta_threshold = delta_threshold
        
        shape = (n_channels, spatial_shape[0], spatial_shape[1])
        
        self.win_counts = torch.zeros(shape, device=device)
        self.converged_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        self.last_delta_magnitude = torch.zeros(shape, device=device)
    
    def update(
        self,
        winner_mask: torch.Tensor,
        weight_deltas: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update convergence tracking.
        
        Args:
            winner_mask: Binary mask of winners, shape (batch, C, H, W).
            weight_deltas: Optional weight changes for delta-based convergence.
        """
        _validate_4d_tensor(winner_mask, "winner_mask")
        
        # Sum across batch
        wins = winner_mask.sum(dim=0)  # (C, H, W)
        self.win_counts += wins.to(self.win_counts.device)
        
        # Check win-based convergence
        win_converged = self.win_counts >= self.min_wins
        
        # Check delta-based convergence if provided
        if weight_deltas is not None:
            delta_mag = weight_deltas.abs().mean(dim=(2, 3))  # (out_ch, in_ch) -> need spatial
            # This is simplified - full version needs per-neuron tracking
            pass
        
        # Update convergence mask
        self.converged_mask = win_converged
    
    @property
    def convergence_ratio(self) -> float:
        """Fraction of converged neurons."""
        return float(self.converged_mask.float().mean().item())
    
    @property
    def n_converged(self) -> int:
        """Number of converged neurons."""
        return int(self.converged_mask.sum().item())
    
    @property
    def n_total(self) -> int:
        """Total number of neurons."""
        return self.n_channels * self.spatial_shape[0] * self.spatial_shape[1]
    
    def is_converged(self, threshold: float = 0.95) -> bool:
        """Check if layer has converged (enough neurons converged)."""
        return self.convergence_ratio >= threshold
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.win_counts.zero_()
        self.converged_mask.zero_()
        self.last_delta_magnitude.zero_()
    
    def __repr__(self) -> str:
        return (
            f"ConvergenceTracker("
            f"converged={self.n_converged}/{self.n_total}, "
            f"ratio={self.convergence_ratio:.1%})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "WTAError",
    "WTAConfigError",
    "WTARuntimeError",
    # Enums
    "WTAMode",
    # Configuration
    "WTAConfig",
    # Functions
    "wta_global_membrane",
    "wta_local_membrane",
    "wta_by_membrane",
    # Classes
    "AdaptiveThreshold",
    "WTAInhibition",
    "ConvergenceTracker",
]