"""
STDP: Spike-Timing Dependent Plasticity.

This module implements unsupervised learning rules for spiking neural networks
based on the biological principle that synaptic strength changes depend on
the relative timing of pre- and post-synaptic spikes.

Paper References:
    Kheradpisheh et al. 2018 - "STDP-based spiking deep convolutional neural 
    networks for object recognition"
    
    Kirkland et al. 2023 (IGARSS) - "Neuromorphic sensing and processing for 
    space domain awareness"

STDP Rule (Equation 3 from Kheradpisheh 2018):
    Δw_ij = a⁺ · w_ij · (1 - w_ij)    if t_j - t_i ≤ 0  (pre fires before/at post → LTP)
    Δw_ij = -a⁻ · w_ij · (1 - w_ij)   if t_j - t_i > 0  (pre fires after post → LTD)
    
    Where:
        - i = post-synaptic neuron index
        - j = pre-synaptic neuron index
        - t_i, t_j = spike times
        - a⁺, a⁻ = learning rates
        - The multiplicative term w·(1-w) provides soft bounds in [0, 1]

Key Insight (from paper):
    "The exact time difference between two spikes does not affect the weight 
    change, but only its sign is considered. Also, it is assumed that if a 
    presynaptic neuron does not fire before the postsynaptic one, it will 
    fire later."

Winner-Take-All Integration:
    - Global intra-map competition: First neuron to fire in feature map wins
    - Local inter-map competition: Winner prevents nearby neurons in OTHER maps
    - Weight sharing: Winner's updated weights copied to all neurons in same map

Convergence Metric (Equation 4):
    C_l = Σ_f Σ_i w_f,i · (1 - w_f,i) / n_w
    
    Stop training when C_l < 0.01 (weights converged to 0 or 1)

Example:
    >>> from spikeseg.learning.stdp import STDPLearner, STDPConfig
    >>> 
    >>> # Create learner with paper parameters
    >>> config = STDPConfig.from_paper("kheradpisheh2018")
    >>> learner = STDPLearner(config)
    >>> 
    >>> # Training loop
    >>> for image in dataset:
    ...     pre_times, post_times, winner_idx = forward_pass(image)
    ...     learner.update_weights(layer.weight, pre_times, post_times, winner_idx)
    ...     
    ...     if learner.has_converged():
    ...         print(f"Converged after {learner.update_count} updates")
    ...         break
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class STDPError(Exception):
    """Base exception for STDP module errors."""
    pass


class STDPConfigError(STDPError):
    """Raised for invalid STDP configuration."""
    pass


class STDPRuntimeError(STDPError):
    """Raised for runtime errors during learning."""
    pass


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_tensor(tensor: Any, name: str) -> None:
    """Validate that input is a torch.Tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")


def _validate_tensor_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Validate tensor has expected number of dimensions."""
    if tensor.dim() != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D, got {tensor.dim()}D with shape {tensor.shape}"
        )


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
    """Validate that a value is within a specified range [min_val, max_val]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ValueError(
            f"{name} must be in range [{min_val}, {max_val}], got {value}"
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


def _validate_same_device(t1: torch.Tensor, t2: torch.Tensor,
                          name1: str, name2: str) -> None:
    """Validate two tensors are on the same device."""
    if t1.device != t2.device:
        raise ValueError(
            f"{name1} and {name2} must be on same device. "
            f"Got {t1.device} and {t2.device}"
        )


# =============================================================================
# STDP CONFIGURATION
# =============================================================================


class STDPVariant(Enum):
    """
    STDP learning rule variants.
    
    Attributes:
        MULTIPLICATIVE: Δw = α · w · (1 - w) - soft bounds (Kheradpisheh 2018)
        ADDITIVE: Δw = α - requires explicit weight clamping
    """
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


@dataclass
class STDPConfig:
    """
    Configuration for STDP learning.
    
    Paper Reference (Kheradpisheh 2018):
        - a⁺ = 0.004 (potentiation learning rate)
        - a⁻ = 0.003 (depression learning rate)  
        - Weight initialization: Normal(μ=0.8, σ=0.05)
        - Convergence threshold: 0.01
    
    Paper Reference (IGARSS 2023):
        - a⁺ = 0.04, a⁻ = 0.03 (10x higher learning rates)
        - Weight initialization: 0.8 ± 0.01
    
    Attributes:
        lr_plus: Learning rate for LTP (potentiation). Pre fires before post.
        lr_minus: Learning rate for LTD (depression). Pre fires after post.
        weight_min: Minimum weight value for clamping. Default 0.0.
        weight_max: Maximum weight value for clamping. Default 1.0.
        variant: STDP variant (MULTIPLICATIVE or ADDITIVE).
        convergence_threshold: Stop training when convergence metric < this.
        weight_init_mean: Mean for weight initialization. Default 0.8.
        weight_init_std: Std for weight initialization. Default 0.01 (IGARSS 2023).
    
    Example:
        >>> config = STDPConfig(lr_plus=0.004, lr_minus=0.003)
        >>> config = STDPConfig.from_paper("kheradpisheh2018")
    """
    lr_plus: float = 0.004
    lr_minus: float = 0.003
    weight_min: float = 0.0
    weight_max: float = 1.0
    variant: STDPVariant = STDPVariant.MULTIPLICATIVE
    convergence_threshold: float = 0.01
    weight_init_mean: float = 0.8
    weight_init_std: float = 0.01  # IGARSS 2023: 0.01

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        _validate_positive_float(self.lr_plus, "lr_plus")
        _validate_positive_float(self.lr_minus, "lr_minus")
        _validate_non_negative_float(self.weight_min, "weight_min")
        _validate_positive_float(self.weight_max, "weight_max")
        _validate_positive_float(self.convergence_threshold, "convergence_threshold")
        _validate_positive_float(self.weight_init_std, "weight_init_std")
        
        if self.weight_min >= self.weight_max:
            raise STDPConfigError(
                f"weight_min ({self.weight_min}) must be < weight_max ({self.weight_max})"
            )
        
        _validate_range(
            self.weight_init_mean, 
            self.weight_min, 
            self.weight_max, 
            "weight_init_mean"
        )
        
        if not isinstance(self.variant, STDPVariant):
            raise STDPConfigError(
                f"variant must be STDPVariant, got {type(self.variant).__name__}"
            )
    
    @classmethod
    def from_paper(cls, paper: str) -> "STDPConfig":
        """
        Create configuration from paper parameters.
        
        Args:
            paper: Paper identifier. Options:
                - "kheradpisheh2018": Original STDP-CNN paper
                - "igarss2023": Space domain awareness paper
        
        Returns:
            STDPConfig with paper parameters.
        
        Raises:
            STDPConfigError: If paper not recognized.
        """
        configs = {
            "kheradpisheh2018": cls(
                lr_plus=0.004,
                lr_minus=0.003,
                weight_init_mean=0.8,
                weight_init_std=0.05,
                convergence_threshold=0.01,
                variant=STDPVariant.MULTIPLICATIVE
            ),
            "igarss2023": cls(
                lr_plus=0.04,
                lr_minus=0.03,
                weight_init_mean=0.8,
                weight_init_std=0.05,  # Kheradpisheh 2018: 0.05 for weight diversity
                convergence_threshold=0.01,
                variant=STDPVariant.MULTIPLICATIVE
            ),
        }
        
        paper_lower = paper.lower()
        if paper_lower not in configs:
            raise STDPConfigError(
                f"Unknown paper '{paper}'. Available: {list(configs.keys())}"
            )
        
        return configs[paper_lower]
    
    def __repr__(self) -> str:
        return (
            f"STDPConfig(a⁺={self.lr_plus}, a⁻={self.lr_minus}, "
            f"variant={self.variant.value})"
        )


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================


def initialize_weights(
    shape: Tuple[int, ...],
    mean: float = 0.8,
    std: float = 0.05,
    weight_min: float = 0.0,
    weight_max: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Initialize weights for STDP learning.
    
    Paper Reference (Kheradpisheh 2018):
        "Synaptic weights of convolutional neurons initiate with random 
        values drawn from a normal distribution with the mean of μ = 0.8 
        and STD of σ = 0.05."
    
    Args:
        shape: Shape of weight tensor (out_ch, in_ch, kH, kW).
        mean: Mean of normal distribution. Default 0.8.
        std: Standard deviation. Default 0.05.
        weight_min: Minimum weight (for clamping). Default 0.0.
        weight_max: Maximum weight (for clamping). Default 1.0.
        device: Device to create tensor on.
        dtype: Data type for tensor.
    
    Returns:
        Initialized weight tensor clamped to [weight_min, weight_max].
    
    Raises:
        ValueError: If parameters are invalid.
    
    Example:
        >>> weights = initialize_weights((4, 2, 5, 5), mean=0.8, std=0.05)
        >>> weights.shape
        torch.Size([4, 2, 5, 5])
        >>> 0.0 <= weights.min() and weights.max() <= 1.0
        True
    """
    _validate_positive_float(std, "std")
    _validate_range(mean, weight_min, weight_max, "mean")
    
    if weight_min >= weight_max:
        raise ValueError(
            f"weight_min ({weight_min}) must be < weight_max ({weight_max})"
        )
    
    # Create from normal distribution
    weights = torch.empty(shape, device=device, dtype=dtype)
    weights.normal_(mean=mean, std=std)
    
    # Clamp to valid range
    weights.clamp_(weight_min, weight_max)
    
    return weights


# =============================================================================
# CONVERGENCE METRIC
# =============================================================================


def compute_convergence_metric(weights: torch.Tensor) -> float:
    """
    Compute STDP convergence metric.
    
    Paper Reference (Equation 4 from Kheradpisheh 2018):
        C_l = Σ_f Σ_i w_f,i · (1 - w_f,i) / n_w
    
    This metric approaches 0 as weights converge to 0 or 1.
    Maximum value is 0.25 (when all weights = 0.5).
    
    Args:
        weights: Weight tensor of any shape.
    
    Returns:
        Convergence metric in [0, 0.25]. Lower = more converged.
    
    Example:
        >>> w_random = torch.rand(10, 10) * 0.5 + 0.25  # Around 0.5
        >>> w_converged = torch.zeros(10, 10)
        >>> w_converged[::2] = 1.0  # Half 0, half 1
        >>> compute_convergence_metric(w_random) > compute_convergence_metric(w_converged)
        True
    """
    _validate_tensor(weights, "weights")
    
    # w * (1 - w) is maximum at w=0.5, zero at w=0 or w=1
    convergence_term = weights * (1.0 - weights)
    
    # Average over all weights
    n_weights = weights.numel()
    if n_weights == 0:
        return 0.0
    
    return float(convergence_term.sum() / n_weights)


def has_converged(
    weights: torch.Tensor, 
    threshold: float = 0.01
) -> bool:
    """
    Check if weights have converged.
    
    Args:
        weights: Weight tensor.
        threshold: Convergence threshold. Default 0.01 (from paper).
    
    Returns:
        True if convergence metric < threshold.
    """
    _validate_positive_float(threshold, "threshold")
    return compute_convergence_metric(weights) < threshold


# =============================================================================
# SPIKE TIMING UTILITIES
# =============================================================================


def get_first_spike_times(
    spikes: torch.Tensor,
    no_spike_value: float = float('inf')
) -> torch.Tensor:
    """
    Get the time of first spike for each neuron.
    
    For STDP, we only consider the first spike time (simplified rule).
    
    Args:
        spikes: Spike tensor of shape (n_timesteps, ...).
                Can be (T, C, H, W) or (T, C) or (T, N).
        no_spike_value: Value for neurons that never spike.
                       Default inf (loses all timing comparisons).
    
    Returns:
        First spike time tensor with shape (...) - time dimension removed.
        Values in [0, n_timesteps-1] or no_spike_value.
    
    Example:
        >>> spikes = torch.zeros(10, 4, 8, 8)
        >>> spikes[3, 0, 2, 2] = 1  # Neuron fires at t=3
        >>> spikes[5, 0, 2, 2] = 1  # Same neuron fires again at t=5
        >>> times = get_first_spike_times(spikes)
        >>> times[0, 2, 2]  # First spike at t=3
        tensor(3.)
    """
    _validate_tensor(spikes, "spikes")
    
    if spikes.dim() < 1:
        raise ValueError("spikes must have at least 1 dimension (time)")
    
    n_timesteps = spikes.shape[0]
    spatial_shape = spikes.shape[1:]
    device = spikes.device
    dtype = spikes.dtype
    
    # Initialize with no_spike_value
    first_times = torch.full(
        spatial_shape, 
        no_spike_value, 
        device=device, 
        dtype=torch.float32
    )
    
    # Iterate through time to find first spike
    for t in range(n_timesteps):
        # Neurons that spike at t AND haven't spiked yet
        newly_spiked = (spikes[t] > 0) & (first_times == no_spike_value)
        first_times[newly_spiked] = float(t)
    
    return first_times


def extract_receptive_field_times(
    pre_spike_times: torch.Tensor,
    post_y: int,
    post_x: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """
    Extract pre-synaptic spike times for a specific post-synaptic neuron.
    
    Given the output location (post_y, post_x), extract the corresponding
    input patch that feeds into that neuron through the convolution.
    
    Args:
        pre_spike_times: First spike times, shape (in_channels, H_in, W_in).
        post_y: Y coordinate of post-synaptic neuron.
        post_x: X coordinate of post-synaptic neuron.
        kernel_size: Size of convolution kernel.
        stride: Convolution stride. Default 1.
        padding: Convolution padding. Default 0.
    
    Returns:
        Spike times for receptive field, shape (in_channels, kernel_size, kernel_size).
    
    Raises:
        ValueError: If coordinates are out of bounds.
    """
    _validate_tensor(pre_spike_times, "pre_spike_times")
    _validate_tensor_dim(pre_spike_times, 3, "pre_spike_times")
    _validate_positive_int(kernel_size, "kernel_size")
    _validate_positive_int(stride, "stride")
    _validate_non_negative_int(padding, "padding")
    _validate_non_negative_int(post_y, "post_y")
    _validate_non_negative_int(post_x, "post_x")
    
    in_channels, h_in, w_in = pre_spike_times.shape
    
    # Compute input coordinates
    y_start = post_y * stride - padding
    x_start = post_x * stride - padding
    y_end = y_start + kernel_size
    x_end = x_start + kernel_size
    
    # Handle padding (use inf for padded regions - they never spike)
    device = pre_spike_times.device
    rf_times = torch.full(
        (in_channels, kernel_size, kernel_size),
        float('inf'),
        device=device
    )
    
    # Compute valid region
    src_y_start = max(0, y_start)
    src_y_end = min(h_in, y_end)
    src_x_start = max(0, x_start)
    src_x_end = min(w_in, x_end)
    
    dst_y_start = src_y_start - y_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    
    # Copy valid region
    if src_y_end > src_y_start and src_x_end > src_x_start:
        rf_times[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            pre_spike_times[:, src_y_start:src_y_end, src_x_start:src_x_end]
    
    return rf_times


# =============================================================================
# STDP WEIGHT UPDATE
# =============================================================================


def compute_stdp_update(
    weights: torch.Tensor,
    pre_spike_times: torch.Tensor,
    post_spike_time: float,
    lr_plus: float,
    lr_minus: float,
    variant: STDPVariant = STDPVariant.MULTIPLICATIVE
) -> torch.Tensor:
    """
    Compute STDP weight update for a single post-synaptic spike.
    
    Paper Reference (Equation 3 from Kheradpisheh 2018):
        Δw_ij = a⁺ · w_ij · (1 - w_ij)    if t_j ≤ t_i  (pre before/at post → LTP)
        Δw_ij = -a⁻ · w_ij · (1 - w_ij)   if t_j > t_i  (pre after post → LTD)
    
    Simplification from paper:
        "The exact time difference between two spikes does not affect the 
        weight change, but only its sign is considered."
    
    Args:
        weights: Current weights, shape (in_channels, kH, kW).
        pre_spike_times: First spike times of pre-synaptic neurons.
                        Shape: (in_channels, kH, kW). inf = no spike.
        post_spike_time: Spike time of post-synaptic neuron.
        lr_plus: Learning rate for LTP (a⁺).
        lr_minus: Learning rate for LTD (a⁻).
        variant: STDP variant (MULTIPLICATIVE or ADDITIVE).
    
    Returns:
        Weight update tensor, same shape as weights.
    
    Example:
        >>> weights = torch.full((4, 5, 5), 0.5)
        >>> pre_times = torch.full((4, 5, 5), 2.0)  # All pre spike at t=2
        >>> post_time = 3.0  # Post spikes at t=3
        >>> delta = compute_stdp_update(weights, pre_times, post_time, 0.004, 0.003)
        >>> (delta > 0).all()  # All LTP since pre < post
        True
    """
    _validate_tensor(weights, "weights")
    _validate_tensor(pre_spike_times, "pre_spike_times")
    _validate_positive_float(lr_plus, "lr_plus")
    _validate_positive_float(lr_minus, "lr_minus")
    
    if weights.shape != pre_spike_times.shape:
        raise ValueError(
            f"weights shape {weights.shape} doesn't match "
            f"pre_spike_times shape {pre_spike_times.shape}"
        )
    
    # LTP mask: pre fires before or at post (t_pre <= t_post)
    ltp_mask = (pre_spike_times <= post_spike_time).float()
    
    # LTD mask: pre fires after post (t_pre > t_post)
    # Note: inf > any finite number, so neurons that don't spike get LTD
    ltd_mask = (pre_spike_times > post_spike_time).float()
    
    if variant == STDPVariant.MULTIPLICATIVE:
        # Soft bounds term: w * (1 - w)
        soft_bounds = weights * (1.0 - weights)
        
        # LTP: positive update
        delta_ltp = lr_plus * soft_bounds * ltp_mask
        
        # LTD: negative update
        delta_ltd = -lr_minus * soft_bounds * ltd_mask
        
    elif variant == STDPVariant.ADDITIVE:
        # Simple additive (requires clamping after)
        delta_ltp = lr_plus * ltp_mask
        delta_ltd = -lr_minus * ltd_mask
    
    else:
        raise STDPRuntimeError(f"Unknown STDP variant: {variant}")
    
    return delta_ltp + delta_ltd


# =============================================================================
# STDP LEARNER
# =============================================================================


@dataclass
class STDPStats:
    """
    Statistics from STDP learning.
    
    Attributes:
        n_updates: Total number of weight updates.
        n_ltp: Number of LTP events (potentiation).
        n_ltd: Number of LTD events (depression).
        mean_delta: Mean absolute weight change.
        convergence_history: History of convergence metric values.
    """
    n_updates: int = 0
    n_ltp: int = 0
    n_ltd: int = 0
    mean_delta: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    
    def record_update(
        self, 
        delta_w: torch.Tensor, 
        convergence: float
    ) -> None:
        """Record statistics from a weight update."""
        self.n_updates += 1
        self.n_ltp += int((delta_w > 0).sum().item())
        self.n_ltd += int((delta_w < 0).sum().item())
        self.mean_delta = float(delta_w.abs().mean().item())
        self.convergence_history.append(convergence)
    
    @property
    def ltp_ratio(self) -> float:
        """Ratio of LTP to total events."""
        total = self.n_ltp + self.n_ltd
        return self.n_ltp / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return (
            f"STDPStats(updates={self.n_updates}, "
            f"LTP/LTD={self.n_ltp}/{self.n_ltd}, "
            f"mean_Δw={self.mean_delta:.6f})"
        )


class STDPLearner:
    """
    STDP learning rule manager for convolutional layers.
    
    Handles weight updates using spike-timing dependent plasticity,
    including Winner-Take-All integration and convergence tracking.
    
    Paper Reference (Kheradpisheh 2018):
        "When a new image is presented, neurons of the convolutional layer 
        compete with each other and those which fire earlier trigger STDP 
        and learn the input pattern."
    
    Layer-wise Training:
        "Learning is done layer by layer, i.e., the learning in a convolutional 
        layer starts when the learning in the previous convolutional layer is 
        finalized."
    
    Attributes:
        config: STDP configuration parameters.
        stats: Learning statistics.
    
    Example:
        >>> config = STDPConfig.from_paper("kheradpisheh2018")
        >>> learner = STDPLearner(config)
        >>> 
        >>> # Initialize weights
        >>> weights = learner.initialize_weights((4, 2, 5, 5))
        >>> 
        >>> # Training loop
        >>> for image in dataset:
        ...     pre_times = get_first_spike_times(input_spikes)
        ...     post_times = get_first_spike_times(output_spikes)
        ...     winner_map, winner_loc = find_wta_winner(output_spikes)
        ...     
        ...     if winner_map is not None:
        ...         learner.update_weights_for_winner(
        ...             weights=weights,
        ...             pre_spike_times=pre_times,
        ...             post_spike_time=post_times[winner_map, winner_loc[0], winner_loc[1]],
        ...             winner_y=winner_loc[0],
        ...             winner_x=winner_loc[1],
        ...             kernel_size=5
        ...         )
        ...     
        ...     if learner.has_converged(weights):
        ...         break
    """
    
    def __init__(self, config: Optional[STDPConfig] = None) -> None:
        """
        Initialize STDP learner.
        
        Args:
            config: STDP configuration. Uses Kheradpisheh2018 defaults if None.
        
        Raises:
            STDPConfigError: If config is invalid.
        """
        if config is None:
            config = STDPConfig.from_paper("kheradpisheh2018")
        
        if not isinstance(config, STDPConfig):
            raise STDPConfigError(
                f"config must be STDPConfig, got {type(config).__name__}"
            )
        
        self.config = config
        self.stats = STDPStats()
    
    def initialize_weights(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Initialize weights using config parameters.
        
        Args:
            shape: Weight tensor shape (out_ch, in_ch, kH, kW).
            device: Device for tensor.
            dtype: Data type for tensor.
        
        Returns:
            Initialized weight tensor.
        """
        return initialize_weights(
            shape=shape,
            mean=self.config.weight_init_mean,
            std=self.config.weight_init_std,
            weight_min=self.config.weight_min,
            weight_max=self.config.weight_max,
            device=device,
            dtype=dtype
        )
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_time: float
    ) -> torch.Tensor:
        """
        Compute STDP weight update (without applying).
        
        Args:
            weights: Current weights for ONE feature map, shape (in_ch, kH, kW).
            pre_spike_times: Pre-synaptic spike times, shape (in_ch, kH, kW).
            post_spike_time: Post-synaptic spike time (scalar).
        
        Returns:
            Weight update tensor.
        """
        return compute_stdp_update(
            weights=weights,
            pre_spike_times=pre_spike_times,
            post_spike_time=post_spike_time,
            lr_plus=self.config.lr_plus,
            lr_minus=self.config.lr_minus,
            variant=self.config.variant
        )
    
    def compute_batch_update(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_times: torch.Tensor,
        winner_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute STDP weight update for batch of neurons (tensor-based).
        
        This is a vectorized version that handles multiple neurons at once,
        suitable for training scripts that process entire feature maps.
        
        Args:
            weights: Weight tensor, shape (out_ch, in_ch, kH, kW).
            pre_spike_times: Pre-synaptic first spike times, 
                            shape (batch, in_ch, H, W) or (in_ch, H, W).
            post_spike_times: Post-synaptic first spike times,
                             shape (batch, out_ch, H, W) or (out_ch, H, W).
            winner_mask: Optional binary mask indicating WTA winners,
                        shape (batch, out_ch, H, W) or (out_ch, H, W).
                        If None, all post-synaptic spikes trigger updates.
        
        Returns:
            Weight update tensor, same shape as weights.
        
        Note:
            This computes an aggregate update across all neurons. The update
            is averaged across spatial locations where winners occur.
        """
        _validate_tensor(weights, "weights")
        _validate_tensor(pre_spike_times, "pre_spike_times")
        _validate_tensor(post_spike_times, "post_spike_times")
        
        if winner_mask is not None:
            _validate_tensor(winner_mask, "winner_mask")
        
        # Ensure 4D tensors (add batch dim if needed)
        if pre_spike_times.dim() == 3:
            pre_spike_times = pre_spike_times.unsqueeze(0)
        if post_spike_times.dim() == 3:
            post_spike_times = post_spike_times.unsqueeze(0)
        if winner_mask is not None and winner_mask.dim() == 3:
            winner_mask = winner_mask.unsqueeze(0)
        
        batch_size, out_ch, H_out, W_out = post_spike_times.shape
        _, in_ch, H_in, W_in = pre_spike_times.shape
        _, _, kH, kW = weights.shape
        
        # Initialize weight update
        delta_w = torch.zeros_like(weights)
        n_updates = 0
        
        # Determine which neurons to update
        if winner_mask is not None:
            update_locations = winner_mask > 0
        else:
            # Use post spikes directly (spike time < inf means it spiked)
            update_locations = post_spike_times < float('inf')
        
        # Process each output channel
        for c in range(out_ch):
            # Get winner locations for this channel
            channel_winners = update_locations[:, c]  # (batch, H, W)
            
            if not channel_winners.any():
                continue
            
            # Get winner indices
            winner_indices = torch.where(channel_winners)
            
            for b, y, x in zip(*winner_indices):
                # Get post spike time for this winner
                post_time = post_spike_times[b, c, y, x].item()
                
                if post_time >= float('inf'):
                    continue  # No spike
                
                # Extract receptive field from pre_spike_times
                # Compute receptive field coordinates
                y_start = max(0, y - kH // 2)
                y_end = min(H_in, y + kH // 2 + 1)
                x_start = max(0, x - kW // 2)
                x_end = min(W_in, x + kW // 2 + 1)
                
                # Get pre spike times in receptive field
                rf_pre_times = pre_spike_times[b, :, y_start:y_end, x_start:x_end]
                
                # Pad if necessary to match kernel size
                if rf_pre_times.shape[1] < kH or rf_pre_times.shape[2] < kW:
                    pad_h = kH - rf_pre_times.shape[1]
                    pad_w = kW - rf_pre_times.shape[2]
                    rf_pre_times = torch.nn.functional.pad(
                        rf_pre_times, 
                        (0, pad_w, 0, pad_h), 
                        value=float('inf')
                    )
                
                # Compute STDP update for this winner
                # LTP: pre fires before/at post (pre_time <= post_time)
                ltp_mask = (rf_pre_times <= post_time).float()
                # LTD: pre fires after post (pre_time > post_time)
                ltd_mask = (rf_pre_times > post_time).float()
                
                # Multiplicative soft bounds: w * (1 - w)
                w = weights[c]
                soft_bounds = w * (1.0 - w)
                
                # Compute delta
                channel_delta = (
                    self.config.lr_plus * soft_bounds * ltp_mask -
                    self.config.lr_minus * soft_bounds * ltd_mask
                )
                
                delta_w[c] += channel_delta
                n_updates += 1
        
        # Average over number of updates
        if n_updates > 0:
            delta_w /= n_updates
        
        # Record statistics
        convergence = compute_convergence_metric(weights + delta_w)
        self.stats.record_update(delta_w, convergence)
        
        return delta_w
    
    def update_weights_for_winner(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_time: float,
        winner_y: int,
        winner_x: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        inplace: bool = True
    ) -> torch.Tensor:
        """
        Apply STDP update for a WTA winner neuron.
        
        This is the main learning function. When a neuron wins the WTA
        competition, this updates the weights of its feature map.
        
        Paper Reference:
            "The winner triggers the STDP and updates its synaptic weights. 
            As mentioned before, neurons in different locations of the same 
            map have the same input synaptic weights (i.e., weight sharing). 
            Hence, the winner neuron prevents other neurons in its own map 
            to do STDP and duplicates its updated synaptic weights into them."
        
        Args:
            weights: Full weight tensor, shape (out_ch, in_ch, kH, kW).
                    Only winner's feature map (weights[winner_map]) is updated.
            pre_spike_times: Pre-synaptic spike times, shape (in_ch, H_in, W_in).
            post_spike_time: Spike time of winner neuron.
            winner_y: Y coordinate of winner in output feature map.
            winner_x: X coordinate of winner in output feature map.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            inplace: If True, modify weights in place. Otherwise return copy.
        
        Returns:
            Updated weight tensor.
        
        Note:
            This updates weights for a SINGLE feature map. The winner_map
            index should be handled by the caller (from WTA).
        """
        _validate_tensor(weights, "weights")
        _validate_tensor(pre_spike_times, "pre_spike_times")
        
        if weights.dim() == 4:
            # Full weights tensor - caller should slice to winner's map
            raise STDPRuntimeError(
                "weights should be for single feature map (3D). "
                "Slice weights[winner_map] before calling."
            )
        
        _validate_tensor_dim(weights, 3, "weights")
        _validate_tensor_dim(pre_spike_times, 3, "pre_spike_times")
        
        # Extract receptive field spike times
        rf_times = extract_receptive_field_times(
            pre_spike_times=pre_spike_times,
            post_y=winner_y,
            post_x=winner_x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Compute STDP update
        delta_w = self.compute_update(
            weights=weights,
            pre_spike_times=rf_times,
            post_spike_time=post_spike_time
        )
        
        # Apply update
        if inplace:
            weights.add_(delta_w)
        else:
            weights = weights + delta_w
        
        # Clamp weights to [0, 1] (Kheradpisheh 2018: "all weights are bounded in [0, 1]")
        # This is required for BOTH additive and multiplicative variants
        # Multiplicative soft bounds reduce but don't eliminate drift
        weights.clamp_(self.config.weight_min, self.config.weight_max)
        
        # Record statistics
        convergence = compute_convergence_metric(weights)
        self.stats.record_update(delta_w, convergence)
        
        return weights
    
    def has_converged(self, weights: torch.Tensor) -> bool:
        """
        Check if weights have converged.
        
        Args:
            weights: Weight tensor to check.
        
        Returns:
            True if convergence metric < threshold.
        """
        return has_converged(weights, self.config.convergence_threshold)
    
    def get_convergence(self, weights: torch.Tensor) -> float:
        """
        Get current convergence metric.
        
        Args:
            weights: Weight tensor.
        
        Returns:
            Convergence metric value.
        """
        return compute_convergence_metric(weights)
    
    def reset_stats(self) -> None:
        """Reset learning statistics."""
        self.stats = STDPStats()
    
    @property
    def update_count(self) -> int:
        """Total number of weight updates performed."""
        return self.stats.n_updates
    
    @property
    def convergence_history(self) -> List[float]:
        """History of convergence metric values."""
        return self.stats.convergence_history
    
    def __repr__(self) -> str:
        return f"STDPLearner(config={self.config}, stats={self.stats})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def find_wta_winner(
    output_spikes: torch.Tensor,
    potentials: Optional[torch.Tensor] = None
) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[float]]:
    """
    Find the Winner-Take-All winner in output spikes.
    
    Paper Reference:
        "When a neuron fires, in a specific location, it inhibits other 
        neurons in that location belonging to other neuronal maps."
        
        "Because of the discretized time variable, it is probable that some 
        competitor neurons fire at the same time step. One possible scenario 
        is to pick one randomly. But a better alternative is to pick the one 
        which has the highest potential."
    
    Args:
        output_spikes: Spike tensor, shape (n_timesteps, out_channels, H, W).
        potentials: Optional membrane potentials for tie-breaking.
                   Shape: (out_channels, H, W).
    
    Returns:
        Tuple of (winner_map, (winner_y, winner_x), spike_time):
            - winner_map: Index of winning feature map (None if no spikes)
            - (winner_y, winner_x): Spatial location of winner
            - spike_time: Time of winning spike
    """
    _validate_tensor(output_spikes, "output_spikes")
    _validate_tensor_dim(output_spikes, 4, "output_spikes")
    
    n_timesteps = output_spikes.shape[0]
    
    # Find first spike across all neurons
    for t in range(n_timesteps):
        spike_locs = torch.nonzero(output_spikes[t], as_tuple=False)
        
        if len(spike_locs) == 0:
            continue
        
        if len(spike_locs) == 1:
            # Single winner
            c, y, x = spike_locs[0].tolist()
            return c, (y, x), float(t)
        
        # Multiple spikes at same time - use potentials for tie-breaking
        if potentials is not None:
            _validate_tensor(potentials, "potentials")
            
            # Find neuron with highest potential among those that spiked
            best_idx = 0
            best_potential = -float('inf')
            
            for i, loc in enumerate(spike_locs):
                c, y, x = loc.tolist()
                pot = potentials[c, y, x].item()
                if pot > best_potential:
                    best_potential = pot
                    best_idx = i
            
            c, y, x = spike_locs[best_idx].tolist()
            return c, (y, x), float(t)
        
        else:
            # Random selection (fallback)
            idx = torch.randint(len(spike_locs), (1,)).item()
            c, y, x = spike_locs[idx].tolist()
            return c, (y, x), float(t)
    
    # No spikes
    return None, None, None


def apply_lateral_inhibition(
    potentials: torch.Tensor,
    winner_map: int,
    winner_y: int,
    winner_x: int,
    inplace: bool = True
) -> torch.Tensor:
    """
    Apply lateral inhibition after a neuron fires.
    
    Paper Reference:
        "When a neuron fires, in a specific location, it inhibits other 
        neurons in that location belonging to other neuronal maps (i.e., 
        resets their potentials to zero) and does not allow them to fire 
        until the next image is shown."
    
    Args:
        potentials: Membrane potentials, shape (out_channels, H, W).
        winner_map: Feature map index of winner.
        winner_y: Y coordinate of winner.
        winner_x: X coordinate of winner.
        inplace: If True, modify potentials in place.
    
    Returns:
        Updated potentials with inhibited neurons reset to zero.
    """
    _validate_tensor(potentials, "potentials")
    _validate_tensor_dim(potentials, 3, "potentials")
    _validate_non_negative_int(winner_map, "winner_map")
    _validate_non_negative_int(winner_y, "winner_y")
    _validate_non_negative_int(winner_x, "winner_x")
    
    if not inplace:
        potentials = potentials.clone()
    
    # Reset all neurons at winner's location (across all feature maps)
    potentials[:, winner_y, winner_x] = 0.0
    
    return potentials


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "STDPError",
    "STDPConfigError",
    "STDPRuntimeError",
    # Configuration
    "STDPVariant",
    "STDPConfig",
    # Weight initialization
    "initialize_weights",
    # Convergence
    "compute_convergence_metric",
    "has_converged",
    # Spike timing
    "get_first_spike_times",
    "extract_receptive_field_times",
    # STDP update
    "compute_stdp_update",
    # Main learner
    "STDPStats",
    "STDPLearner",
    # WTA utilities
    "find_wta_winner",
    "apply_lateral_inhibition",
]