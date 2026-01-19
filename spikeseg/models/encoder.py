"""
SpikeSEG Encoder: Spiking Convolutional Neural Network for Feature Extraction.

This module implements the encoder portion of the SpikeSEG network,
which extracts hierarchical spatio-temporal features from event-based
sensor data using STDP-trained spiking convolutional layers.

Paper References:
    Kirkland et al. 2020 - "SpikeSEG: Spiking segmentation via STDP 
    saliency mapping"
    
    Kirkland et al. 2023 (IGARSS) - "Neuromorphic sensing and processing 
    for space domain awareness"

Architecture:
    Input → Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Classification
    
    Layer Details (IGARSS 2023):
        - Conv1: 5×5 kernel, 4 features, leak=90% of threshold
        - Pool1: 2×2 max pooling (stores indices)
        - Conv2: 5×5 kernel, 36 features, leak=10% of threshold
        - Pool2: 2×2 max pooling (stores indices)
        - Conv3: 7×7 kernel, n_classes features, leak=0%
    
    Key Features:
        - LIF neurons with layer-wise leak factors
        - WTA lateral inhibition for competitive learning
        - Returns pooling indices for decoder
        - STDP-compatible weight structure

Paper Quote (IGARSS 2023):
    "λ is set to 90% and 10% of the neuron threshold in layers 1 and 2 
    respectively, using a 5x5 convolution kernel and a 7x7 final 
    classification kernel with no leakage."

Example:
    >>> from spikeseg.models.encoder import SpikeSEGEncoder, EncoderConfig
    >>> 
    >>> # Create encoder with paper parameters
    >>> config = EncoderConfig.from_paper("igarss2023")
    >>> encoder = SpikeSEGEncoder(config)
    >>> 
    >>> # Forward pass
    >>> class_spikes, pool_indices = encoder(input_events, n_timesteps=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.neurons import LIFNeuron, IFNeuron, create_neuron
from ..core.layers import SpikingConv2d, SpikingPool2d


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class EncoderError(Exception):
    """Base exception for encoder module errors."""
    pass


class EncoderConfigError(EncoderError):
    """Raised for invalid encoder configuration."""
    pass


class EncoderRuntimeError(EncoderError):
    """Raised for runtime errors during encoding."""
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


def _validate_5d_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor is 5D with shape (T, N, C, H, W)."""
    _validate_tensor(tensor, name)
    if tensor.dim() != 5:
        raise ValueError(
            f"{name} must be 5D (T, N, C, H, W), got {tensor.dim()}D with shape {tensor.shape}"
        )


def _validate_positive_int(value: Any, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
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


# =============================================================================
# OUTPUT STRUCTURES
# =============================================================================


class PoolingIndices(NamedTuple):
    """
    Container for pooling indices from encoder.
    
    Used by decoder for unpooling operations.
    """
    pool1_indices: torch.Tensor
    pool2_indices: torch.Tensor
    pool1_output_size: Tuple[int, int, int, int]
    pool2_output_size: Tuple[int, int, int, int]


@dataclass
class EncoderOutput:
    """
    Complete output from encoder forward pass.

    Attributes:
        classification_spikes: Spikes from final layer (T, B, C, H, W).
                              T = timesteps, B = batch, C = classes,
                              H/W = spatial dimensions.
                              Use .sum(dim=0) to get total spikes per location.
        pooling_indices: Indices for decoder unpooling.
        layer_spikes: Dict of spikes at each layer (for HULK/ASH).
                     Each tensor has shape (T, B, C, H, W).
        layer_membranes: Dict of membrane potentials (optional).
        layer_spike_times: Dict of first spike times per layer (B, C, H, W).
                          Value of -1 means neuron never fired.
                          Critical for proper STDP learning.

    Properties:
        has_spikes: True if any classification spikes occurred.
        n_classification_spikes: Total count of classification spikes.

    Example:
        >>> output = encoder(events, n_timesteps=10)
        >>> if output.has_spikes:
        ...     print(f"Detected {output.n_classification_spikes} spikes")
        >>> # Get spikes summed over time
        >>> total_spikes = output.classification_spikes.sum(dim=0)  # (B, C, H, W)
        >>> # Get first spike times for STDP
        >>> conv2_times = output.layer_spike_times['conv2']  # (B, C, H, W)
    """
    classification_spikes: torch.Tensor
    pooling_indices: PoolingIndices
    layer_spikes: Dict[str, torch.Tensor]
    layer_membranes: Optional[Dict[str, torch.Tensor]] = None
    layer_spike_times: Optional[Dict[str, torch.Tensor]] = None
    
    @property
    def has_spikes(self) -> bool:
        """Check if any classification spikes occurred."""
        return self.classification_spikes.sum() > 0
    
    @property
    def n_classification_spikes(self) -> int:
        """Number of classification spikes."""
        return int(self.classification_spikes.sum().item())


# =============================================================================
# ENCODER CONFIGURATION
# =============================================================================


@dataclass
class LayerConfig:
    """Configuration for a single encoder layer."""
    out_channels: int
    kernel_size: int
    threshold: float = 10.0
    leak: float = 0.0
    leak_mode: str = "subtractive"
    
    def __post_init__(self) -> None:
        _validate_positive_int(self.out_channels, "out_channels")
        _validate_positive_int(self.kernel_size, "kernel_size")
        if self.threshold <= 0:
            raise EncoderConfigError(f"threshold must be positive, got {self.threshold}")
        _validate_non_negative_float(self.leak, "leak")
        if self.leak_mode not in ("subtractive", "multiplicative"):
            raise EncoderConfigError(
                f"leak_mode must be 'subtractive' or 'multiplicative', got '{self.leak_mode}'"
            )


@dataclass
class EncoderConfig:
    """
    Configuration for SpikeSEG encoder.
    
    Attributes:
        input_channels: Number of input channels (1 for events).
        conv1: Configuration for Conv1 layer.
        conv2: Configuration for Conv2 layer.
        conv3: Configuration for Conv3 (classification) layer.
        pool_kernel_size: Kernel size for max pooling.
        pool_stride: Stride for max pooling (default: same as kernel).
        use_wta: Enable Winner-Take-All lateral inhibition.
        wta_mode: WTA competition mode ('global', 'local', 'both').
        store_all_spikes: Store spikes at all layers (for HULK).
        store_membranes: Store membrane potentials (for analysis).
    
    Example:
        >>> config = EncoderConfig(
        ...     conv1=LayerConfig(out_channels=4, kernel_size=5, leak=9.0),
        ...     conv2=LayerConfig(out_channels=36, kernel_size=5, leak=1.0),
        ...     conv3=LayerConfig(out_channels=1, kernel_size=7, leak=0.0)
        ... )
    """
    input_channels: int = 1
    conv1: LayerConfig = field(default_factory=lambda: LayerConfig(
        out_channels=4, kernel_size=5, threshold=10.0, leak=9.0  # 90% of threshold
    ))
    conv2: LayerConfig = field(default_factory=lambda: LayerConfig(
        out_channels=36, kernel_size=5, threshold=10.0, leak=1.0  # 10% of threshold
    ))
    conv3: LayerConfig = field(default_factory=lambda: LayerConfig(
        out_channels=1, kernel_size=7, threshold=10.0, leak=0.0  # No leak
    ))
    # Pool1: 7×7 kernel, stride 6 (Kheradpisheh 2018 Table 1)
    pool1_kernel_size: int = 7
    pool1_stride: int = 6
    # Pool2: 2×2 kernel, stride 2 (Kheradpisheh 2018 Table 1)
    pool2_kernel_size: int = 2
    pool2_stride: int = 2
    # Legacy: for backward compatibility
    pool_kernel_size: int = 2  # Deprecated, use pool1/pool2 instead
    pool_stride: Optional[int] = None  # Deprecated
    use_wta: bool = True
    wta_mode: str = "both"  # Paper uses both global + local
    store_all_spikes: bool = True
    store_membranes: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        _validate_positive_int(self.input_channels, "input_channels")
        _validate_positive_int(self.pool1_kernel_size, "pool1_kernel_size")
        _validate_positive_int(self.pool1_stride, "pool1_stride")
        _validate_positive_int(self.pool2_kernel_size, "pool2_kernel_size")
        _validate_positive_int(self.pool2_stride, "pool2_stride")
        
        if self.wta_mode not in ("global", "local", "both"):
            raise EncoderConfigError(
                f"wta_mode must be 'global', 'local', or 'both', got '{self.wta_mode}'"
            )
    
    @classmethod
    def from_paper(cls, paper: str, n_classes: int = 1) -> "EncoderConfig":
        """
        Create configuration from paper parameters.
        
        Args:
            paper: Paper identifier. Options:
                - "igarss2023": Space domain awareness (EBSSA)
                - "spikeseg": Original SpikeSEG face detection
                - "default": Balanced default parameters
            n_classes: Number of output classes (Conv3 output channels).
        
        Returns:
            EncoderConfig with paper parameters.
        
        Raises:
            EncoderConfigError: If paper not recognized.
            ValueError: If n_classes < 1.
        """
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        
        # Base threshold for calculating leak as percentage
        threshold = 10.0
        
        configs = {
            "igarss2023": cls(
                input_channels=1,
                conv1=LayerConfig(
                    out_channels=4,
                    kernel_size=5,
                    threshold=threshold,
                    leak=0.9 * threshold,  # 90% of threshold
                    leak_mode="subtractive"
                ),
                conv2=LayerConfig(
                    out_channels=36,
                    kernel_size=5,
                    threshold=threshold,
                    leak=0.1 * threshold,  # 10% of threshold
                    leak_mode="subtractive"
                ),
                conv3=LayerConfig(
                    out_channels=n_classes,  # User-specified classes
                    kernel_size=7,
                    threshold=threshold,
                    leak=0.0,  # No leak
                    leak_mode="subtractive"
                ),
                # Kheradpisheh 2018 pooling configuration
                pool1_kernel_size=7,
                pool1_stride=6,
                pool2_kernel_size=2,
                pool2_stride=2,
                use_wta=True,
                wta_mode="both"  # Paper uses both global + local
            ),
            "spikeseg": cls(
                input_channels=1,
                conv1=LayerConfig(
                    out_channels=4,
                    kernel_size=5,
                    threshold=10.0,
                    leak=0.0,  # Original uses IF neurons
                    leak_mode="subtractive"
                ),
                conv2=LayerConfig(
                    out_channels=36,
                    kernel_size=5,
                    threshold=10.0,
                    leak=0.0,
                    leak_mode="subtractive"
                ),
                conv3=LayerConfig(
                    out_channels=n_classes,  # User-specified classes
                    kernel_size=7,
                    threshold=10.0,
                    leak=0.0,
                    leak_mode="subtractive"
                ),
                # Kheradpisheh 2018 pooling configuration
                pool1_kernel_size=7,
                pool1_stride=6,
                pool2_kernel_size=2,
                pool2_stride=2,
                use_wta=True,
                wta_mode="both"  # Paper uses both global + local
            ),
            "default": cls().with_n_classes(n_classes)  # Uses default values with n_classes
        }
        
        paper_lower = paper.lower()
        
        # Map aliases to canonical names
        aliases = {
            "hulk2022": "spikeseg",
            "hulksmash": "spikeseg",
            "hulk-smash": "spikeseg",
            "spikeseg2020": "spikeseg",
        }
        paper_lower = aliases.get(paper_lower, paper_lower)
        
        if paper_lower not in configs:
            raise EncoderConfigError(
                f"Unknown paper '{paper}'. Available: {list(configs.keys())} "
                f"(aliases: hulk2022, hulksmash, hulk-smash, spikeseg2020)"
            )
        
        return configs[paper_lower]
    
    def with_n_classes(self, n_classes: int) -> "EncoderConfig":
        """
        Return a new config with different number of output classes.
        
        Args:
            n_classes: Number of classification classes.
        
        Returns:
            New EncoderConfig with modified conv3.
        """
        _validate_positive_int(n_classes, "n_classes")
        
        new_conv3 = LayerConfig(
            out_channels=n_classes,
            kernel_size=self.conv3.kernel_size,
            threshold=self.conv3.threshold,
            leak=self.conv3.leak,
            leak_mode=self.conv3.leak_mode
        )
        
        return EncoderConfig(
            input_channels=self.input_channels,
            conv1=self.conv1,
            conv2=self.conv2,
            conv3=new_conv3,
            pool_kernel_size=self.pool_kernel_size,
            pool_stride=self.pool_stride,
            use_wta=self.use_wta,
            wta_mode=self.wta_mode,
            store_all_spikes=self.store_all_spikes,
            store_membranes=self.store_membranes
        )
    
    def __repr__(self) -> str:
        return (
            f"EncoderConfig(\n"
            f"  conv1: {self.conv1.out_channels}ch, k={self.conv1.kernel_size}, leak={self.conv1.leak}\n"
            f"  conv2: {self.conv2.out_channels}ch, k={self.conv2.kernel_size}, leak={self.conv2.leak}\n"
            f"  conv3: {self.conv3.out_channels}ch, k={self.conv3.kernel_size}, leak={self.conv3.leak}\n"
            f"  pool: {self.pool_kernel_size}x{self.pool_kernel_size}, wta={self.use_wta}\n"
            f")"
        )


# =============================================================================
# SPIKING ENCODER LAYER
# =============================================================================


class SpikingEncoderLayer(nn.Module):
    """
    Single spiking encoder layer: Conv → LIF → (optional WTA).
    
    Wraps SpikingConv2d with LIF neuron dynamics and optional
    lateral inhibition.
    
    Attributes:
        conv: Spiking convolution layer.
        neuron: LIF or IF neuron model.
        membrane: Current membrane potential.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        threshold: float = 10.0,
        leak: float = 0.0,
        leak_mode: str = "subtractive",
        padding: Optional[int] = None
    ) -> None:
        """
        Initialize spiking encoder layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (features).
            kernel_size: Convolution kernel size.
            threshold: Spike threshold.
            leak: Leak factor (0 for IF neuron behavior).
            leak_mode: 'subtractive' or 'multiplicative'.
            padding: Convolution padding (default: kernel_size // 2).
        """
        super().__init__()
        
        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")
        _validate_positive_int(kernel_size, "kernel_size")
        
        if padding is None:
            padding = kernel_size // 2
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.leak = leak
        
        # Convolution (no bias for biological plausibility)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
        # Initialize weights for STDP compatibility (0.8 ± 0.05)
        # Per Kheradpisheh 2018, std=0.05 provides sufficient diversity for feature learning
        nn.init.normal_(self.conv.weight, mean=0.8, std=0.05)
        self.conv.weight.data.clamp_(0, 1)
        
        # Spiking neuron
        # Convert absolute leak value to fraction for LIFNeuron
        # Paper (IGARSS 2023): "λ is set to 90% and 10% of the neuron threshold"
        # LayerConfig stores leak as absolute value (e.g., 9.0 = 90% of threshold 10.0)
        # LIFNeuron expects leak_factor as fraction in [0, 1]
        leak_factor = leak / threshold if threshold > 0 else 0.0

        self.neuron = LIFNeuron(
            threshold=threshold,
            leak_factor=leak_factor,
            leak_mode=leak_mode
        )
        
        # State
        self.register_buffer('membrane', None)
        self.register_buffer('has_fired', None)  # Fire-once tracking (Kheradpisheh 2018)
        self.register_buffer('pre_reset_membrane', None)  # For WTA tie-breaking

    def reset_state(self) -> None:
        """Reset membrane potential and fire-once tracking."""
        self.membrane = None
        self.has_fired = None  # Reset fire-once mask for new stimulus
        self.pre_reset_membrane = None
        # Note: LIFNeuron is stateless - membrane is managed here
    
    def forward(
        self, 
        x: torch.Tensor,
        dynamic_threshold: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single timestep.
        
        Args:
            x: Input tensor (spikes), shape (batch, in_channels, H, W).
            dynamic_threshold: Optional per-channel thresholds for homeostasis.
                              Shape: (out_channels,). If provided, overrides the
                              neuron's fixed threshold for adaptive firing control.
                              (Diehl & Cook 2015, Lee et al. 2018)
        
        Returns:
            Tuple of (output_spikes, membrane_potential).
        """
        _validate_4d_tensor(x, "input")
        
        if x.shape[1] != self.in_channels:
            raise EncoderRuntimeError(
                f"Input channels ({x.shape[1]}) doesn't match "
                f"expected ({self.in_channels})"
            )
        
        # Convolution
        current = self.conv(x)

        # Initialize membrane if needed
        if self.membrane is None or self.membrane.shape != current.shape:
            self.membrane = torch.zeros_like(current)

        # Initialize fire-once tracking if needed (Kheradpisheh 2018)
        if self.has_fired is None or self.has_fired.shape != current.shape:
            self.has_fired = torch.zeros_like(current)

        # LIF dynamics with fire-once constraint and optional adaptive threshold
        # (Kheradpisheh 2018: "neurons are not allowed to fire more than once per stimulus")
        # (Diehl & Cook 2015: adaptive thresholds for homeostasis)
        spikes, self.membrane, self.pre_reset_membrane = self.neuron(
            current, self.membrane, self.has_fired, dynamic_threshold
        )

        # Update fire-once mask
        self.has_fired = torch.clamp(self.has_fired + spikes, 0.0, 1.0)

        return spikes, self.membrane
    
    @property
    def weight(self) -> torch.Tensor:
        """Get convolution weights."""
        return self.conv.weight

    def get_pre_reset_membrane(self) -> Optional[torch.Tensor]:
        """
        Get membrane potential before reset from last forward pass.

        Required for correct WTA tie-breaking (Kheradpisheh 2018):
        "pick the one which has the highest potential"

        Returns:
            Pre-reset membrane tensor, or None if no forward pass yet.
        """
        return self.pre_reset_membrane

    def __repr__(self) -> str:
        return (
            f"SpikingEncoderLayer({self.in_channels} → {self.out_channels}, "
            f"k={self.kernel_size}, leak={self.leak})"
        )


# =============================================================================
# SPIKESEG ENCODER
# =============================================================================


class SpikeSEGEncoder(nn.Module):
    """
    SpikeSEG Encoder: Hierarchical spiking feature extraction.
    
    Processes event-based input through three spiking convolutional
    layers with max pooling, producing classification spikes and
    storing pooling indices for the decoder.
    
    Architecture:
        Input → Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Classification
    
    Paper Reference (IGARSS 2023):
        "We have transitioned from the Integrate and Fire (IF) neuron model 
        to a set of Leak Integrate-and-Fire (LIF) neurons, with layer-wise 
        leak variation... λ is set to 90% and 10% of the neuron threshold 
        in layers 1 and 2 respectively."
    
    Attributes:
        conv1: First spiking conv layer (4 features).
        pool1: First max pooling layer.
        conv2: Second spiking conv layer (36 features).
        pool2: Second max pooling layer.
        conv3: Classification spiking conv layer.
        config: Encoder configuration.
    
    Example:
        >>> config = EncoderConfig.from_paper("igarss2023")
        >>> encoder = SpikeSEGEncoder(config)
        >>> 
        >>> # Process event sequence
        >>> output = encoder(events, n_timesteps=10)
        >>> class_spikes = output.classification_spikes  # (T, B, C, H, W)
        >>> pool_indices = output.pooling_indices
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        """
        Initialize SpikeSEG encoder.
        
        Args:
            config: Encoder configuration. If None, uses default.
        
        Raises:
            EncoderConfigError: If configuration is invalid.
        """
        super().__init__()
        
        if config is None:
            config = EncoderConfig()
        
        if not isinstance(config, EncoderConfig):
            raise EncoderConfigError(
                f"config must be EncoderConfig, got {type(config).__name__}"
            )
        
        self.config = config
        
        # Build layers
        
        # Conv1: Input → 4 features
        self.conv1 = SpikingEncoderLayer(
            in_channels=config.input_channels,
            out_channels=config.conv1.out_channels,
            kernel_size=config.conv1.kernel_size,
            threshold=config.conv1.threshold,
            leak=config.conv1.leak,
            leak_mode=config.conv1.leak_mode
        )
        
        # Pool1: 7×7 max pooling, stride 6 (Kheradpisheh 2018 Table 1)
        self.pool1 = nn.MaxPool2d(
            kernel_size=config.pool1_kernel_size,
            stride=config.pool1_stride,
            return_indices=True
        )

        # Conv2: 4 → 36 features
        self.conv2 = SpikingEncoderLayer(
            in_channels=config.conv1.out_channels,
            out_channels=config.conv2.out_channels,
            kernel_size=config.conv2.kernel_size,
            threshold=config.conv2.threshold,
            leak=config.conv2.leak,
            leak_mode=config.conv2.leak_mode
        )

        # Pool2: 2×2 max pooling, stride 2 (Kheradpisheh 2018 Table 1)
        self.pool2 = nn.MaxPool2d(
            kernel_size=config.pool2_kernel_size,
            stride=config.pool2_stride,
            return_indices=True
        )
        
        # Conv3: 36 → n_classes (classification)
        self.conv3 = SpikingEncoderLayer(
            in_channels=config.conv2.out_channels,
            out_channels=config.conv3.out_channels,
            kernel_size=config.conv3.kernel_size,
            threshold=config.conv3.threshold,
            leak=config.conv3.leak,
            leak_mode=config.conv3.leak_mode
        )
        
        # State storage for decoder
        self._pool1_indices: Optional[torch.Tensor] = None
        self._pool2_indices: Optional[torch.Tensor] = None
        self._pool1_input_size: Optional[Tuple[int, ...]] = None
        self._pool2_input_size: Optional[Tuple[int, ...]] = None
    
    def reset_state(self) -> None:
        """Reset all layer states."""
        self.conv1.reset_state()
        self.conv2.reset_state()
        self.conv3.reset_state()
        
        self._pool1_indices = None
        self._pool2_indices = None
        self._pool1_input_size = None
        self._pool2_input_size = None
    
    def forward_single_timestep(
        self,
        x: torch.Tensor,
        layer_thresholds: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for a single timestep.
        
        Args:
            x: Input events/spikes, shape (batch, channels, H, W).
            layer_thresholds: Optional dict mapping layer names ('conv1', 'conv2', 'conv3')
                             to per-channel threshold tensors (shape: (out_channels,)).
                             Used for adaptive homeostasis during STDP training.
                             (Diehl & Cook 2015, Lee et al. 2018)
        
        Returns:
            Tuple of (classification_spikes, layer_spikes_dict).
        """
        _validate_4d_tensor(x, "input")
        
        layer_spikes = {}
        
        # Get thresholds for each layer (None if not provided)
        thresh_conv1 = layer_thresholds.get('conv1') if layer_thresholds else None
        thresh_conv2 = layer_thresholds.get('conv2') if layer_thresholds else None
        thresh_conv3 = layer_thresholds.get('conv3') if layer_thresholds else None
        
        # Conv1
        spikes1, mem1 = self.conv1(x, thresh_conv1)
        layer_spikes['conv1'] = spikes1
        
        # Pool1 (store indices and input size for decoder)
        self._pool1_input_size = tuple(spikes1.shape)
        pooled1, self._pool1_indices = self.pool1(spikes1)
        layer_spikes['pool1'] = pooled1
        
        # Conv2
        spikes2, mem2 = self.conv2(pooled1, thresh_conv2)
        layer_spikes['conv2'] = spikes2
        
        # Pool2 (store indices and input size for decoder)
        self._pool2_input_size = tuple(spikes2.shape)
        pooled2, self._pool2_indices = self.pool2(spikes2)
        layer_spikes['pool2'] = pooled2
        
        # Conv3 (classification)
        spikes3, mem3 = self.conv3(pooled2, thresh_conv3)
        layer_spikes['conv3'] = spikes3
        
        return spikes3, layer_spikes
    
    def forward(
        self,
        x: torch.Tensor,
        n_timesteps: Optional[int] = None,
        reset_state: bool = True,
        layer_thresholds: Optional[Dict[str, torch.Tensor]] = None
    ) -> EncoderOutput:
        """
        Forward pass for event sequence.
        
        Args:
            x: Input tensor. Can be:
               - 4D (batch, channels, H, W): Single timestep
               - 5D (timesteps, batch, channels, H, W): Sequence
            n_timesteps: Number of timesteps (required if x is 4D and 
                        should be processed multiple times).
            reset_state: Reset membrane potentials before processing.
            layer_thresholds: Optional dict mapping layer names ('conv1', 'conv2', 'conv3')
                             to per-channel threshold tensors (shape: (out_channels,)).
                             Used for adaptive homeostasis during STDP training.
                             
                             Example:
                                 layer_thresholds = {
                                     'conv2': threshold_manager.get_threshold(),  # (36,)
                                     'conv3': threshold_manager.get_threshold(),  # (n_classes,)
                                 }
                             
                             Paper Reference (Diehl & Cook 2015, Lee et al. 2018):
                                 "On spike: increase threshold of entire feature map"
                                 "On non-firing: exponentially decay toward rest"
        
        Returns:
            EncoderOutput containing classification spikes and pooling indices.
        """
        if reset_state:
            self.reset_state()
        
        # Handle input dimensions
        if x.dim() == 4:
            # Single frame - process once or n_timesteps times
            if n_timesteps is None:
                n_timesteps = 1
            
            # Replicate for each timestep (constant input)
            x = x.unsqueeze(0).expand(n_timesteps, -1, -1, -1, -1)
        
        elif x.dim() == 5:
            n_timesteps = x.shape[0]
        
        else:
            raise EncoderRuntimeError(
                f"Input must be 4D (B, C, H, W) or 5D (T, B, C, H, W), "
                f"got {x.dim()}D"
            )
        
        # Process each timestep
        all_class_spikes = []
        all_layer_spikes: Dict[str, List[torch.Tensor]] = {
            'conv1': [], 'pool1': [], 'conv2': [], 'pool2': [], 'conv3': []
        }

        # Track first spike times for STDP (initialized to -1 = never fired)
        # Shape will be (B, C, H, W) for each layer
        layer_spike_times: Dict[str, torch.Tensor] = {}

        for t in range(n_timesteps):
            class_spikes_t, layer_spikes_t = self.forward_single_timestep(
                x[t], layer_thresholds
            )
            all_class_spikes.append(class_spikes_t)

            # Track first spike times for each layer
            for name, spikes in layer_spikes_t.items():
                if name not in layer_spike_times:
                    # Initialize with -1 (never fired)
                    layer_spike_times[name] = torch.full_like(
                        spikes, fill_value=-1.0
                    )

                # Update spike times: record time t for neurons that fire
                # for the first time (current time is -1 and now spiking)
                first_spike_mask = (layer_spike_times[name] < 0) & (spikes > 0)
                layer_spike_times[name] = torch.where(
                    first_spike_mask,
                    torch.full_like(layer_spike_times[name], fill_value=float(t)),
                    layer_spike_times[name]
                )

            if self.config.store_all_spikes:
                for name, spikes in layer_spikes_t.items():
                    all_layer_spikes[name].append(spikes)

        # Stack outputs: (T, B, C, H, W)
        classification_spikes = torch.stack(all_class_spikes, dim=0)

        # Stack layer spikes if stored
        layer_spikes_stacked = {}
        if self.config.store_all_spikes:
            for name, spikes_list in all_layer_spikes.items():
                layer_spikes_stacked[name] = torch.stack(spikes_list, dim=0)

        # Create pooling indices container
        pooling_indices = PoolingIndices(
            pool1_indices=self._pool1_indices,
            pool2_indices=self._pool2_indices,
            pool1_output_size=self._pool1_input_size,
            pool2_output_size=self._pool2_input_size
        )

        return EncoderOutput(
            classification_spikes=classification_spikes,
            pooling_indices=pooling_indices,
            layer_spikes=layer_spikes_stacked,
            layer_spike_times=layer_spike_times
        )
    
    def get_pooling_indices(self) -> PoolingIndices:
        """
        Get stored pooling indices from last forward pass.
        
        Returns:
            PoolingIndices for decoder.
        
        Raises:
            EncoderRuntimeError: If no forward pass has been done.
        """
        if self._pool1_indices is None or self._pool2_indices is None:
            raise EncoderRuntimeError(
                "No pooling indices available. Run forward() first."
            )
        
        return PoolingIndices(
            pool1_indices=self._pool1_indices,
            pool2_indices=self._pool2_indices,
            pool1_output_size=self._pool1_input_size,
            pool2_output_size=self._pool2_input_size
        )
    
    def get_feature_counts(self) -> Dict[str, int]:
        """
        Get number of features at each layer.
        
        Useful for ASH (Active Spike Hash) computation.
        
        Returns:
            Dict mapping layer name to feature count.
        """
        return {
            'conv1': self.config.conv1.out_channels,
            'conv2': self.config.conv2.out_channels,
            'conv3': self.config.conv3.out_channels,
            'total': (
                self.config.conv1.out_channels +
                self.config.conv2.out_channels +
                self.config.conv3.out_channels
            )
        }
    
    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self.config.conv3.out_channels
    
    @property
    def kernel_sizes(self) -> Tuple[int, int, int]:
        """Kernel sizes for each conv layer."""
        return (
            self.config.conv1.kernel_size,
            self.config.conv2.kernel_size,
            self.config.conv3.kernel_size
        )
    
    def __repr__(self) -> str:
        features = self.get_feature_counts()
        return (
            f"SpikeSEGEncoder(\n"
            f"  input: {self.config.input_channels}ch\n"
            f"  conv1: {features['conv1']}ch (leak={self.config.conv1.leak})\n"
            f"  conv2: {features['conv2']}ch (leak={self.config.conv2.leak})\n"
            f"  conv3: {features['conv3']}ch (leak={self.config.conv3.leak})\n"
            f"  total features: {features['total']}\n"
            f")"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_encoder(
    paper: str = "igarss2023",
    n_classes: int = 1,
    **kwargs
) -> SpikeSEGEncoder:
    """
    Factory function to create encoder from paper configuration.
    
    Args:
        paper: Paper identifier ('igarss2023', 'spikeseg', 'default').
        n_classes: Number of output classes.
        **kwargs: Additional config overrides.
    
    Returns:
        Configured SpikeSEGEncoder.
    
    Example:
        >>> encoder = create_encoder("igarss2023", n_classes=5)
    """
    config = EncoderConfig.from_paper(paper)
    
    if n_classes != config.conv3.out_channels:
        config = config.with_n_classes(n_classes)
    
    return SpikeSEGEncoder(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "EncoderError",
    "EncoderConfigError",
    "EncoderRuntimeError",
    # Output structures
    "PoolingIndices",
    "EncoderOutput",
    # Configuration
    "LayerConfig",
    "EncoderConfig",
    # Layers
    "SpikingEncoderLayer",
    # Main encoder
    "SpikeSEGEncoder",
    # Factory
    "create_encoder",
]