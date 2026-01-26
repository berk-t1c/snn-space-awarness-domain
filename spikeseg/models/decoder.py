"""
SpikeSEG Decoder: Spiking Encoder-Decoder for Semantic Segmentation.

This module implements the decoder portion of the SpikeSEG network,
which maps classification spikes back to the original pixel space
to produce saliency-mapped segmentation outputs.

Paper References:
    Kirkland et al. 2020 - "SpikeSEG: Spiking segmentation via STDP 
    saliency mapping"
    
    Kirkland et al. 2022 - "Unsupervised Spiking Instance Segmentation 
    on Event Data using STDP Features" (HULK-SMASH)

Architecture:
    Classification → UnPool2 → TransConv2 → UnPool1 → TransConv1 → Output
    
    Key Features:
        - Tied weights: Decoder uses same weights as encoder
        - Pooling indices: Uses max pooling indices from encoder
        - Delay connections: Maintains temporal continuity of spikes
        - Saliency mapping: Output shows which pixels contributed to classification

Paper Quote:
    "The decoder side of SpikeSEG is then attached to the encoder with 
    the same trained weights as the encoder and decoder layers are mirrored. 
    Utilising max pooling indices and a temporal delay signal (to maintain 
    temporal continuity of the spikes) the classification spikes are then 
    transformed via the transposed convolutions and unpooling layers back 
    into the pixel domain."

Example:
    >>> from spikeseg.models.decoder import SpikeSEGDecoder
    >>> 
    >>> # Create decoder from trained encoder
    >>> decoder = SpikeSEGDecoder.from_encoder(encoder)
    >>> 
    >>> # Forward pass (uses stored pooling indices)
    >>> segmentation = decoder(
    ...     classification_spikes,
    ...     pool1_indices, pool2_indices,
    ...     pool1_output_size, pool2_output_size
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.neurons import LIFNeuron, create_neuron


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class DecoderError(Exception):
    """Base exception for decoder module errors."""
    pass


class DecoderConfigError(DecoderError):
    """Raised for invalid decoder configuration."""
    pass


class DecoderRuntimeError(DecoderError):
    """Raised for runtime errors during decoding."""
    pass


class EncoderCompatibilityError(DecoderError):
    """Raised when encoder is incompatible with decoder."""
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


def _validate_tuple_of_ints(value: Any, length: int, name: str) -> None:
    """Validate that value is a tuple of integers with specified length."""
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {len(value)}")
    for i, v in enumerate(value):
        if not isinstance(v, int):
            raise TypeError(f"{name}[{i}] must be an integer, got {type(v).__name__}")


def _validate_same_device(t1: torch.Tensor, t2: torch.Tensor,
                          name1: str, name2: str) -> None:
    """Validate two tensors are on the same device."""
    if t1.device != t2.device:
        raise ValueError(
            f"{name1} and {name2} must be on same device. "
            f"Got {t1.device} and {t2.device}"
        )


# =============================================================================
# DECODER CONFIGURATION
# =============================================================================


@dataclass
class DecoderConfig:
    """
    Configuration for SpikeSEG decoder.
    
    Attributes:
        use_tied_weights: Use encoder weights for transposed convolutions.
        use_spiking: Use spiking neurons in decoder (vs direct activation).
        threshold: Spike threshold for decoder neurons.
        leak: Leak factor for decoder LIF neurons.
        use_delay_connections: Enable temporal delay for spike continuity.
        delay_steps: Number of timesteps to delay signals.
    
    Example:
        >>> config = DecoderConfig(use_tied_weights=True, use_spiking=True)
    """
    use_tied_weights: bool = True
    use_spiking: bool = True
    threshold: float = 1.0
    leak: float = 0.0  # No leak by default for decoder
    use_delay_connections: bool = True
    delay_steps: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.use_tied_weights, bool):
            raise DecoderConfigError(
                f"use_tied_weights must be bool, got {type(self.use_tied_weights).__name__}"
            )
        if not isinstance(self.use_spiking, bool):
            raise DecoderConfigError(
                f"use_spiking must be bool, got {type(self.use_spiking).__name__}"
            )
        if self.threshold <= 0:
            raise DecoderConfigError(f"threshold must be positive, got {self.threshold}")
        if self.leak < 0:
            raise DecoderConfigError(f"leak must be non-negative, got {self.leak}")
        if self.delay_steps < 0:
            raise DecoderConfigError(f"delay_steps must be non-negative, got {self.delay_steps}")


# =============================================================================
# TRANSPOSED CONVOLUTION LAYER (FOR DECODER)
# =============================================================================


class DecoderTransConv2d(nn.Module):
    """
    Transposed convolution layer for decoder with optional spiking.
    
    Can use tied weights from encoder or learn separate weights.
    
    Attributes:
        trans_conv: Transposed convolution operation.
        neuron: Optional spiking neuron.
        membrane: Current membrane potential.
    
    Example:
        >>> layer = DecoderTransConv2d(
        ...     in_channels=36, out_channels=4,
        ...     kernel_size=5, encoder_weight=conv1.weight
        ... )
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: Optional[int] = None,
        encoder_weight: Optional[torch.Tensor] = None,
        use_spiking: bool = True,
        threshold: float = 1.0,
        leak: float = 0.0
    ) -> None:
        """
        Initialize decoder transposed convolution.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Padding (default: kernel_size // 2 for same output).
            encoder_weight: Tied weights from encoder (optional).
            use_spiking: Use spiking neurons.
            threshold: Spike threshold.
            leak: Leak factor.
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
        self.stride = stride
        self.padding = padding
        self.use_spiking = use_spiking
        
        # Transposed convolution
        # Note: For tied weights, we use the encoder weight directly
        # ConvTranspose2d with weight from Conv2d: swap in/out channels
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Set tied weights if provided
        if encoder_weight is not None:
            _validate_tensor(encoder_weight, "encoder_weight")
            # Encoder weight shape: (out_ch, in_ch, kH, kW)
            # For transposed conv with tied weights, we need (in_ch, out_ch, kH, kW)
            # But PyTorch ConvTranspose2d expects (in_channels, out_channels, kH, kW)
            # where in_channels is the input to transposed conv
            
            # Validate shapes match
            enc_out, enc_in = encoder_weight.shape[:2]
            if enc_out != in_channels or enc_in != out_channels:
                raise DecoderConfigError(
                    f"encoder_weight shape ({enc_out}, {enc_in}, ...) incompatible with "
                    f"trans_conv (in={in_channels}, out={out_channels}). "
                    f"Expected encoder shape ({in_channels}, {out_channels}, ...)"
                )
            
            # Use encoder weights directly (tied)
            with torch.no_grad():
                self.trans_conv.weight.copy_(encoder_weight)
        
        # Spiking neuron (optional)
        self.neuron: Optional[LIFNeuron] = None
        if use_spiking:
            self.neuron = LIFNeuron(
                threshold=threshold,
                leak_factor=leak,
                leak_mode="subtractive"
            )
        
        # Membrane potential buffer
        self.register_buffer('membrane', None)
    
    def reset_state(self) -> None:
        """Reset membrane potential."""
        self.membrane = None
        # Note: LIFNeuron is stateless - membrane is managed by self.membrane
        # No need to call neuron.reset_state() as it requires shape/device params
    
    def forward(self, x: torch.Tensor, output_size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, in_channels, H, W).
            output_size: Optional output size for transposed conv.
        
        Returns:
            Output tensor (spikes if spiking, activations otherwise).
        """
        _validate_4d_tensor(x, "input")
        
        if x.shape[1] != self.in_channels:
            raise DecoderRuntimeError(
                f"Input channels ({x.shape[1]}) doesn't match "
                f"expected ({self.in_channels})"
            )
        
        # Transposed convolution
        if output_size is not None:
            out = self.trans_conv(x, output_size=output_size)
        else:
            out = self.trans_conv(x)
        
        # Apply spiking neuron if enabled
        if self.neuron is not None:
            # Initialize membrane if needed
            if self.membrane is None or self.membrane.shape != out.shape:
                self.membrane = torch.zeros_like(out)
            
            spikes, self.membrane, _ = self.neuron(out, self.membrane)
            return spikes
        
        return out
    
    def __repr__(self) -> str:
        return (
            f"DecoderTransConv2d({self.in_channels} → {self.out_channels}, "
            f"k={self.kernel_size}, spiking={self.use_spiking})"
        )


# =============================================================================
# UNPOOLING LAYER (FOR DECODER)
# =============================================================================


class DecoderUnpool2d(nn.Module):
    """
    Max unpooling layer for decoder.
    
    Uses pooling indices from encoder to place values back
    in their original positions.
    
    Paper Quote:
        "Utilising max pooling indices..."
    """
    
    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        padding: int = 0
    ) -> None:
        """
        Initialize unpooling layer.
        
        Args:
            kernel_size: Pooling kernel size.
            stride: Pooling stride (default: kernel_size).
            padding: Padding.
        """
        super().__init__()
        
        _validate_positive_int(kernel_size, "kernel_size")
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
        self.unpool = nn.MaxUnpool2d(
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        indices: torch.Tensor,
        output_size: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, H, W).
            indices: Pooling indices from encoder.
            output_size: Optional output size (should match encoder input to pool).
        
        Returns:
            Unpooled tensor.
        """
        _validate_4d_tensor(x, "input")
        _validate_4d_tensor(indices, "indices")
        
        if x.shape != indices.shape:
            raise DecoderRuntimeError(
                f"Input shape {x.shape} doesn't match indices shape {indices.shape}"
            )
        
        return self.unpool(x, indices, output_size=output_size)
    
    def __repr__(self) -> str:
        return f"DecoderUnpool2d(kernel_size={self.kernel_size}, stride={self.stride})"


# =============================================================================
# DELAY CONNECTION
# =============================================================================


class DelayConnection(nn.Module):
    """
    Temporal delay connection for maintaining spike continuity.
    
    Stores spikes from previous timesteps and adds them to current
    input, allowing temporal information to propagate through the decoder.
    
    Paper Quote:
        "a temporal delay signal (to maintain temporal continuity of the spikes)"
    """
    
    def __init__(self, delay_steps: int = 1, decay: float = 0.9) -> None:
        """
        Initialize delay connection.
        
        Args:
            delay_steps: Number of timesteps to store.
            decay: Decay factor for delayed signals.
        """
        super().__init__()
        
        if delay_steps < 0:
            raise DecoderConfigError(f"delay_steps must be non-negative, got {delay_steps}")
        if not 0 <= decay <= 1:
            raise DecoderConfigError(f"decay must be in [0, 1], got {decay}")
        
        self.delay_steps = delay_steps
        self.decay = decay
        
        # Buffer for delayed signals
        self._buffer: List[torch.Tensor] = []
    
    def reset_state(self) -> None:
        """Clear delay buffer."""
        self._buffer = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply delay connection.
        
        Args:
            x: Current input tensor.
        
        Returns:
            Input combined with delayed signals.
        """
        _validate_tensor(x, "input")
        
        # Add delayed signals to current input
        output = x.clone()
        
        for i, delayed in enumerate(self._buffer):
            # Apply decay based on how old the signal is
            weight = self.decay ** (len(self._buffer) - i)
            output = output + weight * delayed.to(x.device)
        
        # Store current input for future
        self._buffer.append(x.detach().clone())
        
        # Keep only delay_steps entries
        if len(self._buffer) > self.delay_steps:
            self._buffer = self._buffer[-self.delay_steps:]
        
        return output
    
    def __repr__(self) -> str:
        return f"DelayConnection(delay_steps={self.delay_steps}, decay={self.decay})"


# =============================================================================
# SPIKESEG DECODER
# =============================================================================


class SpikeSEGDecoder(nn.Module):
    """
    SpikeSEG Decoder: Maps classification spikes back to pixel space.
    
    The decoder mirrors the encoder architecture with:
        - Transposed convolutions (tied weights from encoder)
        - Max unpooling (using stored indices)
        - Optional spiking neurons
        - Delay connections for temporal continuity
    
    Architecture:
        Classification → UnPool2 → TransConv2 → UnPool1 → TransConv1 → Output
    
    Paper Reference:
        "The decoder side of SpikeSEG is then attached to the encoder with 
        the same trained weights as the encoder and decoder layers are mirrored."
    
    Attributes:
        trans_conv1: First transposed convolution (mirrors Conv1).
        trans_conv2: Second transposed convolution (mirrors Conv2).
        unpool1: First unpooling layer.
        unpool2: Second unpooling layer.
        delay: Optional delay connection.
    
    Example:
        >>> decoder = SpikeSEGDecoder(
        ...     n_classes=1,
        ...     conv2_channels=36,
        ...     conv1_channels=4,
        ...     input_channels=1,
        ...     kernel_sizes=(5, 5, 7)
        ... )
        >>> 
        >>> # Forward pass
        >>> output = decoder(
        ...     class_spikes,  # (B, n_classes, H3, W3)
        ...     pool1_indices, pool2_indices,
        ...     pool1_output_size, pool2_output_size
        ... )
    """
    
    def __init__(
        self,
        n_classes: int,
        conv2_channels: int,
        conv1_channels: int,
        input_channels: int = 1,
        kernel_sizes: Tuple[int, int, int] = (5, 5, 7),
        pool_kernel_size: int = 2,
        config: Optional[DecoderConfig] = None,
        encoder_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Initialize SpikeSEG decoder.
        
        Args:
            n_classes: Number of classification channels.
            conv2_channels: Channels in Conv2 (input to decoder trans_conv2).
            conv1_channels: Channels in Conv1 (input to decoder trans_conv1).
            input_channels: Original input channels (output of decoder).
            kernel_sizes: Kernel sizes for (Conv1, Conv2, Conv3).
            pool_kernel_size: Pooling kernel size.
            config: Decoder configuration.
            encoder_weights: Dict with 'conv1', 'conv2', 'conv3' weights.
        
        Raises:
            DecoderConfigError: If configuration is invalid.
        """
        super().__init__()
        
        # Validate inputs
        _validate_positive_int(n_classes, "n_classes")
        _validate_positive_int(conv2_channels, "conv2_channels")
        _validate_positive_int(conv1_channels, "conv1_channels")
        _validate_positive_int(input_channels, "input_channels")
        _validate_positive_int(pool_kernel_size, "pool_kernel_size")
        _validate_tuple_of_ints(kernel_sizes, 3, "kernel_sizes")
        
        if config is None:
            config = DecoderConfig()
        
        self.config = config
        self.n_classes = n_classes
        self.conv2_channels = conv2_channels
        self.conv1_channels = conv1_channels
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        
        # Extract kernel sizes
        k1, k2, k3 = kernel_sizes
        
        # Get encoder weights if provided
        conv1_weight = None
        conv2_weight = None
        conv3_weight = None
        
        if encoder_weights is not None:
            conv1_weight = encoder_weights.get('conv1')
            conv2_weight = encoder_weights.get('conv2')
            conv3_weight = encoder_weights.get('conv3')
        
        # Build decoder layers (reverse order of encoder)
        
        # TransConv from classification → Conv2 space
        # Encoder Conv3: (n_classes, conv2_channels, k3, k3)
        # Decoder: (n_classes, conv2_channels, k3, k3) transposed
        self.trans_conv3 = DecoderTransConv2d(
            in_channels=n_classes,
            out_channels=conv2_channels,
            kernel_size=k3,
            encoder_weight=conv3_weight,
            use_spiking=config.use_spiking,
            threshold=config.threshold,
            leak=config.leak
        )
        
        # Unpool2: reverse Pool2
        self.unpool2 = DecoderUnpool2d(
            kernel_size=pool_kernel_size
        )
        
        # TransConv2: Conv2 → Conv1 space
        # Encoder Conv2: (conv2_channels, conv1_channels, k2, k2)
        self.trans_conv2 = DecoderTransConv2d(
            in_channels=conv2_channels,
            out_channels=conv1_channels,
            kernel_size=k2,
            encoder_weight=conv2_weight,
            use_spiking=config.use_spiking,
            threshold=config.threshold,
            leak=config.leak
        )
        
        # Unpool1: reverse Pool1
        self.unpool1 = DecoderUnpool2d(
            kernel_size=pool_kernel_size
        )
        
        # TransConv1: Conv1 → Input space
        # Encoder Conv1: (conv1_channels, input_channels, k1, k1)
        self.trans_conv1 = DecoderTransConv2d(
            in_channels=conv1_channels,
            out_channels=input_channels,
            kernel_size=k1,
            encoder_weight=conv1_weight,
            use_spiking=False,  # Final layer outputs activation, not spikes
            threshold=config.threshold,
            leak=config.leak
        )
        
        # Delay connection for temporal continuity
        self.delay: Optional[DelayConnection] = None
        if config.use_delay_connections:
            self.delay = DelayConnection(
                delay_steps=config.delay_steps,
                decay=0.9
            )
    
    @classmethod
    def from_encoder(
        cls,
        encoder: nn.Module,
        config: Optional[DecoderConfig] = None
    ) -> "SpikeSEGDecoder":
        """
        Create decoder from a trained encoder.
        
        Extracts weights and configuration from encoder.
        
        Args:
            encoder: Trained encoder module.
            config: Optional decoder configuration.
        
        Returns:
            SpikeSEGDecoder instance with tied weights.
        
        Raises:
            EncoderCompatibilityError: If encoder structure is incompatible.
        """
        # Validate encoder has required layers
        required_layers = ['conv1', 'conv2', 'conv3']
        
        for layer_name in required_layers:
            if not hasattr(encoder, layer_name):
                raise EncoderCompatibilityError(
                    f"Encoder missing required layer '{layer_name}'"
                )
        
        try:
            # Extract weights
            conv1 = encoder.conv1
            conv2 = encoder.conv2
            conv3 = encoder.conv3
            
            # Get weight tensors
            conv1_weight = conv1.conv.weight.data.clone()
            conv2_weight = conv2.conv.weight.data.clone()
            conv3_weight = conv3.conv.weight.data.clone()
            
            # Extract dimensions
            n_classes = conv3_weight.shape[0]
            conv2_channels = conv3_weight.shape[1]
            conv1_channels = conv2_weight.shape[1]
            input_channels = conv1_weight.shape[1]
            
            # Extract kernel sizes
            k1 = conv1_weight.shape[2]
            k2 = conv2_weight.shape[2]
            k3 = conv3_weight.shape[2]
            
            # Get pool kernel size if available
            pool_kernel_size = 2
            if hasattr(encoder, 'pool1') and hasattr(encoder.pool1, 'kernel_size'):
                pool_kernel_size = encoder.pool1.kernel_size
            
            encoder_weights = {
                'conv1': conv1_weight,
                'conv2': conv2_weight,
                'conv3': conv3_weight
            }
            
            return cls(
                n_classes=n_classes,
                conv2_channels=conv2_channels,
                conv1_channels=conv1_channels,
                input_channels=input_channels,
                kernel_sizes=(k1, k2, k3),
                pool_kernel_size=pool_kernel_size,
                config=config,
                encoder_weights=encoder_weights
            )
            
        except Exception as e:
            raise EncoderCompatibilityError(
                f"Failed to extract encoder configuration: {e}"
            ) from e
    
    def reset_state(self) -> None:
        """Reset all stateful components."""
        self.trans_conv3.reset_state()
        self.trans_conv2.reset_state()
        self.trans_conv1.reset_state()
        if self.delay is not None:
            self.delay.reset_state()
    
    def forward(
        self,
        classification_spikes: torch.Tensor,
        pool1_indices: torch.Tensor,
        pool2_indices: torch.Tensor,
        pool1_output_size: Tuple[int, int, int, int],
        pool2_output_size: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Decode classification spikes back to pixel space.
        
        Args:
            classification_spikes: Spikes from classification layer.
                                  Shape: (batch, n_classes, H, W).
            pool1_indices: Max pooling indices from encoder Pool1.
            pool2_indices: Max pooling indices from encoder Pool2.
            pool1_output_size: Output size for UnPool1 (input size to Pool1).
            pool2_output_size: Output size for UnPool2 (input size to Pool2).
        
        Returns:
            Saliency map / segmentation output.
            Shape: (batch, input_channels, H_in, W_in).
        """
        _validate_4d_tensor(classification_spikes, "classification_spikes")
        _validate_4d_tensor(pool1_indices, "pool1_indices")
        _validate_4d_tensor(pool2_indices, "pool2_indices")
        
        # Validate class channels
        if classification_spikes.shape[1] != self.n_classes:
            raise DecoderRuntimeError(
                f"classification_spikes channels ({classification_spikes.shape[1]}) "
                f"doesn't match n_classes ({self.n_classes})"
            )
        
        # Apply delay connection if enabled
        x = classification_spikes
        if self.delay is not None:
            x = self.delay(x)
        
        # TransConv3: Classification → Conv2 space
        x = self.trans_conv3(x)
        
        # UnPool2: Expand to pre-pool2 size
        x = self.unpool2(x, pool2_indices, output_size=pool2_output_size)
        
        # TransConv2: Conv2 → Conv1 space
        x = self.trans_conv2(x)
        
        # UnPool1: Expand to pre-pool1 size
        x = self.unpool1(x, pool1_indices, output_size=pool1_output_size)
        
        # TransConv1: Conv1 → Input space (saliency output)
        x = self.trans_conv1(x)
        
        return x
    
    def decode_single_spike(
        self,
        spike_location: Tuple[int, int],
        class_id: int,
        batch_size: int,
        class_spatial_shape: Tuple[int, int],
        pool1_indices: torch.Tensor,
        pool2_indices: torch.Tensor,
        pool1_output_size: Tuple[int, int, int, int],
        pool2_output_size: Tuple[int, int, int, int],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Decode a single classification spike back to pixel space.
        
        Used by HULK algorithm for instance-wise decoding.
        
        Args:
            spike_location: (x, y) location of the spike.
            class_id: Class channel that spiked.
            batch_size: Batch size for output.
            class_spatial_shape: (H, W) of classification layer.
            pool1_indices: Pooling indices from Pool1.
            pool2_indices: Pooling indices from Pool2.
            pool1_output_size: Output size for UnPool1.
            pool2_output_size: Output size for UnPool2.
            device: Device for tensors.
        
        Returns:
            Pixel-space activation for this single spike.
        """
        x, y = spike_location
        h, w = class_spatial_shape
        
        if device is None:
            device = pool1_indices.device
        
        # Create single-spike tensor
        class_spikes = torch.zeros(
            batch_size, self.n_classes, h, w,
            device=device
        )
        class_spikes[:, class_id, y, x] = 1.0
        
        # Forward through decoder
        return self.forward(
            class_spikes,
            pool1_indices,
            pool2_indices,
            pool1_output_size,
            pool2_output_size
        )
    
    def __repr__(self) -> str:
        return (
            f"SpikeSEGDecoder(\n"
            f"  n_classes={self.n_classes},\n"
            f"  conv2_ch={self.conv2_channels},\n"
            f"  conv1_ch={self.conv1_channels},\n"
            f"  input_ch={self.input_channels},\n"
            f"  kernels={self.kernel_sizes}\n"
            f")"
        )


# =============================================================================
# COMBINED ENCODER-DECODER (FULL SPIKESEG)
# =============================================================================


class SpikeSEGEncoderDecoder(nn.Module):
    """
    Complete SpikeSEG encoder-decoder network.
    
    Combines encoder and decoder for end-to-end semantic segmentation.
    The decoder uses tied weights from the encoder.
    
    This is a convenience wrapper that manages:
        - Forward pass through encoder
        - Storage of pooling indices
        - Forward pass through decoder
        - Combined output
    
    Example:
        >>> model = SpikeSEGEncoderDecoder(encoder)
        >>> segmentation = model(input_spikes)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder_config: Optional[DecoderConfig] = None
    ) -> None:
        """
        Initialize encoder-decoder network.
        
        Args:
            encoder: Trained encoder module.
            decoder_config: Configuration for decoder.
        """
        super().__init__()
        
        self.encoder = encoder
        self.decoder = SpikeSEGDecoder.from_encoder(encoder, decoder_config)
        
        # Storage for pooling indices
        self._pool1_indices: Optional[torch.Tensor] = None
        self._pool2_indices: Optional[torch.Tensor] = None
        self._pool1_input_size: Optional[Tuple[int, ...]] = None
        self._pool2_input_size: Optional[Tuple[int, ...]] = None
    
    def reset_state(self) -> None:
        """Reset all stateful components."""
        if hasattr(self.encoder, 'reset_state'):
            self.encoder.reset_state()
        self.decoder.reset_state()
        
        self._pool1_indices = None
        self._pool2_indices = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder-decoder.
        
        Args:
            x: Input tensor, shape (batch, channels, H, W).
        
        Returns:
            Tuple of (classification_spikes, segmentation_output).
        """
        _validate_4d_tensor(x, "input")
        
        # Encoder forward pass
        # Note: Encoder must return classification spikes and store pool indices
        # This assumes encoder has specific interface - may need adjustment
        
        # For now, this is a placeholder showing the expected flow
        # Actual implementation depends on encoder structure
        raise NotImplementedError(
            "Full encoder-decoder forward pass requires specific encoder interface. "
            "Use encoder and decoder separately, passing pooling indices manually."
        )
    
    def __repr__(self) -> str:
        return f"SpikeSEGEncoderDecoder(encoder={type(self.encoder).__name__})"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "DecoderError",
    "DecoderConfigError",
    "DecoderRuntimeError",
    "EncoderCompatibilityError",
    # Configuration
    "DecoderConfig",
    # Layers
    "DecoderTransConv2d",
    "DecoderUnpool2d",
    "DelayConnection",
    # Main decoder
    "SpikeSEGDecoder",
    "SpikeSEGEncoderDecoder",
]