"""
Spiking Neural Network Layers

This module implements spiking versions of standard neural network layers:
- SpikingConv2d: Convolutional layer with spiking neurons
- SpikingPool2d: Max pooling with index storage for unpooling
- SpikingUnpool2d: Unpooling using stored indices (for decoder)

Paper References:
    - Kheradpisheh et al. 2018: Base convolutional SNN architecture
    - Kirkland et al. 2020 (SpikeSEG): Encoder-decoder with transposed convolutions
    - Kirkland et al. 2022 (HULK-SMASH): Instance segmentation via spike decoding
    - Kirkland et al. 2023 (IGARSS): Modified leak ratios for space awareness

Architecture Overview:

    ENCODER:
        Input Events → Conv1 → Pool1 → Conv2 → Pool2 → Conv3
                       (5×5)   (2×2)   (5×5)   (2×2)   (7×7)
                       4 feat  store   36 feat store   C classes
                               indices         indices

    DECODER:
        Conv3 output → Unpool2 → TransConv2 → Unpool1 → TransConv1 → Segmentation
                       (use idx)  (5×5)       (use idx)  (5×5)        mask

Example:
    >>> conv = SpikingConv2d(
    ...     in_channels=2,
    ...     out_channels=4,
    ...     kernel_size=5,
    ...     neuron_type="lif",
    ...     threshold=10.0,
    ...     leak_factor=0.9
    ... )
    >>> x = torch.randn(1, 2, 64, 64)  # Input "events"
    >>> spikes, membrane, spike_times = conv(x, n_timesteps=10)
"""

############################################################################
# Imports
############################################################################
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Union
from .neurons import BaseNeuron, IFNeuron, LIFNeuron, LeakMode, create_neuron

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# HELPER FUNCTIONS FOR VALIDATION
# =============================================================================


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


def _validate_positive_float(value: float, name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_kernel_size(kernel_size: Union[int, Tuple[int, int]], name: str = "kernel_size") -> None:
    """Validate kernel size is positive int or tuple of positive ints."""
    if isinstance(kernel_size, int):
        if kernel_size <= 0:
            raise ValueError(f"{name} must be positive, got {kernel_size}")
    elif isinstance(kernel_size, tuple):
        if len(kernel_size) != 2:
            raise ValueError(f"{name} tuple must have 2 elements, got {len(kernel_size)}")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"{name} values must be positive, got {kernel_size}")
    else:
        raise TypeError(f"{name} must be int or tuple of ints, got {type(kernel_size).__name__}")


def _validate_4d_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate that a tensor is 4-dimensional (batch, channels, height, width)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D (batch, channels, height, width), "
            f"got {tensor.dim()}D with shape {tensor.shape}"
        )


# =============================================================================
# SPIKING CONVOLUTIONAL LAYER
# =============================================================================


class SpikingConv2d(nn.Module):
    """
    Spiking Convolutional Layer.

    Combines a standard Conv2d (for weights) with spiking neurons.
    Processes input over multiple timesteps, accumulating spikes.

    Architecture:
        Input → Conv2d (weights) → Neuron (LIF/IF) → Output Spikes
                    ↓                    ↓
               Weighted sum      Membrane dynamics
               (spatial)         (temporal)

    Key Features:
        - Tracks spike times for STDP learning
        - Supports both LIF and IF neurons
        - Can freeze weights (for fixed Gabor filters in Conv1)

    Paper Reference:
        Kheradpisheh et al. 2018:
        "The convolution operation is performed on the input spike
         patterns to compute the membrane potentials"

        IGARSS 2023 layer configs:
            Conv1: 5×5, 4 channels, threshold=10, leak=90%
            Conv2: 5×5, 36 channels, threshold=60, leak=10%
            Conv3: 7×7, C channels, threshold=2, leak=0%

    Example:
        >>> # Conv1 from IGARSS paper
        >>> conv1 = SpikingConv2d(
        ...     in_channels=2,      # ON/OFF polarity
        ...     out_channels=4,     # 4 oriented edges
        ...     kernel_size=5,
        ...     neuron_type="lif",
        ...     threshold=10.0,
        ...     leak_factor=0.9,    # 90% leak
        ...     learnable=False     # Fixed Gabor filters
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[int, str] = 0,
        neuron_type: str = "lif",
        threshold: float = 1.0,
        leak_factor: float = 0.0,
        leak_mode: LeakMode = "subtractive",
        learnable: bool = True,
        weight_init_mean: float = 0.8,
        weight_init_std: float = 0.05,
    ) -> None:
        """
        Initialize SpikingConv2d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (feature maps).
            kernel_size: Size of convolutional kernel (e.g., 5 or (5, 5)).
            stride: Convolution stride. Default 1.
            padding: Padding mode. Use "same" for output size = input size,
                    or int for explicit padding.
            neuron_type: "lif" or "if".
            threshold: Neuron firing threshold.
            leak_factor: Leak factor for LIF neurons (0.0 to 1.0).
            leak_mode: "subtractive" or "multiplicative".
            learnable: If True, weights can be updated via STDP.
                      If False, weights are frozen (e.g., for Gabor filters).
            weight_init_mean: Mean for weight initialization.
                             Papers use 0.8 for STDP to work well.
            weight_init_std: Std for weight initialization.
                            Papers use 0.01-0.05.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values are invalid.
        """
        super().__init__()

        # =====================================================================
        # Input Validation
        # =====================================================================

        # Validate channel counts
        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")

        # Validate kernel size
        _validate_kernel_size(kernel_size, "kernel_size")

        # Validate stride
        if isinstance(stride, int):
            _validate_positive_int(stride, "stride")
        else:
            raise TypeError(f"stride must be an integer, got {type(stride).__name__}")

        # Validate padding (can be int or "same")
        if isinstance(padding, int):
            _validate_non_negative_int(padding, "padding")
        elif isinstance(padding, str):
            if padding not in ("same", "valid"):
                raise ValueError(f"padding string must be 'same' or 'valid', got '{padding}'")
        else:
            raise TypeError(f"padding must be int or str, got {type(padding).__name__}")

        # Validate neuron_type
        neuron_type_lower = neuron_type.lower().strip()
        if neuron_type_lower not in ("lif", "if"):
            raise ValueError(f"neuron_type must be 'lif' or 'if', got '{neuron_type}'")

        # Validate threshold
        _validate_positive_float(threshold, "threshold")

        # Validate leak_factor
        if not isinstance(leak_factor, (int, float)):
            raise TypeError(f"leak_factor must be a number, got {type(leak_factor).__name__}")
        if not 0.0 <= leak_factor <= 1.0:
            raise ValueError(f"leak_factor must be in [0, 1], got {leak_factor}")

        # Validate leak_mode
        if leak_mode not in ("subtractive", "multiplicative"):
            raise ValueError(f"leak_mode must be 'subtractive' or 'multiplicative', got '{leak_mode}'")

        # Validate learnable
        if not isinstance(learnable, bool):
            raise TypeError(f"learnable must be a boolean, got {type(learnable).__name__}")

        # Validate weight init params
        if not isinstance(weight_init_mean, (int, float)):
            raise TypeError(f"weight_init_mean must be a number, got {type(weight_init_mean).__name__}")
        if not isinstance(weight_init_std, (int, float)):
            raise TypeError(f"weight_init_std must be a number, got {type(weight_init_std).__name__}")
        if weight_init_std < 0:
            raise ValueError(f"weight_init_std must be non-negative, got {weight_init_std}")

        # =====================================================================
        # Store Configuration
        # =====================================================================

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learnable = learnable

        # =====================================================================
        # Create Layers
        # =====================================================================

        # Create the convolution layer (no bias - STDP doesn't use bias)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias in STDP networks
        )

        # Initialize weights according to paper recommendations
        self._init_weights(weight_init_mean, weight_init_std)

        # Freeze weights if not learnable
        if not learnable:
            self.conv.weight.requires_grad = False

        # Create the spiking neuron
        self.neuron = create_neuron(
            neuron_type=neuron_type_lower,
            threshold=threshold,
            leak_factor=leak_factor,
            leak_mode=leak_mode
        )

        # Store neuron config for later access
        self.neuron_type = neuron_type_lower
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.leak_mode = leak_mode

    def _init_weights(self, mean: float, std: float) -> None:
        """
        Initialize weights for STDP learning.

        STDP learning rule: Δw = a⁺ * w * (1-w) or a⁻ * w * (1-w)

        This rule pushes weights toward 0 or 1. For it to work well,
        weights should start near the middle of [0, 1], hence mean=0.8.

        Args:
            mean: Mean of initial weights.
            std: Standard deviation of initial weights.
        """
        nn.init.normal_(self.conv.weight, mean=mean, std=std)
        # Clamp to valid range [0, 1] for STDP
        with torch.no_grad():
            self.conv.weight.clamp_(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        n_timesteps: int = 1,
        membrane: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]
    ]:
        """
        Forward pass through spiking conv layer.

        Processes input over multiple timesteps, simulating temporal dynamics.

        Args:
            x: Input tensor. Shape: (batch, in_channels, height, width)
               This represents the input current at each timestep.
            n_timesteps: Number of simulation timesteps. Must be positive.
            membrane: Initial membrane potential. If None, starts at zero.
                     Shape: (batch, out_channels, out_height, out_width)
            return_all_timesteps: If True, also return list of spikes at each timestep.

        Returns:
            Tuple of (total_spikes, final_membrane, spike_times, last_pre_reset_membrane):
                - total_spikes: Sum of spikes over all timesteps.
                - final_membrane: Membrane potential after last timestep.
                - spike_times: Time of first spike for each neuron (-1 if never fired).
                - last_pre_reset_membrane: Pre-reset membrane from last timestep (for WTA).

            If return_all_timesteps=True, also returns:
                - all_spikes: List of spike tensors, one per timestep.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If input shapes or values are invalid.

        Example:
            >>> conv = SpikingConv2d(2, 4, 5, threshold=10.0)
            >>> x = torch.randn(1, 2, 32, 32)
            >>> spikes, membrane, times, pre_mem = conv(x, n_timesteps=10)
        """
        # =====================================================================
        # Input Validation
        # =====================================================================

        # Validate input tensor
        _validate_4d_tensor(x, "x")

        # Validate input channels match
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, "
                f"got {x.shape[1]} (input shape: {x.shape})"
            )

        # Validate n_timesteps
        _validate_positive_int(n_timesteps, "n_timesteps")

        # =====================================================================
        # Forward Pass
        # =====================================================================

        batch_size = x.shape[0]

        # Compute convolution (weighted sum of inputs)
        conv_out = self.conv(x)

        # Get output spatial dimensions
        out_shape = conv_out.shape  # (batch, out_channels, H, W)

        # Initialize or validate membrane potential
        if membrane is None:
            membrane = self.neuron.reset_state(out_shape, x.device)
        else:
            # Validate provided membrane
            _validate_4d_tensor(membrane, "membrane")
            if membrane.shape != out_shape:
                raise ValueError(
                    f"Membrane shape mismatch: expected {out_shape}, "
                    f"got {membrane.shape}"
                )
            if membrane.device != x.device:
                raise ValueError(
                    f"Device mismatch: x on {x.device}, membrane on {membrane.device}"
                )

        # Initialize spike tracking
        total_spikes = torch.zeros_like(conv_out)
        spike_times = torch.full_like(conv_out, fill_value=-1.0)
        pre_reset_membrane = torch.zeros_like(conv_out)
        # Fire-once constraint (Kheradpisheh 2018): track which neurons have already fired
        has_fired = torch.zeros_like(conv_out)

        # Optional: track spikes at each timestep
        all_spikes: List[torch.Tensor] = []

        # Simulate over timesteps
        for t in range(n_timesteps):
            # Process through neuron with fire-once constraint
            # (Kheradpisheh 2018: "neurons are not allowed to fire more than once")
            spikes, membrane, pre_reset_membrane = self.neuron(conv_out, membrane, has_fired)

            # Update has_fired mask
            has_fired = torch.clamp(has_fired + spikes, 0.0, 1.0)

            # Accumulate total spikes
            total_spikes = total_spikes + spikes

            # Record first spike time
            first_spike_mask = (spike_times < 0) & (spikes > 0)
            spike_times = torch.where(
                first_spike_mask,
                torch.full_like(spike_times, fill_value=float(t)),
                spike_times
            )

            if return_all_timesteps:
                all_spikes.append(spikes.clone())

        if return_all_timesteps:
            return total_spikes, membrane, spike_times, pre_reset_membrane, all_spikes
        return total_spikes, membrane, spike_times, pre_reset_membrane

    def reset(self) -> None:
        """Reset layer state (membrane potential is handled in forward)."""
        pass  # Membrane is passed as argument, so no internal state to reset

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SpikingConv2d("
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"neuron={self.neuron_type}, "
            f"threshold={self.threshold:.1f}, "
            f"leak={self.leak_factor:.2f}, "
            f"learnable={self.learnable})"
        )


# =============================================================================
# SPIKING POOLING LAYER
# =============================================================================


class SpikingPool2d(nn.Module):
    """
    Spiking Max Pooling Layer with Index Storage.

    Performs max pooling on spike counts/times and stores the indices
    of the "winning" neurons for later unpooling in the decoder.

    Two pooling strategies:
        1. "spike_count": Pool based on total spike count (more spikes = winner)
        2. "first_spike": Pool based on first spike time (earlier = winner)

    The stored indices are CRITICAL for the encoder-decoder architecture.
    They allow the decoder to reconstruct spatial information.

    Paper Reference:
        Kirkland et al. 2020 (SpikeSEG):
        "Unpooling layers use the indices stored during max-pooling
         to place activations back in their original positions"

    Example:
        >>> pool = SpikingPool2d(kernel_size=2, stride=2)
        >>> spikes = torch.tensor([[[[4, 2], [1, 3]]]], dtype=torch.float32)
        >>> pooled, indices = pool(spikes)
        >>> pooled  # tensor([[[[4]]]])  (max of 4,2,1,3)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: int = 0,
        pool_mode: Literal["spike_count", "first_spike"] = "spike_count"
    ) -> None:
        """
        Initialize SpikingPool2d layer.

        Args:
            kernel_size: Size of pooling window.
            stride: Stride of pooling. Default = kernel_size.
            padding: Padding before pooling.
            pool_mode: Pooling strategy.
                      "spike_count": Neuron with most spikes wins.
                      "first_spike": Neuron that fired first wins.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values are invalid.
        """
        super().__init__()

        # Validate kernel_size
        _validate_kernel_size(kernel_size, "kernel_size")

        # Validate stride
        if stride is not None:
            _validate_kernel_size(stride, "stride")

        # Validate padding
        if isinstance(padding, int):
            _validate_non_negative_int(padding, "padding")
        else:
            raise TypeError(f"padding must be an integer, got {type(padding).__name__}")

        # Validate pool_mode
        if pool_mode not in ("spike_count", "first_spike"):
            raise ValueError(
                f"pool_mode must be 'spike_count' or 'first_spike', got '{pool_mode}'"
            )

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.pool_mode = pool_mode

    def forward(
        self,
        x: torch.Tensor,
        spike_times: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through pooling layer.

        Args:
            x: Input tensor (spike counts).
               Shape: (batch, channels, height, width)
            spike_times: First spike times (only needed if pool_mode="first_spike").
                        Shape: (batch, channels, height, width)
                        Values: -1 for never fired, 0+ for spike time.

        Returns:
            Tuple of (pooled, indices):
                - pooled: Max-pooled output.
                - indices: Indices of max elements (for unpooling).

        Raises:
            TypeError: If input types are invalid.
            ValueError: If input shapes are invalid or spike_times missing when required.
        """
        # Validate input tensor
        _validate_4d_tensor(x, "x")

        if self.pool_mode == "spike_count":
            # Pool based on spike count (standard max pooling)
            pooled, indices = F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                return_indices=True
            )

        elif self.pool_mode == "first_spike":
            # Pool based on first spike time (earlier = better)
            if spike_times is None:
                raise ValueError(
                    "spike_times is required when pool_mode='first_spike'. "
                    "Pass the spike_times output from SpikingConv2d."
                )

            # Validate spike_times
            _validate_4d_tensor(spike_times, "spike_times")
            if spike_times.shape != x.shape:
                raise ValueError(
                    f"spike_times shape {spike_times.shape} must match "
                    f"input shape {x.shape}"
                )

            # Convert times to "priority" where earlier is higher
            priority = torch.where(
                spike_times >= 0,
                -spike_times,  # Negate so earlier becomes larger
                torch.tensor(float('-inf'), device=spike_times.device)
            )

            # Max pooling on priority gives us earliest spike
            _, indices = F.max_pool2d(
                priority,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                return_indices=True
            )

            # Use indices to gather actual spike counts
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, channels, -1)
            indices_flat = indices.view(batch_size, channels, -1)
            pooled_flat = torch.gather(x_flat, dim=2, index=indices_flat)
            pooled = pooled_flat.view_as(indices)

        else:
            # This shouldn't happen due to __init__ validation, but be safe
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return pooled, indices

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SpikingPool2d("
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"mode={self.pool_mode})"
        )


# =============================================================================
# SPIKING UNPOOLING LAYER
# =============================================================================


class SpikingUnpool2d(nn.Module):
    """
    Spiking Unpooling Layer.

    Reverses the pooling operation using stored indices.
    Places values back in their original positions, with zeros elsewhere.

    This is essential for the decoder to reconstruct spatial resolution.

    Paper Reference:
        Kirkland et al. 2020 (SpikeSEG):
        "The decoding section mirrors the encoding layers using
         transposed convolutions and unpooling operations"

    Example:
        >>> pool = SpikingPool2d(kernel_size=2, stride=2)
        >>> unpool = SpikingUnpool2d(kernel_size=2, stride=2)
        >>>
        >>> x = torch.tensor([[[[4., 2.], [1., 3.]]]])  # 1x1x2x2
        >>> pooled, indices = pool(x)  # 1x1x1x1
        >>> unpooled = unpool(pooled, indices, output_size=x.shape)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: int = 0
    ) -> None:
        """
        Initialize SpikingUnpool2d layer.

        Args:
            kernel_size: Size of unpooling window (must match pooling).
            stride: Stride of unpooling (must match pooling).
            padding: Padding (must match pooling).

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values are invalid.
        """
        super().__init__()

        # Validate kernel_size
        _validate_kernel_size(kernel_size, "kernel_size")

        # Validate stride
        if stride is not None:
            _validate_kernel_size(stride, "stride")

        # Validate padding
        if isinstance(padding, int):
            _validate_non_negative_int(padding, "padding")
        else:
            raise TypeError(f"padding must be an integer, got {type(padding).__name__}")

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        output_size: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Forward pass through unpooling layer.

        Args:
            x: Pooled input tensor.
               Shape: (batch, channels, pooled_h, pooled_w)
            indices: Indices from corresponding pooling operation.
                    Shape: same as x
            output_size: Desired output size. If None, computed from kernel/stride.
                        Usually pass the shape of the original pre-pooled tensor.

        Returns:
            Unpooled tensor with values placed at original positions.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If input shapes don't match.
        """
        # Validate input tensor
        _validate_4d_tensor(x, "x")

        # Validate indices tensor
        _validate_4d_tensor(indices, "indices")

        # Validate shapes match
        if x.shape != indices.shape:
            raise ValueError(
                f"x shape {x.shape} must match indices shape {indices.shape}"
            )

        # Validate output_size if provided
        if output_size is not None:
            if not isinstance(output_size, (tuple, list, torch.Size)):
                raise TypeError(
                    f"output_size must be a tuple, got {type(output_size).__name__}"
                )
            if len(output_size) != 4:
                raise ValueError(
                    f"output_size must have 4 elements (batch, channels, H, W), "
                    f"got {len(output_size)}"
                )

        return F.max_unpool2d(
            x,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SpikingUnpool2d("
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride})"
        )


# =============================================================================
# SPIKING TRANSPOSED CONVOLUTION (for decoder)
# =============================================================================


class SpikingTransposedConv2d(nn.Module):
    """
    Spiking Transposed Convolutional Layer.

    Used in the decoder to upsample feature maps.
    Can optionally share weights with a corresponding encoder layer.

    Note: In SpikeSEG, decoder transposed convolutions share weights
    with encoder convolutions (tied weights). This is set up after
    both encoder and decoder are created.

    Paper Reference:
        Kirkland et al. 2020 (SpikeSEG):
        "Transposed convolutions with tied weights from the encoder"

    Example:
        >>> encoder_conv = SpikingConv2d(4, 36, kernel_size=5)
        >>> decoder_tconv = SpikingTransposedConv2d(36, 4, kernel_size=5)
        >>> decoder_tconv.tie_weights(encoder_conv)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0
    ) -> None:
        """
        Initialize SpikingTransposedConv2d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolutional kernel.
            stride: Convolution stride.
            padding: Input padding.
            output_padding: Additional size added to output.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values are invalid.
        """
        super().__init__()

        # Validate channels
        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")

        # Validate kernel_size
        _validate_kernel_size(kernel_size, "kernel_size")

        # Validate stride
        _validate_positive_int(stride, "stride")

        # Validate padding
        _validate_non_negative_int(padding, "padding")

        # Validate output_padding
        _validate_non_negative_int(output_padding, "output_padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Create transposed convolution
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False
        )

        # Track if weights are tied
        self._weights_tied = False
        self._tied_to: Optional[SpikingConv2d] = None

    def tie_weights(self, encoder_conv: SpikingConv2d) -> None:
        """
        Tie weights to an encoder convolution layer.

        After calling this, the transposed conv will use the encoder's
        weights (transposed), and they will stay synchronized.

        Args:
            encoder_conv: The encoder SpikingConv2d to tie weights to.

        Raises:
            TypeError: If encoder_conv is not a SpikingConv2d.
            ValueError: If channel dimensions don't match.
        """
        # Validate encoder_conv type
        if not isinstance(encoder_conv, SpikingConv2d):
            raise TypeError(
                f"encoder_conv must be a SpikingConv2d, "
                f"got {type(encoder_conv).__name__}"
            )

        # For tied weights: encoder (out, in, k, k) -> decoder needs (in, out, k, k)
        # decoder.in_channels should equal encoder.out_channels
        if self.in_channels != encoder_conv.out_channels:
            raise ValueError(
                f"Channel mismatch: decoder in_channels ({self.in_channels}) "
                f"must equal encoder out_channels ({encoder_conv.out_channels})"
            )
        if self.out_channels != encoder_conv.in_channels:
            raise ValueError(
                f"Channel mismatch: decoder out_channels ({self.out_channels}) "
                f"must equal encoder in_channels ({encoder_conv.in_channels})"
            )

        self._weights_tied = True
        self._tied_to = encoder_conv

        # Remove own parameters to save memory
        del self.conv_transpose.weight
        self.conv_transpose.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transposed convolution.

        Args:
            x: Input tensor. Shape: (batch, in_channels, height, width)

        Returns:
            Output tensor. Shape: (batch, out_channels, out_height, out_width)

        Raises:
            TypeError: If input is not a tensor.
            ValueError: If input shape is invalid.
        """
        # Validate input
        _validate_4d_tensor(x, "x")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, "
                f"got {x.shape[1]} (input shape: {x.shape})"
            )

        if self._weights_tied and self._tied_to is not None:
            # Use encoder's weights
            weight = self._tied_to.conv.weight
            return F.conv_transpose2d(
                x,
                weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding
            )
        else:
            return self.conv_transpose(x)

    def __repr__(self) -> str:
        """String representation for debugging."""
        tied_str = ", tied=True" if self._weights_tied else ""
        return (
            f"SpikingTransposedConv2d("
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}{tied_str})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SpikingConv2d",
    "SpikingPool2d",
    "SpikingUnpool2d",
    "SpikingTransposedConv2d",
]
