"""
HULK: Hierarchical Unravelling of Linked Kernels.

This module implements the decoder unraveling process that traces
classification spikes back to the pixel domain, recording intermediate
spike activity for ASH computation.

The HULK process is the bridge between the encoder-decoder network
and the SMASH instance segmentation algorithm.

Pipeline:
    Classification Spike → HULK → (Pixel Mask, Spike Activity) → ASH → SMASH

Paper Reference:
    Kirkland et al. 2022 - "Unsupervised Spiking Instance Segmentation 
    on Event Data using STDP Features"
    
    "The Hierarchical Unravelling of Linked Kernels (HULK) process permits 
    spiking activity from the classification convolution layer, to be tracked 
    as it propagates through the decoding layers of HULK. It in essence works 
    the same as if you passed each classification spike into a separate 
    SpikeSEG decoding network."

Key Insight:
    Instead of decoding ALL classification spikes together (semantic segmentation),
    HULK decodes EACH spike individually (instance segmentation). This reveals
    which pixels/features contributed to each specific classification activation.

Example:
    >>> from spikeseg.algorithms.hulk import HULKDecoder
    >>> 
    >>> # Create HULK decoder from trained encoder
    >>> hulk = HULKDecoder.from_encoder(encoder)
    >>> 
    >>> # Unravel each classification spike
    >>> for spike_loc in classification_spikes:
    ...     pixel_mask, spike_activity = hulk.unravel(spike_loc, pool_indices)
    ...     ash = ActiveSpikeHash.from_spike_activity(spike_activity, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .smash import ActiveSpikeHash, BoundingBox, Instance


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class HULKError(Exception):
    """Base exception for HULK module errors."""
    pass


class HULKConfigError(HULKError):
    """Raised for configuration/initialization errors."""
    pass


class HULKRuntimeError(HULKError):
    """Raised for runtime errors during unraveling."""
    pass


class EncoderCompatibilityError(HULKError):
    """Raised when encoder is incompatible with HULK."""
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


def _validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...], 
    name: str
) -> None:
    """Validate tensor has expected shape."""
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {tensor.shape}, expected {expected_shape}"
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


def _validate_range(value: Any, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within a specified range [min_val, max_val]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ValueError(
            f"{name} must be in range [{min_val}, {max_val}], got {value}"
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


def _validate_4d_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor is 4D with shape (N, C, H, W)."""
    _validate_tensor(tensor, name)
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D (N, C, H, W), got {tensor.dim()}D with shape {tensor.shape}"
        )


def _validate_weight_tensor(
    tensor: torch.Tensor, 
    expected_dim: int,
    name: str
) -> None:
    """Validate a weight tensor for convolution operations."""
    _validate_tensor(tensor, name)
    _validate_tensor_dim(tensor, expected_dim, name)
    
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} must be a floating point tensor, got {tensor.dtype}"
        )


def _validate_same_device(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                          name1: str, name2: str) -> None:
    """Validate two tensors are on the same device."""
    if tensor1.device != tensor2.device:
        raise ValueError(
            f"{name1} and {name2} must be on the same device. "
            f"Got {tensor1.device} and {tensor2.device}"
        )


# =============================================================================
# SPIKE ACTIVITY RECORDER
# =============================================================================


@dataclass
class LayerSpikeActivity:
    """
    Records spike activity at a single decoder layer.
    
    Tracks spatial (x, y), featural (channel), and temporal (timestep)
    information for all spikes at this layer.
    
    Attributes:
        layer_name: Name of the layer (e.g., "trans_conv2", "unpool1").
        n_channels: Number of feature channels at this layer.
        spikes: List of (x, y, channel, timestep) tuples.
        spatial_shape: (height, width) of the feature map.
    
    Example:
        >>> activity = LayerSpikeActivity(
        ...     layer_name="trans_conv1",
        ...     n_channels=4,
        ...     spatial_shape=(32, 32)
        ... )
        >>> activity.add_spike(x=10, y=20, channel=2, timestep=5)
        >>> activity.n_spikes
        1
    """
    layer_name: str
    n_channels: int
    spatial_shape: Tuple[int, int]
    spikes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate layer activity configuration."""
        if not isinstance(self.layer_name, str):
            raise TypeError(
                f"layer_name must be a string, got {type(self.layer_name).__name__}"
            )
        if not self.layer_name:
            raise ValueError("layer_name cannot be empty")
        
        _validate_positive_int(self.n_channels, "n_channels")
        _validate_tuple_of_ints(self.spatial_shape, 2, "spatial_shape")
        
        if self.spatial_shape[0] <= 0 or self.spatial_shape[1] <= 0:
            raise ValueError(
                f"spatial_shape dimensions must be positive, got {self.spatial_shape}"
            )
    
    def add_spike(self, x: int, y: int, channel: int, timestep: int) -> None:
        """
        Record a spike at this layer.
        
        Args:
            x: X coordinate (column).
            y: Y coordinate (row).
            channel: Channel/feature index.
            timestep: Time index.
        
        Raises:
            ValueError: If coordinates are out of bounds.
        """
        _validate_non_negative_int(x, "x")
        _validate_non_negative_int(y, "y")
        _validate_non_negative_int(channel, "channel")
        _validate_non_negative_int(timestep, "timestep")
        
        # Bounds checking
        h, w = self.spatial_shape
        if x >= w:
            raise ValueError(f"x={x} out of bounds for width={w}")
        if y >= h:
            raise ValueError(f"y={y} out of bounds for height={h}")
        if channel >= self.n_channels:
            raise ValueError(
                f"channel={channel} out of bounds for n_channels={self.n_channels}"
            )
        
        self.spikes.append((x, y, channel, timestep))
    
    def add_spikes_from_tensor(
        self, 
        spike_tensor: torch.Tensor, 
        timestep: int,
        feature_offset: int = 0
    ) -> None:
        """
        Add spikes from a binary tensor.
        
        Args:
            spike_tensor: Binary tensor of shape (channels, height, width).
            timestep: Current timestep.
            feature_offset: Offset to add to channel indices (for global feature numbering).
        
        Raises:
            TypeError: If spike_tensor is not a tensor.
            ValueError: If spike_tensor is not 3D or dimensions don't match.
        """
        _validate_tensor(spike_tensor, "spike_tensor")
        _validate_tensor_dim(spike_tensor, 3, "spike_tensor")
        _validate_non_negative_int(timestep, "timestep")
        _validate_non_negative_int(feature_offset, "feature_offset")
        
        c, h, w = spike_tensor.shape
        if c != self.n_channels:
            raise ValueError(
                f"spike_tensor channels ({c}) doesn't match n_channels ({self.n_channels})"
            )
        if (h, w) != self.spatial_shape:
            raise ValueError(
                f"spike_tensor spatial shape ({h}, {w}) doesn't match "
                f"spatial_shape {self.spatial_shape}"
            )
        
        # Find all spike locations
        nonzero = torch.nonzero(spike_tensor, as_tuple=False)
        
        for idx in nonzero:
            channel, y, x = idx.tolist()
            self.spikes.append((x, y, channel + feature_offset, timestep))
    
    def to_feature_time_dict(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Convert to dict format for ASH creation.
        
        Returns:
            Dict mapping feature_id → list of (x, y, timestep) tuples.
        """
        result: Dict[int, List[Tuple[int, int, int]]] = {}
        
        for x, y, channel, timestep in self.spikes:
            if channel not in result:
                result[channel] = []
            result[channel].append((x, y, timestep))
        
        return result
    
    @property
    def n_spikes(self) -> int:
        """Total number of spikes recorded."""
        return len(self.spikes)
    
    def __repr__(self) -> str:
        return (
            f"LayerSpikeActivity(layer={self.layer_name}, "
            f"n_spikes={self.n_spikes})"
        )


@dataclass 
class HULKResult:
    """
    Result of HULK unraveling for a single classification spike.
    
    Contains all information needed to create an Instance for SMASH.
    
    Attributes:
        spike_location: (x, y) location of the classification spike.
        pixel_mask: Binary mask in original image space showing contributing pixels.
        layer_activities: Dict mapping layer_name → LayerSpikeActivity.
        ash: Computed Active Spike Hash (if n_features and n_timesteps provided).
        bbox: Computed bounding box from pixel_mask.
    
    Example:
        >>> result = HULKResult(
        ...     spike_location=(10, 20),
        ...     pixel_mask=torch.zeros(64, 64)
        ... )
        >>> result.compute_bbox()
        >>> result.to_instance(instance_id=0, n_features=41, n_timesteps=10)
    """
    spike_location: Tuple[int, int]
    pixel_mask: torch.Tensor
    layer_activities: Dict[str, LayerSpikeActivity] = field(default_factory=dict)
    ash: Optional[ActiveSpikeHash] = None
    bbox: Optional[BoundingBox] = None
    
    def __post_init__(self) -> None:
        """Validate HULK result configuration."""
        _validate_tuple_of_ints(self.spike_location, 2, "spike_location")
        _validate_tensor(self.pixel_mask, "pixel_mask")
        
        if self.pixel_mask.dim() not in (2, 4):
            raise ValueError(
                f"pixel_mask must be 2D (H, W) or 4D (1, 1, H, W), "
                f"got {self.pixel_mask.dim()}D"
            )
    
    def compute_ash(self, n_features: int, n_timesteps: int) -> ActiveSpikeHash:
        """
        Compute ASH from recorded layer activities.
        
        Args:
            n_features: Total number of features across all layers.
            n_timesteps: Number of timesteps in the sequence.
        
        Returns:
            ActiveSpikeHash instance.
        
        Raises:
            ValueError: If n_features or n_timesteps invalid.
        """
        _validate_positive_int(n_features, "n_features")
        _validate_positive_int(n_timesteps, "n_timesteps")
        
        # Combine all layer activities
        combined_spikes: Dict[int, List[Tuple[int, int, int]]] = {}
        
        for layer_activity in self.layer_activities.values():
            layer_dict = layer_activity.to_feature_time_dict()
            for feature_id, spikes in layer_dict.items():
                if feature_id not in combined_spikes:
                    combined_spikes[feature_id] = []
                combined_spikes[feature_id].extend(spikes)
        
        # Get device from pixel_mask
        device = self.pixel_mask.device if self.pixel_mask.dim() == 2 else self.pixel_mask.device
        
        self.ash = ActiveSpikeHash.from_spike_activity(
            combined_spikes, 
            n_features, 
            n_timesteps,
            device=device
        )
        return self.ash
    
    def compute_bbox(self) -> Optional[BoundingBox]:
        """Compute bounding box from pixel mask."""
        mask = self.pixel_mask
        if mask.dim() == 4:
            mask = mask[0, 0]
        self.bbox = BoundingBox.from_mask(mask)
        return self.bbox
    
    def to_instance(
        self, 
        instance_id: int, 
        n_features: int, 
        n_timesteps: int,
        class_id: int = 0
    ) -> Instance:
        """
        Convert to Instance for SMASH processing.
        
        Args:
            instance_id: Unique identifier for this instance.
            n_features: Total number of features for ASH.
            n_timesteps: Number of timesteps for ASH.
            class_id: Class label (default 0 for single-class).
        
        Returns:
            Instance object ready for SMASH.
        
        Raises:
            ValueError: If pixel_mask is empty (no bounding box).
            TypeError: If instance_id or class_id not integers.
        """
        _validate_non_negative_int(instance_id, "instance_id")
        _validate_positive_int(n_features, "n_features")
        _validate_positive_int(n_timesteps, "n_timesteps")
        _validate_non_negative_int(class_id, "class_id")
        
        if self.ash is None:
            self.compute_ash(n_features, n_timesteps)
        
        if self.bbox is None:
            self.compute_bbox()
        
        if self.bbox is None:
            raise ValueError(
                f"Cannot create Instance: pixel_mask is empty (no nonzero pixels). "
                f"spike_location={self.spike_location}"
            )
        
        # Ensure pixel_mask is 2D for Instance
        mask = self.pixel_mask
        if mask.dim() == 4:
            mask = mask[0, 0]
        
        return Instance(
            instance_id=instance_id,
            ash=self.ash,
            bbox=self.bbox,
            class_id=class_id,
            mask=mask,
            spike_location=self.spike_location
        )
    
    def __repr__(self) -> str:
        return (
            f"HULKResult(spike_loc={self.spike_location}, "
            f"n_layers={len(self.layer_activities)}, "
            f"bbox={self.bbox})"
        )


# =============================================================================
# HULK DECODER
# =============================================================================


class HULKDecoder(nn.Module):
    """
    HULK: Hierarchical Unravelling of Linked Kernels.
    
    Decodes individual classification spikes back to pixel space,
    tracking intermediate spike activity for ASH computation.
    
    Unlike standard decoding (which processes all spikes together),
    HULK processes each classification spike individually to determine
    which pixels/features contributed to that specific activation.
    
    Paper Reference:
        "It in essence works the same as if you passed each classification 
        spike into a separate SpikeSEG decoding network, so that only that 
        spiking instance in the classification layer is mapped to the pixel 
        domain, showing directly the features and pixels that cause the 
        classification spike."
    
    Architecture (mirrors encoder):
        Class Spike → UnPool2 → TransConv2 → UnPool1 → TransConv1 → Pixel Mask
    
    Example:
        >>> hulk = HULKDecoder.from_encoder(encoder)
        >>> 
        >>> # For each classification spike
        >>> for t, spike_map in enumerate(classification_spikes):
        ...     spike_locs = torch.nonzero(spike_map, as_tuple=False)
        ...     for loc in spike_locs:
        ...         result = hulk.unravel_spike(
        ...             spike_location=(loc[1].item(), loc[0].item()),
        ...             timestep=t,
        ...             pool_indices=[pool1_idx, pool2_idx],
        ...             output_sizes=[pool1_input_size, pool2_input_size]
        ...         )
    """
    
    def __init__(
        self,
        trans_conv1_weight: torch.Tensor,
        trans_conv2_weight: torch.Tensor,
        conv3_weight: torch.Tensor,
        kernel_sizes: Tuple[int, int, int] = (5, 5, 7),
        pool_kernel_size: int = 2
    ) -> None:
        """
        Initialize HULK decoder.
        
        Args:
            trans_conv1_weight: Weights for TransConv1 (tied to Conv1).
                              Shape: (out_channels, in_channels, kH, kW)
            trans_conv2_weight: Weights for TransConv2 (tied to Conv2).
                              Shape: (out_channels, in_channels, kH, kW)
            conv3_weight: Weights for Conv3 (classification layer).
                         Used for backward projection.
                         Shape: (n_classes, in_channels, kH, kW)
            kernel_sizes: Kernel sizes for (Conv1, Conv2, Conv3).
            pool_kernel_size: Pooling kernel size.
        
        Raises:
            HULKConfigError: If weight shapes are invalid or incompatible.
            TypeError: If inputs are not the correct types.
        """
        super().__init__()
        
        # =================================================================
        # Validate all inputs
        # =================================================================
        
        # Validate weight tensors
        try:
            _validate_weight_tensor(trans_conv1_weight, 4, "trans_conv1_weight")
            _validate_weight_tensor(trans_conv2_weight, 4, "trans_conv2_weight")
            _validate_weight_tensor(conv3_weight, 4, "conv3_weight")
        except (TypeError, ValueError) as e:
            raise HULKConfigError(f"Invalid weight tensor: {e}") from e
        
        # Validate kernel_sizes
        _validate_tuple_of_ints(kernel_sizes, 3, "kernel_sizes")
        for i, k in enumerate(kernel_sizes):
            if k <= 0:
                raise HULKConfigError(
                    f"kernel_sizes[{i}] must be positive, got {k}"
                )
            if k % 2 == 0:
                raise HULKConfigError(
                    f"kernel_sizes[{i}] must be odd for symmetric padding, got {k}"
                )
        
        # Validate pool_kernel_size
        _validate_positive_int(pool_kernel_size, "pool_kernel_size")
        
        # Validate weight shape consistency
        # trans_conv1: (input_channels, conv1_out_channels, k1, k1)
        # trans_conv2: (conv1_out_channels, conv2_out_channels, k2, k2)
        # conv3: (n_classes, conv2_out_channels, k3, k3)
        
        # Check kernel sizes match
        if trans_conv1_weight.shape[2] != kernel_sizes[0]:
            raise HULKConfigError(
                f"trans_conv1_weight kernel size ({trans_conv1_weight.shape[2]}) "
                f"doesn't match kernel_sizes[0] ({kernel_sizes[0]})"
            )
        if trans_conv2_weight.shape[2] != kernel_sizes[1]:
            raise HULKConfigError(
                f"trans_conv2_weight kernel size ({trans_conv2_weight.shape[2]}) "
                f"doesn't match kernel_sizes[1] ({kernel_sizes[1]})"
            )
        if conv3_weight.shape[2] != kernel_sizes[2]:
            raise HULKConfigError(
                f"conv3_weight kernel size ({conv3_weight.shape[2]}) "
                f"doesn't match kernel_sizes[2] ({kernel_sizes[2]})"
            )
        
        # Check square kernels
        for w, name in [
            (trans_conv1_weight, "trans_conv1_weight"),
            (trans_conv2_weight, "trans_conv2_weight"),
            (conv3_weight, "conv3_weight")
        ]:
            if w.shape[2] != w.shape[3]:
                raise HULKConfigError(
                    f"{name} must have square kernel, got {w.shape[2]}x{w.shape[3]}"
                )
        
        # Check channel compatibility
        # conv3 input channels should match trans_conv2 output channels
        if conv3_weight.shape[1] != trans_conv2_weight.shape[0]:
            raise HULKConfigError(
                f"conv3_weight input channels ({conv3_weight.shape[1]}) must match "
                f"trans_conv2_weight output channels ({trans_conv2_weight.shape[0]})"
            )
        
        # trans_conv2 input channels should match trans_conv1 output channels
        if trans_conv2_weight.shape[1] != trans_conv1_weight.shape[0]:
            raise HULKConfigError(
                f"trans_conv2_weight input channels ({trans_conv2_weight.shape[1]}) must match "
                f"trans_conv1_weight output channels ({trans_conv1_weight.shape[0]})"
            )
        
        # Check all weights on same device
        if trans_conv1_weight.device != trans_conv2_weight.device:
            raise HULKConfigError(
                f"All weights must be on same device. "
                f"trans_conv1_weight on {trans_conv1_weight.device}, "
                f"trans_conv2_weight on {trans_conv2_weight.device}"
            )
        if trans_conv1_weight.device != conv3_weight.device:
            raise HULKConfigError(
                f"All weights must be on same device. "
                f"trans_conv1_weight on {trans_conv1_weight.device}, "
                f"conv3_weight on {conv3_weight.device}"
            )
        
        # =================================================================
        # Store configuration
        # =================================================================
        
        # Store weights as buffers (not parameters - not trained)
        self.register_buffer('trans_conv1_weight', trans_conv1_weight.clone())
        self.register_buffer('trans_conv2_weight', trans_conv2_weight.clone())
        self.register_buffer('conv3_weight', conv3_weight.clone())
        
        self.kernel_sizes = kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        
        # Extract dimensions
        # Weight shapes: conv1=(4,2,k,k), conv2=(36,4,k,k), conv3=(2,36,k,k)
        self.n_classes = conv3_weight.shape[0]           # 2 (output classes)
        self.conv2_channels = conv3_weight.shape[1]      # 36 (pool2_indices channels)
        self.conv1_channels = trans_conv2_weight.shape[1]  # 4 (pool1_indices channels)
        self.input_channels = trans_conv1_weight.shape[1]  # 2 (input image channels)
        
        # Total features for ASH (from paper: 4 + 36 + 1 = 41 for face network)
        self.n_features = self.conv1_channels + self.conv2_channels + self.n_classes
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> "HULKDecoder":
        """
        Create HULK decoder from a trained encoder.
        
        Extracts weights from encoder convolutional layers and creates
        the decoder with tied weights.
        
        Args:
            encoder: Trained encoder module with conv1, conv2, conv3 layers.
                    Each layer must have a .conv attribute with .weight.
        
        Returns:
            HULKDecoder instance.
        
        Raises:
            EncoderCompatibilityError: If encoder doesn't have required structure.
        """
        # Validate encoder has required attributes
        required_layers = ['conv1', 'conv2', 'conv3']
        
        for layer_name in required_layers:
            if not hasattr(encoder, layer_name):
                raise EncoderCompatibilityError(
                    f"Encoder missing required layer '{layer_name}'. "
                    f"Expected layers: {required_layers}"
                )
            
            layer = getattr(encoder, layer_name)
            
            if not hasattr(layer, 'conv'):
                raise EncoderCompatibilityError(
                    f"Encoder.{layer_name} missing 'conv' attribute. "
                    f"Expected SpikingConv2d layer with .conv (nn.Conv2d)"
                )
            
            conv = layer.conv
            if not hasattr(conv, 'weight'):
                raise EncoderCompatibilityError(
                    f"Encoder.{layer_name}.conv missing 'weight' attribute"
                )
            
            if not isinstance(conv.weight, torch.Tensor):
                raise EncoderCompatibilityError(
                    f"Encoder.{layer_name}.conv.weight must be a tensor, "
                    f"got {type(conv.weight).__name__}"
                )
        
        try:
            # Extract weights (assuming encoder has SpikingConv2d layers)
            conv1_weight = encoder.conv1.conv.weight.data.clone()
            conv2_weight = encoder.conv2.conv.weight.data.clone()
            conv3_weight = encoder.conv3.conv.weight.data.clone()
            
            # Extract kernel sizes
            k1 = conv1_weight.shape[2]
            k2 = conv2_weight.shape[2]
            k3 = conv3_weight.shape[2]
            
            return cls(
                trans_conv1_weight=conv1_weight,
                trans_conv2_weight=conv2_weight,
                conv3_weight=conv3_weight,
                kernel_sizes=(k1, k2, k3)
            )
        except Exception as e:
            raise EncoderCompatibilityError(
                f"Failed to extract weights from encoder: {e}"
            ) from e
    
    def _validate_pool_indices(
        self,
        pool_indices: torch.Tensor,
        name: str,
        expected_channels: int
    ) -> None:
        """Validate pooling indices tensor."""
        _validate_tensor(pool_indices, name)
        _validate_4d_tensor(pool_indices, name)
        
        if pool_indices.shape[0] != 1:
            raise HULKRuntimeError(
                f"{name} batch size must be 1, got {pool_indices.shape[0]}"
            )
        
        if pool_indices.shape[1] != expected_channels:
            raise HULKRuntimeError(
                f"{name} channels ({pool_indices.shape[1]}) doesn't match "
                f"expected ({expected_channels})"
            )
    
    def _validate_output_size(
        self,
        output_size: Tuple[int, int, int, int],
        name: str
    ) -> None:
        """Validate output size tuple."""
        _validate_tuple_of_ints(output_size, 4, name)
        
        for i, dim in enumerate(output_size):
            if dim <= 0:
                raise HULKRuntimeError(
                    f"{name}[{i}] must be positive, got {dim}"
                )
    
    def unravel_spike(
        self,
        spike_location: Tuple[int, int],
        timestep: int,
        pool1_indices: torch.Tensor,
        pool2_indices: torch.Tensor,
        pool1_output_size: Tuple[int, int, int, int],
        pool2_output_size: Tuple[int, int, int, int],
        class_id: int = 0,
        threshold: float = 0.5
    ) -> HULKResult:
        """
        Unravel a single classification spike back to pixel space.
        
        Args:
            spike_location: (x, y) location of the classification spike.
            timestep: Current timestep (for ASH recording).
            pool1_indices: Pooling indices from Pool1 in encoder.
            pool2_indices: Pooling indices from Pool2 in encoder.
            pool1_output_size: Output size for UnPool1 (input size to Pool1).
            pool2_output_size: Output size for UnPool2 (input size to Pool2).
            class_id: Class channel that spiked (for multi-class).
            threshold: Threshold for considering activation as contributing.
        
        Returns:
            HULKResult containing pixel mask and layer activities.
        
        Raises:
            HULKRuntimeError: If inputs are invalid or incompatible.
        """
        # =================================================================
        # Validate all inputs
        # =================================================================
        
        _validate_tuple_of_ints(spike_location, 2, "spike_location")
        x, y = spike_location
        
        _validate_non_negative_int(timestep, "timestep")
        _validate_non_negative_int(class_id, "class_id")
        _validate_range(threshold, 0.0, 1.0, "threshold")
        
        # Validate class_id range
        if class_id >= self.n_classes:
            raise HULKRuntimeError(
                f"class_id ({class_id}) out of range [0, {self.n_classes})"
            )
        
        # Validate pool indices
        self._validate_pool_indices(pool1_indices, "pool1_indices", self.conv1_channels)
        self._validate_pool_indices(pool2_indices, "pool2_indices", self.conv2_channels)
        
        # Validate output sizes
        self._validate_output_size(pool1_output_size, "pool1_output_size")
        self._validate_output_size(pool2_output_size, "pool2_output_size")
        
        # Validate spike location within classification map bounds
        class_h, class_w = pool2_indices.shape[2], pool2_indices.shape[3]
        if x >= class_w:
            raise HULKRuntimeError(
                f"spike_location x={x} out of bounds for classification width={class_w}"
            )
        if y >= class_h:
            raise HULKRuntimeError(
                f"spike_location y={y} out of bounds for classification height={class_h}"
            )
        
        # Ensure pool indices on same device as weights
        device = self.trans_conv1_weight.device
        pool1_indices = pool1_indices.to(device)
        pool2_indices = pool2_indices.to(device)
        
        # =================================================================
        # Initialize result
        # =================================================================
        
        result = HULKResult(
            spike_location=spike_location,
            pixel_mask=torch.zeros(1, 1, 1, 1, device=device),  # Placeholder
            layer_activities={}
        )
        
        # =====================================================================
        # Step 1: Create single-spike activation at classification layer
        # =====================================================================
        
        # Create activation with single spike
        class_activation = torch.zeros(1, self.n_classes, class_h, class_w, device=device)
        class_activation[0, class_id, y, x] = 1.0
        
        # Record classification spike
        class_activity = LayerSpikeActivity(
            layer_name="classification",
            n_channels=self.n_classes,
            spatial_shape=(class_h, class_w)
        )
        class_activity.add_spike(x, y, class_id, timestep)
        result.layer_activities["classification"] = class_activity
        
        # =====================================================================
        # Step 2: Project through Conv3 weights (backward)
        # =====================================================================
        
        # Use transposed convolution to project back
        # Conv3: (n_classes, conv2_channels, k, k)
        # We need the contribution of this spike to conv2 feature maps
        
        k3 = self.kernel_sizes[2]
        padding3 = k3 // 2
        
        conv2_activation = F.conv_transpose2d(
            class_activation,
            self.conv3_weight,
            padding=padding3
        )
        
        # Threshold to get contributing features
        conv2_activation = (conv2_activation > threshold).float()
        
        # Record Conv2/TransConv2 activity
        conv2_activity = LayerSpikeActivity(
            layer_name="trans_conv2",
            n_channels=self.conv2_channels,
            spatial_shape=(conv2_activation.shape[2], conv2_activation.shape[3])
        )
        conv2_activity.add_spikes_from_tensor(
            conv2_activation[0], 
            timestep,
            feature_offset=self.conv1_channels  # Offset for global feature numbering
        )
        result.layer_activities["trans_conv2"] = conv2_activity
        
        # =====================================================================
        # Step 3: UnPool2
        # =====================================================================
        
        unpool2_output = F.max_unpool2d(
            conv2_activation,
            pool2_indices,
            kernel_size=self.pool_kernel_size,
            output_size=pool2_output_size
        )
        
        # =====================================================================
        # Step 4: TransConv2
        # =====================================================================
        
        k2 = self.kernel_sizes[1]
        padding2 = k2 // 2
        
        conv1_activation = F.conv_transpose2d(
            unpool2_output,
            self.trans_conv2_weight,
            padding=padding2
        )
        
        conv1_activation = (conv1_activation > threshold).float()
        
        # Record Conv1/TransConv1 activity
        conv1_activity = LayerSpikeActivity(
            layer_name="trans_conv1",
            n_channels=self.conv1_channels,
            spatial_shape=(conv1_activation.shape[2], conv1_activation.shape[3])
        )
        conv1_activity.add_spikes_from_tensor(
            conv1_activation[0],
            timestep,
            feature_offset=0  # First features
        )
        result.layer_activities["trans_conv1"] = conv1_activity
        
        # =====================================================================
        # Step 5: UnPool1
        # =====================================================================
        
        unpool1_output = F.max_unpool2d(
            conv1_activation,
            pool1_indices,
            kernel_size=self.pool_kernel_size,
            output_size=pool1_output_size
        )
        
        # =====================================================================
        # Step 6: TransConv1 → Pixel space
        # =====================================================================
        
        k1 = self.kernel_sizes[0]
        padding1 = k1 // 2
        
        pixel_activation = F.conv_transpose2d(
            unpool1_output,
            self.trans_conv1_weight,
            padding=padding1
        )
        
        # Final pixel mask (aggregate across channels)
        pixel_mask = (pixel_activation.sum(dim=1, keepdim=True) > 0).float()
        
        result.pixel_mask = pixel_mask[0, 0]  # Remove batch and channel dims
        result.compute_bbox()
        
        return result
    
    def unravel_all_spikes(
        self,
        classification_spikes: torch.Tensor,
        pool1_indices: torch.Tensor,
        pool2_indices: torch.Tensor,
        pool1_output_size: Tuple[int, int, int, int],
        pool2_output_size: Tuple[int, int, int, int],
        n_timesteps: int,
        threshold: float = 0.5
    ) -> List[HULKResult]:
        """
        Unravel all classification spikes in a sequence.
        
        Args:
            classification_spikes: Spike tensor of shape (n_timesteps, n_classes, H, W).
            pool1_indices: Pooling indices from Pool1.
            pool2_indices: Pooling indices from Pool2.
            pool1_output_size: Output size for UnPool1.
            pool2_output_size: Output size for UnPool2.
            n_timesteps: Number of timesteps.
            threshold: Activation threshold.
        
        Returns:
            List of HULKResult, one per classification spike.
        
        Raises:
            HULKRuntimeError: If inputs are invalid.
        """
        # Validate classification_spikes
        _validate_tensor(classification_spikes, "classification_spikes")
        
        if classification_spikes.dim() != 4:
            raise HULKRuntimeError(
                f"classification_spikes must be 4D (T, C, H, W), "
                f"got {classification_spikes.dim()}D"
            )
        
        _validate_positive_int(n_timesteps, "n_timesteps")
        
        if classification_spikes.shape[0] != n_timesteps:
            raise HULKRuntimeError(
                f"classification_spikes timesteps ({classification_spikes.shape[0]}) "
                f"doesn't match n_timesteps ({n_timesteps})"
            )
        
        if classification_spikes.shape[1] != self.n_classes:
            raise HULKRuntimeError(
                f"classification_spikes channels ({classification_spikes.shape[1]}) "
                f"doesn't match n_classes ({self.n_classes})"
            )
        
        _validate_range(threshold, 0.0, 1.0, "threshold")
        
        results = []
        
        for t in range(n_timesteps):
            # Find all spikes at this timestep
            spike_map = classification_spikes[t]  # (n_classes, H, W)
            
            # For each class, find all spikes
            for class_id in range(self.n_classes):
                spike_locs = torch.nonzero(spike_map[class_id], as_tuple=False)
                
                # For each spike, unravel it
                for loc in spike_locs:
                    y, x = loc[0].item(), loc[1].item()
                    
                    # Unravel the spike
                    result = self.unravel_spike(
                        spike_location=(x, y),
                        timestep=t,
                        pool1_indices=pool1_indices,
                        pool2_indices=pool2_indices,
                        pool1_output_size=pool1_output_size,
                        pool2_output_size=pool2_output_size,
                        class_id=class_id,
                        threshold=threshold
                    )
                    results.append(result)
        
        return results
    
    def process_to_instances(
        self,
        classification_spikes: torch.Tensor,
        pool1_indices: torch.Tensor,
        pool2_indices: torch.Tensor,
        pool1_output_size: Tuple[int, int, int, int],
        pool2_output_size: Tuple[int, int, int, int],
        n_timesteps: int,
        threshold: float = 0.5
    ) -> List[Instance]:
        """
        Full HULK pipeline: classification spikes → Instances for SMASH.
        
        Args:
            classification_spikes: Spike tensor of shape (n_timesteps, n_classes, H, W).
            pool1_indices: Pooling indices from Pool1.
            pool2_indices: Pooling indices from Pool2.
            pool1_output_size: Output size for UnPool1.
            pool2_output_size: Output size for UnPool2.
            n_timesteps: Number of timesteps.
            threshold: Activation threshold.
        
        Returns:
            List of Instance objects ready for SMASH processing.
            Empty instances (no pixels in mask) are skipped.
        """
        results = self.unravel_all_spikes(
            classification_spikes=classification_spikes,
            pool1_indices=pool1_indices,
            pool2_indices=pool2_indices,
            pool1_output_size=pool1_output_size,
            pool2_output_size=pool2_output_size,
            n_timesteps=n_timesteps,
            threshold=threshold
        )
        
        instances = []
        skipped_count = 0
        
        for i, result in enumerate(results):
            try:
                instance = result.to_instance(
                    instance_id=i,
                    n_features=self.n_features,
                    n_timesteps=n_timesteps
                )
                instances.append(instance)
            except ValueError:
                # Skip empty results (no pixels in mask)
                skipped_count += 1
                continue
        
        return instances
    
    def __repr__(self) -> str:
        return (
            f"HULKDecoder("
            f"n_classes={self.n_classes}, "
            f"conv2_ch={self.conv2_channels}, "
            f"conv1_ch={self.conv1_channels}, "
            f"n_features={self.n_features}, "
            f"kernel_sizes={self.kernel_sizes}, "
            f"pool_kernel_size={self.pool_kernel_size})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "HULKError",
    "HULKConfigError", 
    "HULKRuntimeError",
    "EncoderCompatibilityError",
    # Data structures
    "LayerSpikeActivity",
    "HULKResult",
    # Main decoder
    "HULKDecoder",
]
