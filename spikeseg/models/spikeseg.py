"""
SpikeSEG: Complete Spiking Neural Network for Semantic Segmentation.

This module provides the complete SpikeSEG model that combines the encoder
and decoder for end-to-end event-based semantic segmentation.

Paper References:
    Kirkland et al. 2020 - "SpikeSEG: Spiking segmentation via STDP 
    saliency mapping"
    
    Kirkland et al. 2022 - "Unsupervised Spiking Instance Segmentation 
    on Event Data using STDP Features" (HULK-SMASH)
    
    Kirkland et al. 2023 (IGARSS) - "Neuromorphic sensing and processing 
    for space domain awareness"

Architecture:
    Input → [Encoder] → Classification → [Decoder] → Segmentation
    
    Encoder: Conv1 → Pool1 → Conv2 → Pool2 → Conv3
    Decoder: TransConv1 ← UnPool1 ← TransConv2 ← UnPool2 ← TransConv3

Features:
    - STDP-trained encoder features
    - Tied weights in decoder
    - Layer-wise leak for different temporal scales (IGARSS 2023)
    - Instance segmentation support (HULK-SMASH integration)

Example:
    >>> from spikeseg.models import SpikeSEG
    >>> 
    >>> # Create from paper configuration
    >>> model = SpikeSEG.from_paper("igarss2023", n_classes=5)
    >>> 
    >>> # Full forward pass
    >>> model.reset_state()
    >>> for t in range(n_timesteps):
    ...     segmentation, encoder_output = model(input_spikes[t])
    ...     if encoder_output.has_spikes:
    ...         process_detections(encoder_output)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoder import (
    SpikeSEGEncoder,
    EncoderConfig,
    EncoderOutput,
    LayerConfig,
)
from .decoder import SpikeSEGDecoder, DecoderConfig


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


# =============================================================================
# COMPLETE SPIKESEG MODEL
# =============================================================================


class SpikeSEG(nn.Module):
    """
    Complete SpikeSEG model with encoder and decoder.
    
    Combines the encoder and decoder for full semantic segmentation.
    The decoder uses tied weights from the encoder.
    
    Architecture:
        Input → [Encoder] → Classification → [Decoder] → Segmentation
    
    The encoder extracts hierarchical features using STDP-trained
    spiking convolutions. The decoder maps classification spikes back
    to pixel space using transposed convolutions with tied weights.
    
    Attributes:
        config: Encoder configuration.
        encoder: SpikeSEG encoder network.
        decoder: SpikeSEG decoder network (lazy initialization).
    
    Example:
        >>> # Create model from paper
        >>> model = SpikeSEG.from_paper("igarss2023", n_classes=1)
        >>> 
        >>> # Process event sequence
        >>> model.reset_state()
        >>> for t, events in enumerate(event_stream):
        ...     seg, enc_out = model(events)
        ...     if enc_out.has_spikes:
        ...         print(f"t={t}: {enc_out.n_classification_spikes} detections")
        >>> 
        >>> # Or use encoder/decoder separately
        >>> enc_out = model.encode(events)
        >>> seg = model.decode(enc_out)
    """
    
    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize SpikeSEG model.
        
        Args:
            config: Encoder configuration. If None, uses default IGARSS 2023.
            decoder_config: Decoder configuration. If None, uses default.
            device: Device for tensors.
        """
        super().__init__()
        
        if config is None:
            config = EncoderConfig.from_paper("igarss2023")
        
        self.config = config
        self._decoder_config = decoder_config
        
        # Build encoder
        self.encoder = SpikeSEGEncoder(config)
        
        # Decoder is created lazily to get proper sizes
        self._decoder: Optional[SpikeSEGDecoder] = None
        
        # Move to device
        if device is not None:
            self.to(device)
    
    @classmethod
    def from_paper(
        cls,
        paper: str,
        n_classes: int = 1,
        decoder_config: Optional[DecoderConfig] = None,
        device: Optional[torch.device] = None
    ) -> "SpikeSEG":
        """
        Create model from paper configuration.
        
        Args:
            paper: Paper identifier:
                - "igarss2023": Space domain awareness (LIF neurons)
                - "hulk2022" or "hulksmash": Instance segmentation (IF neurons)
                - "spikeseg2020" or "spikeseg": Original SpikeSEG
            n_classes: Number of output classes.
            decoder_config: Optional decoder configuration.
            device: Device for tensors.
        
        Returns:
            SpikeSEG model with paper parameters.
        
        Example:
            >>> # Space domain awareness (satellite detection)
            >>> model = SpikeSEG.from_paper("igarss2023", n_classes=1)
            >>> 
            >>> # Multi-class face detection
            >>> model = SpikeSEG.from_paper("hulk2022", n_classes=5)
        """
        config = EncoderConfig.from_paper(paper, n_classes=n_classes)
        return cls(config, decoder_config=decoder_config, device=device)
    
    @classmethod
    def from_config(
        cls,
        conv1_channels: int = 4,
        conv2_channels: int = 36,
        n_classes: int = 1,
        kernel_sizes: Tuple[int, int, int] = (5, 5, 7),
        thresholds: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        leaks: Tuple[float, float, float] = (9.0, 1.0, 0.0),
        device: Optional[torch.device] = None
    ) -> "SpikeSEG":
        """
        Create model from explicit configuration.
        
        Convenience method for custom configurations.
        
        Args:
            conv1_channels: Number of Conv1 features.
            conv2_channels: Number of Conv2 features (STDP-learned).
            n_classes: Number of output classes.
            kernel_sizes: Kernel sizes for (Conv1, Conv2, Conv3).
            thresholds: Spike thresholds for (Conv1, Conv2, Conv3).
            leaks: Leak factors for (Conv1, Conv2, Conv3).
            device: Device for tensors.
        
        Returns:
            SpikeSEG model with custom configuration.
        
        Example:
            >>> model = SpikeSEG.from_config(
            ...     conv2_channels=64,
            ...     n_classes=10,
            ...     leaks=(5.0, 0.5, 0.0)
            ... )
        """
        k1, k2, k3 = kernel_sizes
        t1, t2, t3 = thresholds
        l1, l2, l3 = leaks
        
        config = EncoderConfig(
            input_channels=1,
            conv1=LayerConfig(
                out_channels=conv1_channels,
                kernel_size=k1,
                threshold=t1,
                leak=l1
            ),
            conv2=LayerConfig(
                out_channels=conv2_channels,
                kernel_size=k2,
                threshold=t2,
                leak=l2
            ),
            conv3=LayerConfig(
                out_channels=n_classes,
                kernel_size=k3,
                threshold=t3,
                leak=l3
            )
        )
        
        return cls(config, device=device)
    
    def _create_decoder(self) -> None:
        """Create decoder from encoder (lazy initialization)."""
        if self._decoder is None:
            self._decoder = SpikeSEGDecoder.from_encoder(
                self.encoder,
                config=self._decoder_config
            )
    
    @property
    def decoder(self) -> SpikeSEGDecoder:
        """Get decoder (creates if needed)."""
        self._create_decoder()
        return self._decoder
    
    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self.config.conv3.out_channels
    
    def reset_state(self) -> None:
        """Reset all stateful components (membrane potentials, delays)."""
        self.encoder.reset_state()
        if self._decoder is not None:
            self._decoder.reset_state()
    
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass through encoder only.
        
        Args:
            x: Input spikes, shape (batch, channels, H, W).
        
        Returns:
            EncoderOutput with classification spikes and pooling indices.
        """
        return self.encoder(x)
    
    def decode(self, encoder_output: EncoderOutput) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            encoder_output: Output from encoder.
        
        Returns:
            Saliency/segmentation map, shape (batch, input_channels, H, W).
        """
        self._create_decoder()
        
        return self._decoder(
            classification_spikes=encoder_output.classification_spikes,
            pool1_indices=encoder_output.pooling_indices.pool1_indices,
            pool2_indices=encoder_output.pooling_indices.pool2_indices,
            pool1_output_size=encoder_output.pooling_indices.pool1_output_size,
            pool2_output_size=encoder_output.pooling_indices.pool2_output_size
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_encoder_output: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, EncoderOutput]]:
        """
        Full forward pass through encoder and decoder.
        
        Args:
            x: Input spikes, shape (batch, channels, H, W).
            return_encoder_output: Also return encoder output for
                                  instance segmentation or analysis.
        
        Returns:
            If return_encoder_output=True:
                Tuple of (segmentation, encoder_output).
            Otherwise:
                Just segmentation tensor.
        
        Example:
            >>> # Get both outputs
            >>> seg, enc_out = model(events)
            >>> 
            >>> # Get just segmentation
            >>> seg = model(events, return_encoder_output=False)
        """
        _validate_4d_tensor(x, "input")
        
        # Encode
        encoder_output = self.encode(x)
        
        # Decode
        segmentation = self.decode(encoder_output)
        
        if return_encoder_output:
            return segmentation, encoder_output
        return segmentation
    
    def get_layer_weights(self) -> dict:
        """
        Get all layer weights (for STDP training or analysis).
        
        Returns:
            Dict with 'conv1', 'conv2', 'conv3' weight tensors.
        """
        return {
            'conv1': self.encoder.conv1.conv.weight,
            'conv2': self.encoder.conv2.conv.weight,
            'conv3': self.encoder.conv3.conv.weight,
        }
    
    def freeze_layer(self, layer_name: str) -> None:
        """
        Freeze a layer's weights (disable gradient computation).
        
        Args:
            layer_name: 'conv1', 'conv2', or 'conv3'.
        """
        layer = getattr(self.encoder, layer_name)
        for param in layer.parameters():
            param.requires_grad = False
    
    def unfreeze_layer(self, layer_name: str) -> None:
        """
        Unfreeze a layer's weights (enable gradient computation).
        
        Args:
            layer_name: 'conv1', 'conv2', or 'conv3'.
        """
        layer = getattr(self.encoder, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
    
    def __repr__(self) -> str:
        decoder_status = 'initialized' if self._decoder else 'lazy'
        return (
            f"SpikeSEG(\n"
            f"  encoder: {self.config.input_channels}→"
            f"{self.config.conv1.out_channels}→"
            f"{self.config.conv2.out_channels}→"
            f"{self.config.conv3.out_channels}\n"
            f"  decoder: {decoder_status}\n"
            f")"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SpikeSEG",
]