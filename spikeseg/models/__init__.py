"""
Complete model architectures for SpikeSEG.

This module provides the encoder, decoder, and complete SpikeSEG models
for spiking neural network-based semantic segmentation.

Models:
    - **SpikeSEGEncoder**: 3-layer hierarchical spiking encoder
    - **SpikeSEGDecoder**: Decoder with tied weights for saliency maps
    - **SpikeSEG**: Complete encoder-decoder model

Architecture (from Kirkland et al. papers):

    ENCODER:
        Input → Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Classification
               (5×5)   (2×2)   (5×5)   (2×2)   (7×7)
               4 feat  store   36 feat store   C classes
                       indices         indices
    
    DECODER:
        Conv3 → UnPool2 → TransConv2 → UnPool1 → TransConv1 → Pixel Mask
                (use idx)  (tied wts)  (use idx)  (tied wts)

Paper References:
    - Kirkland et al. 2020: SpikeSEG
    - Kirkland et al. 2022: HULK-SMASH instance segmentation
    - Kirkland et al. 2023 (IGARSS): Space domain awareness

Example:
    >>> from spikeseg.models import SpikeSEG
    >>> 
    >>> # Create from paper configuration
    >>> model = SpikeSEG.from_paper("igarss2023", n_classes=1)
    >>> 
    >>> # Process event sequence
    >>> model.reset_state()
    >>> for events in event_stream:
    ...     segmentation, encoder_output = model(events)
    ...     if encoder_output.has_spikes:
    ...         # Process classification spikes with HULK-SMASH
    ...         pass
"""

from __future__ import annotations

# Encoder
from .encoder import (
    # Exceptions
    EncoderError,
    EncoderConfigError,
    EncoderRuntimeError,
    # Data classes
    LayerConfig,
    PoolingIndices,
    EncoderOutput,
    EncoderConfig,
    # Main encoder
    SpikeSEGEncoder,
    # Factory
    create_encoder,
)

# Decoder
from .decoder import (
    # Exceptions  
    DecoderError,
    DecoderConfigError,
    DecoderRuntimeError,
    EncoderCompatibilityError,
    # Data classes
    DecoderConfig,
    # Main decoder
    SpikeSEGDecoder,
)

# Complete model
from .spikeseg import SpikeSEG


__all__ = [
    # Encoder exceptions
    "EncoderError",
    "EncoderConfigError", 
    "EncoderRuntimeError",
    # Encoder data classes
    "LayerConfig",
    "PoolingIndices",
    "EncoderOutput",
    "EncoderConfig",
    # Encoder class
    "SpikeSEGEncoder",
    # Encoder factory
    "create_encoder",
    
    # Decoder exceptions
    "DecoderError",
    "DecoderConfigError",
    "DecoderRuntimeError",
    "EncoderCompatibilityError",
    # Decoder data classes
    "DecoderConfig",
    # Decoder class
    "SpikeSEGDecoder",
    
    # Complete model
    "SpikeSEG",
]

