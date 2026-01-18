"""
Algorithms for spike-based instance segmentation.

This module provides the post-processing algorithms for converting SNN outputs
into instance segmentations:

- **HULK** (Hierarchical Unravelling of Linked Kernels): Traces classification 
  spikes back to pixel space, recording intermediate spike activity.
  
- **SMASH** (Similarity Matching through Active Spike Hashing): Groups instances
  based on featural-temporal similarity (ASH) and spatial proximity (IoU).

Pipeline:
    Classification Spikes → HULK → (Pixel Mask, Spike Activity) → ASH → SMASH → Objects

Paper Reference:
    Kirkland et al. 2022 - "Unsupervised Spiking Instance Segmentation 
    on Event Data using STDP Features"

Example:
    >>> from spikeseg.algorithms import HULKDecoder, group_instances_to_objects
    >>> 
    >>> # Create HULK decoder from trained encoder
    >>> hulk = HULKDecoder.from_encoder(encoder)
    >>> 
    >>> # Process classification spikes to instances
    >>> instances = hulk.process_to_instances(
    ...     classification_spikes=class_spikes,
    ...     pool1_indices=pool1_idx,
    ...     pool2_indices=pool2_idx,
    ...     pool1_output_size=pool1_size,
    ...     pool2_output_size=pool2_size,
    ...     n_timesteps=10
    ... )
    >>> 
    >>> # Group instances into objects using SMASH
    >>> objects = group_instances_to_objects(instances, smash_threshold=0.1)
"""

# =============================================================================
# SMASH: Similarity Matching through Active Spike Hashing
# =============================================================================

from .smash import (
    # Data structures
    BoundingBox,
    ActiveSpikeHash,
    Instance,
    Object,
    # Functions
    compute_smash_score,
    group_instances_to_objects,
    match_objects_across_sequences,
)

# =============================================================================
# HULK: Hierarchical Unravelling of Linked Kernels
# =============================================================================

from .hulk import (
    # Exceptions
    HULKError,
    HULKConfigError,
    HULKRuntimeError,
    EncoderCompatibilityError,
    # Data structures
    LayerSpikeActivity,
    HULKResult,
    # Main decoder
    HULKDecoder,
)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # =========================================================================
    # SMASH Components
    # =========================================================================
    # Data structures
    "BoundingBox",
    "ActiveSpikeHash", 
    "Instance",
    "Object",
    # Functions
    "compute_smash_score",
    "group_instances_to_objects",
    "match_objects_across_sequences",
    
    # =========================================================================
    # HULK Components
    # =========================================================================
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

