"""
SMASH: Similarity Matching through Active Spike Hashing.

This module implements the instance segmentation algorithms from:
    Kirkland et al. 2022 - "Unsupervised Spiking Instance Segmentation 
    on Event Data using STDP Features"

Components:
    - ASH (Active Spike Hashing): Compress 4D spike data to 2D binary matrix
    - SMASH Score: Combine featural-temporal similarity with spatial proximity
    - Object Grouping: Cluster instances into objects based on SMASH scores

The key insight is that STDP features encode both spatial AND temporal 
information. By hashing spike activity into a (feature, time) matrix,
we can compare instances based on WHEN features fired, not just WHERE.

Algorithm Overview:
    1. HULK: Decode each classification spike back to pixel space
    2. ASH: Hash 4D (x, y, feature, time) → 2D (feature, time) binary matrix
    3. Compute bounding boxes from decoded pixels
    4. SMASH = Jaccard(ASH₁, ASH₂) × IoU(BBox₁, BBox₂)
    5. Group instances with high SMASH scores into objects

Paper Reference:
    "The ASH process then results in a 2D featural-temporal hashing of 
    the spiking activity. This essentially works as a spike train with 
    each feature neuron activity over time being mapped out."

Example:
    >>> from spikeseg.algorithms.smash import (
    ...     ActiveSpikeHash, compute_smash_score, group_instances_to_objects
    ... )
    >>> 
    >>> # Create ASH from spike activity
    >>> ash1 = ActiveSpikeHash.from_spike_times(spike_times_1, n_features=41, n_timesteps=10)
    >>> ash2 = ActiveSpikeHash.from_spike_times(spike_times_2, n_features=41, n_timesteps=10)
    >>> 
    >>> # Compute SMASH score
    >>> score = compute_smash_score(ash1, bbox1, ash2, bbox2)
"""

############################################################################
# Imports
############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate that input is a torch.Tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_float(value: float, name: str) -> None:
    """Validate that a value is a non-negative number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# =============================================================================
# BOUNDING BOX
# =============================================================================

# Bounding box class for instance segmentation
@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box for an instance.
    
    Computed from the min/max x,y coordinates of the decoded spike pixels.
    
    Attributes:
        x_min: Left edge (inclusive).
        y_min: Top edge (inclusive).
        x_max: Right edge (inclusive).
        y_max: Bottom edge (inclusive).
    
    Paper Reference:
        "This in turn calculates a bounding box of each instance S_BB 
        in the pixel space, by taking the max and min value in the 
        x and y coordinates."
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    # Validate bounding box coordinates
    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        # if x_min is greater than x_max, raise an error
        if self.x_min > self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be <= x_max ({self.x_max})"
            )
        # Validate y coordinates
        if self.y_min > self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be <= y_max ({self.y_max})"
            )
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.x_max - self.x_min + 1
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.y_max - self.y_min + 1
    
    @property
    def area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates (x, y)."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )
    
    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> Optional["BoundingBox"]:
        """
        Create bounding box from a binary mask.
        
        Args:
            mask: Binary mask tensor of shape (H, W) or (1, H, W).
                  Nonzero values indicate the instance.
        
        Returns:
            BoundingBox if mask has any nonzero values, None otherwise.
        """
        _validate_tensor(mask, "mask")
        
        # Handle 3D input
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        
        if mask.dim() != 2:
            raise ValueError(f"mask must be 2D or 3D, got {mask.dim()}D")
        
        # Find nonzero coordinates
        nonzero = torch.nonzero(mask, as_tuple=True)
        
        if len(nonzero[0]) == 0:
            return None  # Empty mask
        
        y_coords, x_coords = nonzero
        
        return cls(
            x_min=int(x_coords.min().item()),
            y_min=int(y_coords.min().item()),
            x_max=int(x_coords.max().item()),
            y_max=int(y_coords.max().item())
        )
    
    @classmethod
    def from_coordinates(
        cls, 
        x_coords: torch.Tensor, 
        y_coords: torch.Tensor
    ) -> Optional["BoundingBox"]:
        """
        Create bounding box from coordinate tensors.
        
        Args:
            x_coords: 1D tensor of x coordinates.
            y_coords: 1D tensor of y coordinates (same length).
        
        Returns:
            BoundingBox if coordinates are non-empty, None otherwise.
        """
        _validate_tensor(x_coords, "x_coords")
        _validate_tensor(y_coords, "y_coords")
        
        if len(x_coords) == 0:
            return None
        
        return cls(
            x_min=int(x_coords.min().item()),
            y_min=int(y_coords.min().item()),
            x_max=int(x_coords.max().item()),
            y_max=int(y_coords.max().item())
        )
    
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """
        Compute intersection with another bounding box.
        
        Args:
            other: Another BoundingBox.
        
        Returns:
            Intersection BoundingBox, or None if no overlap.
        """
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        
        if x_min > x_max or y_min > y_max:
            return None  # No intersection
        
        return BoundingBox(x_min, y_min, x_max, y_max)
    
    def union_area(self, other: "BoundingBox") -> int:
        """
        Compute union area with another bounding box.
        
        Args:
            other: Another BoundingBox.
        
        Returns:
            Area of the union (not a single box, just the total area).
        """
        intersection = self.intersection(other)
        intersection_area = intersection.area if intersection else 0
        
        return self.area + other.area - intersection_area
    
    def iou(self, other: "BoundingBox") -> float:
        """
        Compute Intersection over Union (IoU) with another bounding box.
        
        This is the Jaccard index for bounding boxes:
            IoU = |A ∩ B| / |A ∪ B|
        
        Paper Reference:
            "The Proximity Score is calculated with the Jaccard Index 
            J(S_BB, S_BB')"
        
        Args:
            other: Another BoundingBox.
        
        Returns:
            IoU value in [0, 1].
        """
        intersection = self.intersection(other)
        
        if intersection is None:
            return 0.0
        
        union_area = self.union_area(other)
        
        if union_area == 0:
            return 0.0
        
        return intersection.area / union_area
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to [x, y, width, height] format."""
        return (self.x_min, self.y_min, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to [x_min, y_min, x_max, y_max] format."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x_min}:{self.x_max}, y={self.y_min}:{self.y_max})"


# =============================================================================
# ACTIVE SPIKE HASHING (ASH)
# =============================================================================


@dataclass
class ActiveSpikeHash:
    """
    Active Spike Hash (ASH) - 2D binary representation of spike activity.
    
    ASH compresses 4D spike data (x, y, feature, time) into a 2D binary 
    matrix (feature, time). This works because:
        1. Convolution handles spatial translation invariance
        2. Bounding box handles spatial proximity separately
        3. What matters for similarity is WHICH features fired WHEN
    
    The hash is binary: we only care if a feature fired at a timestep,
    not how many times (spike count is more about total activity).
    
    Paper Reference:
        "A further memory reduction to the ASH process is permitted by 
        storing the values as binary terms. As the number of spikes 
        recorded within each map per timestep is more a measure of the 
        total spiking activity, rather than the featural-temporal 
        characteristics we evaluate with the similarity measure."
    
    Attributes:
        hash_matrix: Binary matrix of shape (n_features, n_timesteps).
                    hash_matrix[f, t] = 1 if feature f fired at time t.
        n_features: Total number of features across all layers.
        n_timesteps: Number of timesteps in the sequence.
    
    Example:
        >>> # Feature 3 fired at t=0, feature 5 fired at t=2
        >>> hash_matrix = torch.zeros(10, 5)
        >>> hash_matrix[3, 0] = 1
        >>> hash_matrix[5, 2] = 1
        >>> ash = ActiveSpikeHash(hash_matrix, n_features=10, n_timesteps=5)
    """
    hash_matrix: torch.Tensor
    n_features: int
    n_timesteps: int
    
    def __post_init__(self) -> None:
        """Validate hash matrix."""
        _validate_tensor(self.hash_matrix, "hash_matrix")
        
        if self.hash_matrix.dim() != 2:
            raise ValueError(
                f"hash_matrix must be 2D, got {self.hash_matrix.dim()}D"
            )
        
        if self.hash_matrix.shape != (self.n_features, self.n_timesteps):
            raise ValueError(
                f"hash_matrix shape {self.hash_matrix.shape} doesn't match "
                f"(n_features={self.n_features}, n_timesteps={self.n_timesteps})"
            )
    
    @classmethod
    def from_spike_activity(
        cls,
        spike_times: Dict[int, List[Tuple[int, int, int]]],
        n_features: int,
        n_timesteps: int,
        device: Optional[torch.device] = None
    ) -> "ActiveSpikeHash":
        """
        Create ASH from recorded spike activity.
        
        Args:
            spike_times: Dict mapping feature_id → list of (x, y, t) tuples.
                        Each tuple is a spike at position (x, y) at time t.
            n_features: Total number of features.
            n_timesteps: Total number of timesteps.
            device: Device to create tensor on.
        
        Returns:
            ActiveSpikeHash instance.
        
        Example:
            >>> spike_times = {
            ...     0: [(10, 20, 0), (11, 20, 1)],  # Feature 0 fired twice
            ...     3: [(5, 5, 2)]                   # Feature 3 fired once
            ... }
            >>> ash = ActiveSpikeHash.from_spike_activity(spike_times, 10, 5)
        """
        _validate_positive_int(n_features, "n_features")
        _validate_positive_int(n_timesteps, "n_timesteps")
        
        hash_matrix = torch.zeros(n_features, n_timesteps, device=device)
        
        out_of_range_count = 0
        for feature_id, spikes in spike_times.items():
            if not 0 <= feature_id < n_features:
                raise ValueError(
                    f"feature_id {feature_id} out of range [0, {n_features})"
                )

            for x, y, t in spikes:
                if 0 <= t < n_timesteps:
                    hash_matrix[feature_id, t] = 1.0
                else:
                    out_of_range_count += 1

        if out_of_range_count > 0:
            import warnings
            warnings.warn(
                f"ASH: {out_of_range_count} spikes had timestep outside [0, {n_timesteps}), "
                f"these were excluded from the hash matrix."
            )
        
        return cls(hash_matrix, n_features, n_timesteps)
    
    @classmethod
    def from_layer_spikes(
        cls,
        layer_spikes: List[torch.Tensor],
        device: Optional[torch.device] = None
    ) -> "ActiveSpikeHash":
        """
        Create ASH from a list of layer spike tensors.
        
        This is the typical use case: we have spike tensors from each
        layer of the decoder (Trans-Conv1, Trans-Conv2, etc.) and want
        to hash them into a single ASH.
        
        Args:
            layer_spikes: List of spike tensors, each of shape 
                         (n_timesteps, channels, height, width).
                         Channels from all layers are concatenated.
            device: Device to create tensor on.
        
        Returns:
            ActiveSpikeHash instance.
        
        Example:
            >>> # 3 layers with 4, 36, 1 channels over 10 timesteps
            >>> spikes_conv1 = torch.randint(0, 2, (10, 4, 32, 32)).float()
            >>> spikes_conv2 = torch.randint(0, 2, (10, 36, 16, 16)).float()
            >>> spikes_class = torch.randint(0, 2, (10, 1, 8, 8)).float()
            >>> ash = ActiveSpikeHash.from_layer_spikes([spikes_conv1, spikes_conv2, spikes_class])
            >>> ash.n_features  # 4 + 36 + 1 = 41
        """
        if not layer_spikes:
            raise ValueError("layer_spikes cannot be empty")
        
        # Determine dimensions
        n_timesteps = layer_spikes[0].shape[0]
        
        # Validate all layers have same n_timesteps
        for i, spikes in enumerate(layer_spikes):
            _validate_tensor(spikes, f"layer_spikes[{i}]")
            if spikes.dim() != 4:
                raise ValueError(
                    f"layer_spikes[{i}] must be 4D (T, C, H, W), "
                    f"got {spikes.dim()}D"
                )
            if spikes.shape[0] != n_timesteps:
                raise ValueError(
                    f"All layers must have same n_timesteps. "
                    f"layer_spikes[0] has {n_timesteps}, "
                    f"layer_spikes[{i}] has {spikes.shape[0]}"
                )
        
        # Count total features
        n_features = sum(s.shape[1] for s in layer_spikes)
        
        # Create hash matrix
        if device is None:
            device = layer_spikes[0].device
        
        hash_matrix = torch.zeros(n_features, n_timesteps, device=device)
        
        # Fill hash matrix
        feature_offset = 0
        for spikes in layer_spikes:
            n_channels = spikes.shape[1]
            
            # For each timestep and channel, check if ANY spatial location fired
            # spikes: (T, C, H, W) → any spike in (H, W)?
            for t in range(n_timesteps):
                for c in range(n_channels):
                    # If any spike in this feature map at this time
                    if spikes[t, c].any():
                        hash_matrix[feature_offset + c, t] = 1.0
            
            feature_offset += n_channels
        
        return cls(hash_matrix, n_features, n_timesteps)
    
    @classmethod
    def from_spike_tensor(
        cls,
        spikes: torch.Tensor
    ) -> "ActiveSpikeHash":
        """
        Create ASH from a single spike tensor.
        
        Args:
            spikes: Spike tensor of shape (n_timesteps, n_features, height, width)
                   or (n_timesteps, n_features) if already spatially aggregated.
        
        Returns:
            ActiveSpikeHash instance.
        """
        _validate_tensor(spikes, "spikes")
        
        if spikes.dim() == 4:
            # (T, C, H, W) → check if any spike in each (C, t) pair
            n_timesteps, n_features = spikes.shape[:2]
            
            # Reduce spatial dimensions: any spike in H, W?
            # (T, C, H, W) → (T, C)
            has_spike = spikes.amax(dim=(2, 3)) > 0
            
            # Transpose to (C, T) for hash_matrix
            hash_matrix = has_spike.T.float()
            
        elif spikes.dim() == 2:
            # Already (T, C) or (C, T)?
            # Assume (T, C)
            n_timesteps, n_features = spikes.shape
            hash_matrix = spikes.T.float()
            
        else:
            raise ValueError(
                f"spikes must be 2D or 4D, got {spikes.dim()}D"
            )
        
        # Binarize
        hash_matrix = (hash_matrix > 0).float()
        
        return cls(hash_matrix, n_features, n_timesteps)
    
    def similarity(self, other: "ActiveSpikeHash") -> float:
        """
        Compute Jaccard similarity with another ASH.
        
        Since values are binary, this uses logical AND/OR:
            J(A, B) = |A ∧ B| / |A ∨ B|
        
        Paper Reference:
            "However, due to the ASH process storing binary values of 
            the activities, this equation can be simplified down to a 
            logical calculation performed with only ORs and ANDs."
        
        Args:
            other: Another ActiveSpikeHash.
        
        Returns:
            Jaccard similarity in [0, 1].
        
        Raises:
            ValueError: If dimensions don't match.
        """
        if self.n_features != other.n_features:
            raise ValueError(
                f"n_features mismatch: {self.n_features} vs {other.n_features}"
            )
        if self.n_timesteps != other.n_timesteps:
            raise ValueError(
                f"n_timesteps mismatch: {self.n_timesteps} vs {other.n_timesteps}"
            )
        
        # Ensure both on same device
        other_hash = other.hash_matrix.to(self.hash_matrix.device)
        
        # Binary operations
        intersection = (self.hash_matrix * other_hash).sum()
        union = ((self.hash_matrix + other_hash) > 0).float().sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def to_spike_train_plot(self) -> Tuple[List[int], List[int]]:
        """
        Convert to coordinates for spike raster plot.
        
        Returns:
            Tuple of (times, features) for scatter plot.
        """
        nonzero = torch.nonzero(self.hash_matrix, as_tuple=True)
        features = nonzero[0].tolist()
        times = nonzero[1].tolist()
        return times, features
    
    @property
    def sparsity(self) -> float:
        """Fraction of zeros in the hash matrix."""
        total = self.n_features * self.n_timesteps
        nonzero = self.hash_matrix.sum().item()
        return 1.0 - (nonzero / total) if total > 0 else 1.0
    
    @property
    def n_active(self) -> int:
        """Number of active (feature, time) pairs."""
        return int(self.hash_matrix.sum().item())
    
    def __repr__(self) -> str:
        return (
            f"ActiveSpikeHash(n_features={self.n_features}, "
            f"n_timesteps={self.n_timesteps}, "
            f"n_active={self.n_active}, "
            f"sparsity={self.sparsity:.2%})"
        )


# =============================================================================
# SMASH SCORE
# =============================================================================


def compute_smash_score(
    ash1: ActiveSpikeHash,
    bbox1: BoundingBox,
    ash2: ActiveSpikeHash,
    bbox2: BoundingBox
) -> float:
    """
    Compute SMASH score between two instances.
    
    SMASH = Similarity × Proximity
          = Jaccard(ASH₁, ASH₂) × IoU(BBox₁, BBox₂)
    
    The score is 0 if either:
        - ASH similarity is 0 (no shared featural-temporal activity)
        - Bounding boxes don't overlap (no spatial proximity)
    
    Paper Reference:
        "Multiplication of the similarity score, with the IoU, results 
        in the novel proposed SMASH score for each instance:
        SMASH(Ci, Ci') = J(S_Ci, S_Ci') × J(CiBB, CiBB')"
    
    Args:
        ash1: Active Spike Hash for instance 1.
        bbox1: Bounding box for instance 1.
        ash2: Active Spike Hash for instance 2.
        bbox2: Bounding box for instance 2.
    
    Returns:
        SMASH score in [0, 1].
    
    Example:
        >>> ash1 = ActiveSpikeHash(torch.tensor([[1,0,1],[0,1,0]]).float(), 2, 3)
        >>> ash2 = ActiveSpikeHash(torch.tensor([[1,1,0],[0,1,0]]).float(), 2, 3)
        >>> bbox1 = BoundingBox(0, 0, 10, 10)
        >>> bbox2 = BoundingBox(5, 5, 15, 15)
        >>> score = compute_smash_score(ash1, bbox1, ash2, bbox2)
    """
    # Compute similarity (featural-temporal)
    similarity = ash1.similarity(ash2)
    
    # Compute proximity (spatial)
    proximity = bbox1.iou(bbox2)
    
    # SMASH = Similarity × Proximity
    return similarity * proximity


@dataclass
class Instance:
    """
    A decoded instance from the classification layer.
    
    Contains all information needed for SMASH comparison:
        - instance_id: Unique identifier
        - ash: Active Spike Hash (featural-temporal fingerprint)
        - bbox: Bounding box (spatial location)
        - mask: Optional pixel mask for visualization
    
    Attributes:
        instance_id: Unique identifier for this instance.
        ash: Active Spike Hash for this instance.
        bbox: Bounding box for this instance.
        class_id: Class label (if multi-class).
        mask: Optional binary mask of shape (H, W).
        spike_location: (x, y) location of the classification spike.
    """
    instance_id: int
    ash: ActiveSpikeHash
    bbox: BoundingBox
    class_id: int = 0
    mask: Optional[torch.Tensor] = None
    spike_location: Optional[Tuple[int, int]] = None
    
    def smash_score(self, other: "Instance") -> float:
        """Compute SMASH score with another instance."""
        return compute_smash_score(self.ash, self.bbox, other.ash, other.bbox)
    
    def __repr__(self) -> str:
        return (
            f"Instance(id={self.instance_id}, "
            f"class={self.class_id}, "
            f"bbox={self.bbox}, "
            f"ash={self.ash})"
        )


# =============================================================================
# OBJECT GROUPING (INTRA-SEQUENCE)
# =============================================================================


@dataclass
class Object:
    """
    An object composed of grouped instances.
    
    Multiple instances with high SMASH scores are grouped into a single object.
    The object maintains the combined ASH for inter-sequence matching.
    
    Attributes:
        object_id: Unique identifier for this object.
        instances: List of Instance objects belonging to this object.
        combined_ash: Union of all instance ASHs (for inter-sequence matching).
        combined_bbox: Union bounding box of all instances.
    """
    object_id: int
    instances: List[Instance] = field(default_factory=list)
    combined_ash: Optional[ActiveSpikeHash] = None
    combined_bbox: Optional[BoundingBox] = None
    
    def add_instance(self, instance: Instance) -> None:
        """Add an instance to this object."""
        self.instances.append(instance)
        self._update_combined()
    
    def _update_combined(self) -> None:
        """Update combined ASH and bbox after adding instance."""
        if not self.instances:
            self.combined_ash = None
            self.combined_bbox = None
            return
        
        # Combined bbox: union of all bboxes
        x_min = min(inst.bbox.x_min for inst in self.instances)
        y_min = min(inst.bbox.y_min for inst in self.instances)
        x_max = max(inst.bbox.x_max for inst in self.instances)
        y_max = max(inst.bbox.y_max for inst in self.instances)
        self.combined_bbox = BoundingBox(x_min, y_min, x_max, y_max)
        
        # Combined ASH: OR of all ASHs
        first_ash = self.instances[0].ash
        combined_matrix = torch.zeros_like(first_ash.hash_matrix)
        
        for inst in self.instances:
            combined_matrix = torch.maximum(combined_matrix, inst.ash.hash_matrix)
        
        self.combined_ash = ActiveSpikeHash(
            combined_matrix,
            first_ash.n_features,
            first_ash.n_timesteps
        )
    
    @property
    def n_instances(self) -> int:
        """Number of instances in this object."""
        return len(self.instances)
    
    def __repr__(self) -> str:
        return (
            f"Object(id={self.object_id}, "
            f"n_instances={self.n_instances}, "
            f"bbox={self.combined_bbox})"
        )


def group_instances_to_objects(
    instances: List[Instance],
    smash_threshold: float = 0.0
) -> List[Object]:
    """
    Group instances into objects based on SMASH scores.
    
    Uses a greedy approach: for each instance, find the best matching
    instance and group them together. Instances with SMASH=0 become
    separate objects.
    
    Paper Reference:
        "Once the SMASH score is calculated for each instance, the maximum 
        of each instance is assigned to a class object S_Co:
        S_Co = argmax_Ci(SMASH(Ci, Ci'))"
    
    Args:
        instances: List of Instance objects to group.
        smash_threshold: Minimum SMASH score to consider as same object.
                        Default 0.0 (any overlap = same object).
    
    Returns:
        List of Object instances.
    
    Example:
        >>> instances = [inst1, inst2, inst3, inst4, inst5]
        >>> objects = group_instances_to_objects(instances)
        >>> len(objects)  # Might be fewer than 5 if some grouped
    """
    _validate_non_negative_float(smash_threshold, "smash_threshold")
    
    if not instances:
        return []
    
    n = len(instances)
    
    # Compute all pairwise SMASH scores
    smash_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            score = instances[i].smash_score(instances[j])
            smash_matrix[i, j] = score
            smash_matrix[j, i] = score
    
    # Union-Find for grouping
    parent = list(range(n))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Group instances with SMASH > threshold
    # Use argmax approach from paper
    for i in range(n):
        # Find best match for instance i
        best_j = -1
        best_score = smash_threshold
        
        for j in range(n):
            if i != j and smash_matrix[i, j] > best_score:
                best_score = smash_matrix[i, j]
                best_j = j
        
        if best_j >= 0:
            union(i, best_j)
    
    # Group instances by their root parent
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Create Object instances
    objects = []
    for obj_id, (_, instance_indices) in enumerate(groups.items()):
        obj = Object(object_id=obj_id)
        for idx in instance_indices:
            obj.add_instance(instances[idx])
        objects.append(obj)
    
    return objects

# =============================================================================
# INTER-SEQUENCE MATCHING
# =============================================================================

def match_objects_across_sequences(
    current_objects: List[Object],
    previous_objects: List[Object],
    similarity_threshold: float = 0.0
) -> Dict[int, int]:
    """
    Match objects from current sequence to previous sequence.
    
    Used for tracking objects across time (e.g., across frames/buffers).
    
    Paper Reference:
        "These class objects within the inter-sequence then allow sequence 
        to sequence continuity, thus allowing tracking of objects permitted 
        that they maintain a level of feature similarity."
    
    Args:
        current_objects: Objects detected in current sequence.
        previous_objects: Objects from previous sequence.
        similarity_threshold: Minimum similarity to consider a match.
    
    Returns:
        Dict mapping current_object_id → previous_object_id.
        Objects with no match are not included.
    
    Example:
        >>> matches = match_objects_across_sequences(current_objs, prev_objs)
        >>> for curr_id, prev_id in matches.items():
        ...     print(f"Object {curr_id} matches previous object {prev_id}")
    """
    _validate_non_negative_float(similarity_threshold, "similarity_threshold")
    
    if not current_objects or not previous_objects:
        return {}
    
    matches: Dict[int, int] = {}
    
    for curr_obj in current_objects:
        if curr_obj.combined_ash is None:
            continue
        
        best_match = -1
        best_score = similarity_threshold
        
        for prev_obj in previous_objects:
            if prev_obj.combined_ash is None:
                continue
            
            # Use ASH similarity (spatial proximity less relevant across time)
            score = curr_obj.combined_ash.similarity(prev_obj.combined_ash)
            
            if score > best_score:
                best_score = score
                best_match = prev_obj.object_id
        
        if best_match >= 0:
            matches[curr_obj.object_id] = best_match
    
    return matches
