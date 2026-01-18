"""
Event Camera Datasets for SpikeSEG.

This module provides dataset classes for:
    - EBSSA: Event-Based Space Situational Awareness (satellite detection)
    - N-MNIST: Neuromorphic MNIST (for testing/benchmarking)
    - Generic event file loading (.mat, .h5, .npy, .aedat)

Event Representation:
    Raw events are stored as (x, y, polarity, timestamp) tuples.
    For SNN processing, we convert to voxel grids: (T, C, H, W)
    where T=timesteps, C=polarity channels (1 or 2), H/W=spatial dims.

Paper References:
    EBSSA Dataset:
        Afshar et al. 2020 - "Event-Based Object Detection and Tracking 
        for Space Situational Awareness"
        - 84 labelled recordings, 153 unlabelled
        - Resolution: 240×180 (ATIS/DAVIS sensors)
        - Events: [x, y, p, t] format in .mat files
    
    IGARSS 2023 Processing:
        Kirkland et al. 2023 - "Neuromorphic sensing and processing for 
        space domain awareness"
        - Temporal buffering into 20 timesteps
        - 10 parsed event streams per buffer

Example:
    >>> from spikeseg.data import EBSSADataset, create_dataloader
    >>> 
    >>> # Create dataset
    >>> dataset = EBSSADataset(
    ...     root="./data/EBSSA",
    ...     split="train",
    ...     n_timesteps=10,
    ...     height=128,
    ...     width=128
    ... )
    >>> 
    >>> # Create dataloader
    >>> loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    >>> 
    >>> for events, labels in loader:
    ...     # events: (B, T, C, H, W)
    ...     output = model(events)

Author: SpikeSEG Team
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from abc import ABC, abstractmethod

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Event array: structured array with fields (x, y, p, t) or (N, 4) array
EventArray = Union[np.ndarray, Dict[str, np.ndarray]]

# Voxel grid: (T, C, H, W) or (T, H, W) tensor
VoxelGrid = torch.Tensor


# =============================================================================
# CONSTANTS
# =============================================================================

# EBSSA sensor resolutions
EBSSA_ATIS_RESOLUTION = (304, 240)  # ATIS sensor
EBSSA_DAVIS_RESOLUTION = (240, 180)  # DAVIS240C sensor
EBSSA_DEFAULT_RESOLUTION = (240, 180)

# N-MNIST resolution
NMNIST_RESOLUTION = (34, 34)


# =============================================================================
# EVENT REPRESENTATION UTILITIES
# =============================================================================


@dataclass
class EventData:
    """
    Container for event camera data.
    
    Attributes:
        x: X coordinates (column), shape (N,)
        y: Y coordinates (row), shape (N,)
        p: Polarity (0 or 1, or -1/+1), shape (N,)
        t: Timestamps in microseconds, shape (N,)
        height: Sensor height in pixels
        width: Sensor width in pixels
    """
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray
    height: int
    width: int
    
    def __post_init__(self):
        """Validate event data."""
        n = len(self.x)
        if len(self.y) != n:
            raise ValueError(f"y length {len(self.y)} != x length {n}")
        if len(self.p) != n:
            raise ValueError(f"p length {len(self.p)} != x length {n}")
        if len(self.t) != n:
            raise ValueError(f"t length {len(self.t)} != x length {n}")
    
    def __len__(self) -> int:
        return len(self.x)
    
    @property
    def n_events(self) -> int:
        return len(self.x)
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if len(self.t) == 0:
            return 0.0
        return (self.t[-1] - self.t[0]) / 1e6
    
    @property
    def event_rate(self) -> float:
        """Events per second."""
        d = self.duration
        return self.n_events / d if d > 0 else 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to (N, 4) array [x, y, p, t]."""
        return np.column_stack([self.x, self.y, self.p, self.t])
    
    @classmethod
    def from_array(
        cls, 
        events: np.ndarray, 
        height: int, 
        width: int
    ) -> "EventData":
        """Create from (N, 4) array."""
        return cls(
            x=events[:, 0].astype(np.int32),
            y=events[:, 1].astype(np.int32),
            p=events[:, 2].astype(np.int8),
            t=events[:, 3].astype(np.int64),
            height=height,
            width=width
        )
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, np.ndarray],
        height: int,
        width: int
    ) -> "EventData":
        """Create from dictionary with x, y, p, t keys."""
        return cls(
            x=data['x'].astype(np.int32),
            y=data['y'].astype(np.int32),
            p=data['p'].astype(np.int8),
            t=data['t'].astype(np.int64),
            height=height,
            width=width
        )
    
    def filter_by_time(
        self, 
        t_start: float, 
        t_end: float
    ) -> "EventData":
        """Filter events within time range [t_start, t_end] in microseconds."""
        mask = (self.t >= t_start) & (self.t < t_end)
        return EventData(
            x=self.x[mask],
            y=self.y[mask],
            p=self.p[mask],
            t=self.t[mask],
            height=self.height,
            width=self.width
        )
    
    def filter_by_region(
        self,
        x_min: int, x_max: int,
        y_min: int, y_max: int
    ) -> "EventData":
        """Filter events within spatial region."""
        mask = (
            (self.x >= x_min) & (self.x < x_max) &
            (self.y >= y_min) & (self.y < y_max)
        )
        return EventData(
            x=self.x[mask] - x_min,
            y=self.y[mask] - y_min,
            p=self.p[mask],
            t=self.t[mask],
            height=y_max - y_min,
            width=x_max - x_min
        )


# =============================================================================
# EVENT TO VOXEL GRID CONVERSION
# =============================================================================


def events_to_voxel_grid(
    events: EventData,
    n_timesteps: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
    normalize: bool = True,
    polarity_channels: bool = True
) -> torch.Tensor:
    """
    Convert events to a voxel grid (temporal binning).
    
    This is the standard representation for SNNs - events are binned
    into discrete timesteps creating a (T, C, H, W) tensor.
    
    Paper Reference (IGARSS 2023):
        "The input events are fed into the network via a temporal 
        buffering stage... the buffered data is parsed into 20 steps."
    
    Args:
        events: EventData object with (x, y, p, t) events.
        n_timesteps: Number of temporal bins.
        height: Output height (default: events.height).
        width: Output width (default: events.width).
        normalize: Normalize event counts to [0, 1].
        polarity_channels: If True, separate channels for pos/neg polarity.
                          Shape: (T, 2, H, W). If False: (T, 1, H, W).
    
    Returns:
        Voxel grid tensor of shape (T, C, H, W).
    
    Example:
        >>> events = load_events("recording.mat")
        >>> voxel = events_to_voxel_grid(events, n_timesteps=10)
        >>> voxel.shape
        torch.Size([10, 2, 180, 240])
    """
    if height is None:
        height = events.height
    if width is None:
        width = events.width
    
    n_channels = 2 if polarity_channels else 1
    voxel = torch.zeros(n_timesteps, n_channels, height, width, dtype=torch.float32)
    
    if len(events) == 0:
        return voxel
    
    # Normalize timestamps to [0, n_timesteps)
    t = events.t.astype(np.float64)
    t_min, t_max = t.min(), t.max()
    
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (n_timesteps - 1e-6)
    else:
        t_norm = np.zeros_like(t)
    
    t_idx = t_norm.astype(np.int32)
    t_idx = np.clip(t_idx, 0, n_timesteps - 1)
    
    # Clip spatial coordinates
    x = np.clip(events.x, 0, width - 1).astype(np.int32)
    y = np.clip(events.y, 0, height - 1).astype(np.int32)
    
    # Determine polarity channel
    p = events.p
    if polarity_channels:
        # Channel 0: positive (ON), Channel 1: negative (OFF)
        # Handle various polarity conventions: (0,1), (-1,1), (1,2), etc.
        p_min, p_max = p.min(), p.max()
        if p_min < 0:
            # Convention: -1/+1 -> map to 0/1
            c_idx = ((p + 1) // 2).astype(np.int32)
        elif p_min == 0 and p_max <= 1:
            # Convention: 0/1 -> use directly
            c_idx = p.astype(np.int32)
        else:
            # Convention: 1/2 or other -> normalize to 0/1
            c_idx = (p - p_min).astype(np.int32)
        # Ensure indices are valid (0 or 1)
        c_idx = np.clip(c_idx, 0, 1)
    else:
        c_idx = np.zeros_like(x)
    
    # Accumulate events into voxel grid
    np.add.at(
        voxel.numpy(),
        (t_idx, c_idx, y, x),
        1.0
    )
    
    if normalize:
        # Normalize per-timestep to [0, 1]
        for ti in range(n_timesteps):
            max_val = voxel[ti].max()
            if max_val > 0:
                voxel[ti] /= max_val
    
    return voxel


def events_to_frame(
    events: EventData,
    height: Optional[int] = None,
    width: Optional[int] = None,
    accumulate: bool = True,
    polarity_channels: bool = True
) -> torch.Tensor:
    """
    Convert events to a single frame (accumulate all events).
    
    Args:
        events: EventData object.
        height: Output height.
        width: Output width.
        accumulate: If True, count events. If False, binary (event/no event).
        polarity_channels: Separate channels for polarities.
    
    Returns:
        Frame tensor of shape (C, H, W).
    """
    voxel = events_to_voxel_grid(
        events, n_timesteps=1, height=height, width=width,
        normalize=False, polarity_channels=polarity_channels
    )
    frame = voxel[0]  # (C, H, W)
    
    if not accumulate:
        frame = (frame > 0).float()
    
    return frame


def events_to_time_surface(
    events: EventData,
    height: Optional[int] = None,
    width: Optional[int] = None,
    tau: float = 0.1,
    polarity_channels: bool = True
) -> torch.Tensor:
    """
    Convert events to a time surface (exponential decay).
    
    Recent events have higher values, older events decay.
    
    Args:
        events: EventData object.
        height: Output height.
        width: Output width.
        tau: Decay time constant (fraction of total duration).
        polarity_channels: Separate channels for polarities.
    
    Returns:
        Time surface tensor of shape (C, H, W).
    """
    if height is None:
        height = events.height
    if width is None:
        width = events.width
    
    n_channels = 2 if polarity_channels else 1
    surface = torch.zeros(n_channels, height, width, dtype=torch.float32)
    
    if len(events) == 0:
        return surface
    
    t = events.t.astype(np.float64)
    t_max = t.max()
    duration = t_max - t.min()
    
    if duration == 0:
        return surface
    
    # Exponential decay from most recent time
    decay = np.exp(-(t_max - t) / (duration * tau))
    
    x = np.clip(events.x, 0, width - 1).astype(np.int32)
    y = np.clip(events.y, 0, height - 1).astype(np.int32)
    
    p = events.p
    if polarity_channels:
        if p.min() < 0:
            c_idx = ((p + 1) // 2).astype(np.int32)
        else:
            c_idx = p.astype(np.int32)
    else:
        c_idx = np.zeros_like(x)
    
    # Take maximum decay value for each pixel
    for i in range(len(events)):
        c, yi, xi = c_idx[i], y[i], x[i]
        surface[c, yi, xi] = max(surface[c, yi, xi], decay[i])
    
    return surface


# =============================================================================
# FILE LOADERS
# =============================================================================


def load_events_mat(
    filepath: Union[str, Path],
    height: int = 180,
    width: int = 240
) -> EventData:
    """
    Load events from MATLAB .mat file (EBSSA format).
    
    EBSSA .mat files contain a TD structure with fields:
        TD.x, TD.y, TD.p, TD.ts
    
    Args:
        filepath: Path to .mat file.
        height: Sensor height.
        width: Sensor width.
    
    Returns:
        EventData object.
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy required for .mat file loading: pip install scipy")
    
    filepath = Path(filepath)
    
    try:
        # Try loading as MATLAB v7.3 (HDF5 format)
        import h5py
        with h5py.File(filepath, 'r') as f:
            if 'TD' in f:
                td = f['TD']
                x = np.array(td['x']).flatten().astype(np.int32)
                y = np.array(td['y']).flatten().astype(np.int32)
                p = np.array(td['p']).flatten().astype(np.int8)
                t = np.array(td['ts']).flatten().astype(np.int64)
            else:
                # Try direct field access
                x = np.array(f['x']).flatten().astype(np.int32)
                y = np.array(f['y']).flatten().astype(np.int32)
                p = np.array(f['p']).flatten().astype(np.int8)
                t = np.array(f['ts'] if 'ts' in f else f['t']).flatten().astype(np.int64)
    except (OSError, KeyError, ImportError):
        # Try loading as older MATLAB format
        mat = sio.loadmat(filepath)
        
        if 'TD' in mat:
            td = mat['TD']
            # Handle structured array
            if hasattr(td, 'dtype') and td.dtype.names:
                x = td['x'][0, 0].flatten().astype(np.int32)
                y = td['y'][0, 0].flatten().astype(np.int32)
                p = td['p'][0, 0].flatten().astype(np.int8)
                t = td['ts'][0, 0].flatten().astype(np.int64)
            else:
                x = td[0, 0]['x'].flatten().astype(np.int32)
                y = td[0, 0]['y'].flatten().astype(np.int32)
                p = td[0, 0]['p'].flatten().astype(np.int8)
                t = td[0, 0]['ts'].flatten().astype(np.int64)
        else:
            # Direct field access
            x = mat['x'].flatten().astype(np.int32)
            y = mat['y'].flatten().astype(np.int32)
            p = mat['p'].flatten().astype(np.int8)
            t_key = 'ts' if 'ts' in mat else 't'
            t = mat[t_key].flatten().astype(np.int64)
    
    return EventData(x=x, y=y, p=p, t=t, height=height, width=width)


def load_events_h5(
    filepath: Union[str, Path],
    recording_name: Optional[str] = None,
    height: int = 180,
    width: int = 240
) -> EventData:
    """
    Load events from HDF5 file.
    
    Args:
        filepath: Path to .h5 file.
        recording_name: Specific recording to load (for combined files).
        height: Sensor height.
        width: Sensor width.
    
    Returns:
        EventData object.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for .h5 file loading: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        if recording_name is not None:
            grp = f[recording_name]
        else:
            # Use first group or root
            keys = list(f.keys())
            if len(keys) == 1:
                grp = f[keys[0]]
            else:
                grp = f
        
        # Try common field names
        x_key = 'x' if 'x' in grp else 'X'
        y_key = 'y' if 'y' in grp else 'Y'
        p_key = 'p' if 'p' in grp else ('pol' if 'pol' in grp else 'polarity')
        t_key = 't' if 't' in grp else ('ts' if 'ts' in grp else 'timestamp')
        
        x = np.array(grp[x_key]).flatten().astype(np.int32)
        y = np.array(grp[y_key]).flatten().astype(np.int32)
        p = np.array(grp[p_key]).flatten().astype(np.int8)
        t = np.array(grp[t_key]).flatten().astype(np.int64)
    
    return EventData(x=x, y=y, p=p, t=t, height=height, width=width)


def load_events_npy(
    filepath: Union[str, Path],
    height: int = 180,
    width: int = 240
) -> EventData:
    """
    Load events from NumPy .npy file.
    
    Expected format: (N, 4) array with [x, y, p, t] columns.
    
    Args:
        filepath: Path to .npy file.
        height: Sensor height.
        width: Sensor width.
    
    Returns:
        EventData object.
    """
    events = np.load(filepath)
    
    if events.ndim == 2 and events.shape[1] >= 4:
        return EventData.from_array(events[:, :4], height, width)
    else:
        raise ValueError(f"Expected (N, 4) array, got shape {events.shape}")


def load_events(
    filepath: Union[str, Path],
    height: int = 180,
    width: int = 240
) -> EventData:
    """
    Load events from file (auto-detect format).
    
    Supported formats: .mat, .h5, .hdf5, .npy
    
    Args:
        filepath: Path to event file.
        height: Sensor height.
        width: Sensor width.
    
    Returns:
        EventData object.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.mat':
        return load_events_mat(filepath, height, width)
    elif suffix in ['.h5', '.hdf5']:
        return load_events_h5(filepath, height=height, width=width)
    elif suffix == '.npy':
        return load_events_npy(filepath, height, width)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# =============================================================================
# LABEL LOADERS
# =============================================================================


def load_labels_mat(
    filepath: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load labels from MATLAB .mat file (EBSSA format).
    
    EBSSA label files contain bounding box annotations.
    
    Returns:
        Dictionary with label data.
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy required for .mat file loading")
    
    mat = sio.loadmat(filepath, squeeze_me=True)
    
    # Extract label fields (format varies)
    labels = {}
    for key in mat.keys():
        if not key.startswith('_'):
            labels[key] = mat[key]
    
    return labels


# =============================================================================
# DATA AUGMENTATION
# =============================================================================


class EventAugmentation:
    """
    Data augmentation for event data.
    
    Augmentations preserve temporal structure while
    varying spatial appearance.
    """
    
    def __init__(
        self,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        flip_polarity: bool = False,
        random_crop: Optional[Tuple[int, int]] = None,
        noise_rate: float = 0.0,
        drop_rate: float = 0.0
    ):
        """
        Initialize augmentation.
        
        Args:
            flip_horizontal: Random horizontal flip.
            flip_vertical: Random vertical flip.
            flip_polarity: Random polarity inversion.
            random_crop: Crop size (height, width) or None.
            noise_rate: Rate of noise events to add.
            drop_rate: Rate of events to drop.
        """
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.flip_polarity = flip_polarity
        self.random_crop = random_crop
        self.noise_rate = noise_rate
        self.drop_rate = drop_rate
    
    def __call__(self, events: EventData) -> EventData:
        """Apply augmentations."""
        x = events.x.copy()
        y = events.y.copy()
        p = events.p.copy()
        t = events.t.copy()
        h, w = events.height, events.width
        
        # Horizontal flip
        if self.flip_horizontal and np.random.rand() > 0.5:
            x = w - 1 - x
        
        # Vertical flip
        if self.flip_vertical and np.random.rand() > 0.5:
            y = h - 1 - y
        
        # Polarity flip
        if self.flip_polarity and np.random.rand() > 0.5:
            p = 1 - p  # Assumes polarity is 0/1
        
        # Random crop
        if self.random_crop is not None:
            crop_h, crop_w = self.random_crop
            if crop_h < h and crop_w < w:
                y_off = np.random.randint(0, h - crop_h)
                x_off = np.random.randint(0, w - crop_w)
                
                mask = (
                    (x >= x_off) & (x < x_off + crop_w) &
                    (y >= y_off) & (y < y_off + crop_h)
                )
                x = x[mask] - x_off
                y = y[mask] - y_off
                p = p[mask]
                t = t[mask]
                h, w = crop_h, crop_w
        
        # Drop events
        if self.drop_rate > 0:
            keep_mask = np.random.rand(len(x)) > self.drop_rate
            x = x[keep_mask]
            y = y[keep_mask]
            p = p[keep_mask]
            t = t[keep_mask]
        
        # Add noise events
        if self.noise_rate > 0 and len(t) > 0:
            n_noise = int(len(x) * self.noise_rate)
            if n_noise > 0:
                noise_x = np.random.randint(0, w, n_noise).astype(np.int32)
                noise_y = np.random.randint(0, h, n_noise).astype(np.int32)
                noise_p = np.random.randint(0, 2, n_noise).astype(np.int8)
                noise_t = np.random.randint(t.min(), t.max() + 1, n_noise).astype(np.int64)
                
                x = np.concatenate([x, noise_x])
                y = np.concatenate([y, noise_y])
                p = np.concatenate([p, noise_p])
                t = np.concatenate([t, noise_t])
                
                # Sort by time
                sort_idx = np.argsort(t)
                x, y, p, t = x[sort_idx], y[sort_idx], p[sort_idx], t[sort_idx]
        
        return EventData(x=x, y=y, p=p, t=t, height=h, width=w)


# =============================================================================
# BASE DATASET CLASS
# =============================================================================


class EventDataset(Dataset, ABC):
    """
    Abstract base class for event camera datasets.
    
    Subclasses must implement:
        - __len__: Return dataset size
        - _load_sample: Load raw events and labels for index
    """
    
    def __init__(
        self,
        n_timesteps: int = 10,
        height: Optional[int] = None,
        width: Optional[int] = None,
        normalize: bool = True,
        polarity_channels: bool = True,
        transform: Optional[Callable] = None,
        augmentation: Optional[EventAugmentation] = None
    ):
        """
        Initialize dataset.
        
        Args:
            n_timesteps: Number of temporal bins for voxel grid.
            height: Output height (None = use sensor default).
            width: Output width (None = use sensor default).
            normalize: Normalize voxel values to [0, 1].
            polarity_channels: Use separate channels for polarities.
            transform: Optional transform to apply to voxel grid.
            augmentation: Optional event augmentation.
        """
        self.n_timesteps = n_timesteps
        self.height = height
        self.width = width
        self.normalize = normalize
        self.polarity_channels = polarity_channels
        self.transform = transform
        self.augmentation = augmentation
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def _load_sample(self, index: int) -> Tuple[EventData, Any]:
        """
        Load events and label for a sample.
        
        Args:
            index: Sample index.
        
        Returns:
            Tuple of (EventData, label).
        """
        pass
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Get a sample.
        
        Args:
            index: Sample index.
        
        Returns:
            Tuple of (voxel_grid, label).
            voxel_grid: (T, C, H, W) tensor.
        """
        # Try to load sample, with fallback to next valid sample
        max_attempts = min(10, len(self.recordings))
        for attempt in range(max_attempts):
            try:
                actual_index = (index + attempt) % len(self.recordings)
                events, label = self._load_sample(actual_index)
                
                # Skip empty events
                if len(events) == 0:
                    continue
                
                # Apply augmentation
                if self.augmentation is not None:
                    events = self.augmentation(events)
                
                # Convert to voxel grid
                voxel = events_to_voxel_grid(
                    events,
                    n_timesteps=self.n_timesteps,
                    height=self.height or events.height,
                    width=self.width or events.width,
                    normalize=self.normalize,
                    polarity_channels=self.polarity_channels
                )
                
                # Apply transform
                if self.transform is not None:
                    voxel = self.transform(voxel)
                
                return voxel, label
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Last attempt - raise error
                    raise RuntimeError(f"Failed to load any valid sample after {max_attempts} attempts: {e}")
                continue
        
        raise RuntimeError("No valid samples found")


# =============================================================================
# EBSSA DATASET
# =============================================================================


class EBSSADataset(EventDataset):
    """
    EBSSA (Event-Based Space Situational Awareness) Dataset.
    
    Dataset of event camera recordings of satellites, planets, and stars.
    Used for satellite detection and tracking in space domain awareness.
    
    Paper Reference:
        Afshar et al. 2020 - "Event-Based Object Detection and Tracking 
        for Space Situational Awareness"
        
        - 84 labelled recordings
        - 153 unlabelled recordings
        - Sensors: ATIS (304×240), DAVIS240C (240×180)
    
    Directory Structure:
        EBSSA/
        ├── Labelled Data/
        │   ├── ATIS/
        │   │   ├── recording_001.mat
        │   │   └── ...
        │   ├── DAVIS/
        │   │   └── ...
        │   └── Labels/
        │       ├── recording_001_labels.mat
        │       └── ...
        └── Unlabelled/
            └── ...
    
    Example:
        >>> dataset = EBSSADataset(
        ...     root="./data/EBSSA",
        ...     split="train",
        ...     sensor="DAVIS",
        ...     n_timesteps=10
        ... )
        >>> voxel, label = dataset[0]
    """
    
    SENSORS = {
        'ATIS': {'height': 240, 'width': 304},
        'DAVIS': {'height': 180, 'width': 240},
    }
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        sensor: str = "DAVIS",
        n_timesteps: int = 10,
        height: Optional[int] = None,
        width: Optional[int] = None,
        normalize: bool = True,
        polarity_channels: bool = True,
        use_labels: bool = True,
        include_unlabelled: bool = False,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
        augmentation: Optional[EventAugmentation] = None,
        max_samples: Optional[int] = None,
        windows_per_recording: int = 1,
        window_overlap: float = 0.5
    ):
        """
        Initialize EBSSA dataset.
        
        Args:
            root: Path to EBSSA dataset root.
            split: "train", "val", "test", or "all".
            sensor: "ATIS", "DAVIS", or "all".
            n_timesteps: Number of temporal bins.
            height: Output height (default: sensor resolution).
            width: Output width (default: sensor resolution).
            normalize: Normalize voxel values.
            polarity_channels: Separate polarity channels.
            use_labels: Load labels if available.
            include_unlabelled: Include unlabelled data (for unsupervised training).
            train_ratio: Train/val split ratio.
            transform: Optional transform.
            augmentation: Optional augmentation.
            max_samples: Maximum samples to load (for debugging).
            windows_per_recording: Number of sliding windows per recording (1=original behavior).
            window_overlap: Overlap between consecutive windows (0.0-0.9).
        """
        # Set default resolution based on sensor
        if sensor in self.SENSORS:
            default_h = self.SENSORS[sensor]['height']
            default_w = self.SENSORS[sensor]['width']
        else:
            default_h, default_w = 180, 240
        
        super().__init__(
            n_timesteps=n_timesteps,
            height=height or default_h,
            width=width or default_w,
            normalize=normalize,
            polarity_channels=polarity_channels,
            transform=transform,
            augmentation=augmentation
        )
        
        self.root = Path(root)
        self.split = split
        self.sensor = sensor
        self.use_labels = use_labels
        self.include_unlabelled = include_unlabelled
        self.train_ratio = train_ratio
        self.windows_per_recording = max(1, windows_per_recording)
        self.window_overlap = min(0.9, max(0.0, window_overlap))
        
        # Default sensor resolution for loading
        self._sensor_height = default_h
        self._sensor_width = default_w
        
        # Find recordings
        self.recordings = self._find_recordings()
        
        # Apply split
        self._apply_split()
        
        # Limit samples
        if max_samples is not None:
            # Limit recordings, not windows
            max_recordings = max(1, max_samples // self.windows_per_recording)
            self.recordings = self.recordings[:max_recordings]
        
        # Build sample index mapping: sample_idx -> (recording_idx, window_idx)
        self._build_sample_index()
    
    def _find_recordings(self) -> List[Dict[str, Any]]:
        """Find all recording files."""
        recordings = []
        
        # Build list of directories to search
        search_dirs = []
        
        # Check labelled directory structures
        labelled_dir = self.root / "Labelled Data"
        if not labelled_dir.exists():
            labelled_dir = self.root / "Labelled_Data"
        if labelled_dir.exists():
            search_dirs.append(labelled_dir)
        
        # Add unlabelled directory if requested (for unsupervised STDP)
        if self.include_unlabelled:
            unlabelled_dir = self.root / "Unlabelled Data"
            if not unlabelled_dir.exists():
                unlabelled_dir = self.root / "Unlabelled_Data"
            if unlabelled_dir.exists():
                search_dirs.append(unlabelled_dir)
        
        # Fallback to root if no specific dirs found
        if not search_dirs:
            search_dirs.append(self.root)
        
        # Find .mat files in all search directories
        for search_dir in search_dirs:
            for mat_path in sorted(search_dir.rglob("*.mat")):
                # Skip pure label files (e.g., *_labels.mat), but keep *_labelled.mat
                # EBSSA format: event files are named *_labelled.mat (contains events + labels)
                stem_lower = mat_path.stem.lower()
                if stem_lower.endswith('_labels') or (
                    '_labels' in stem_lower and not stem_lower.endswith('_labelled')
                ):
                    continue
                
                # Check sensor
                parent_name = mat_path.parent.name.upper()
                if self.sensor != "all" and self.sensor.upper() not in parent_name:
                    # Check if it's in the correct sensor folder
                    if self.sensor.upper() not in str(mat_path).upper():
                        continue
                
                # Find corresponding label file
                label_path = None
                if self.use_labels:
                    label_candidates = [
                        mat_path.parent / f"{mat_path.stem}_labels.mat",
                        mat_path.parent / "Labels" / f"{mat_path.stem}_labels.mat",
                        search_dir / "Labels" / f"{mat_path.stem}_labels.mat",
                    ]
                    for candidate in label_candidates:
                        if candidate.exists():
                            label_path = candidate
                            break
                
                recordings.append({
                    'event_path': mat_path,
                    'label_path': label_path,
                    'name': mat_path.stem
                })
        
        return recordings
    
    def _apply_split(self) -> None:
        """Apply train/val/test split."""
        if self.split == "all":
            return
        
        n = len(self.recordings)
        n_train = int(n * self.train_ratio)
        
        if self.split == "train":
            self.recordings = self.recordings[:n_train]
        elif self.split in ["val", "test"]:
            self.recordings = self.recordings[n_train:]
    
    def _build_sample_index(self) -> None:
        """Build mapping from sample index to (recording_idx, window_idx).
        
        With sliding windows, each recording produces multiple samples.
        This maps the flat sample index to the specific recording and window.
        """
        self._sample_map: List[Tuple[int, int]] = []  # (recording_idx, window_idx)
        
        for rec_idx in range(len(self.recordings)):
            for win_idx in range(self.windows_per_recording):
                self._sample_map.append((rec_idx, win_idx))
    
    def __len__(self) -> int:
        return len(self._sample_map)
    
    def _load_sample(self, index: int) -> Tuple[EventData, Any]:
        """Load events and labels for a recording with sliding window.
        
        Uses the sample index to determine which recording and which
        time window to extract events from.
        """
        # Map sample index to recording and window
        rec_idx, win_idx = self._sample_map[index]
        rec = self.recordings[rec_idx]
        
        # Load all events with error handling for different .mat formats
        try:
            all_events = load_events_mat(
                rec['event_path'],
                height=self._sensor_height,
                width=self._sensor_width
            )
        except (IndexError, KeyError, OSError, ValueError) as e:
            # File format not supported - create empty events
            warnings.warn(f"Failed to load {rec['name']}: {e}. Using empty events.")
            all_events = EventData(
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                p=np.array([], dtype=np.int8),
                t=np.array([], dtype=np.int64),
                height=self._sensor_height,
                width=self._sensor_width
            )
        
        # Apply sliding window to extract subset of events
        events = self._extract_window(all_events, win_idx)
        
        # Load labels
        label = None
        if rec['label_path'] is not None:
            try:
                label = load_labels_mat(rec['label_path'])
            except Exception as e:
                warnings.warn(f"Failed to load labels for {rec['name']}: {e}")

        # EBSSA: Labels may be embedded in the event file (Obj field)
        if label is None and self.use_labels:
            try:
                import scipy.io as sio
                mat = sio.loadmat(rec['event_path'], squeeze_me=True)
                if 'Obj' in mat:
                    label = {'Obj': mat['Obj']}
            except Exception:
                pass
        
        # For segmentation, create binary mask
        # (placeholder - actual format depends on label file structure)
        if label is None:
            # No label - return zeros
            label = torch.zeros(self.height, self.width, dtype=torch.long)
        elif isinstance(label, dict):
            # Convert bounding box to mask if needed
            label = self._labels_to_mask(label, events)
        
        return events, label
    
    def _extract_window(self, events: EventData, window_idx: int) -> EventData:
        """Extract a time window from the full event stream.
        
        Divides the recording into overlapping windows and returns the
        events that fall within the specified window.
        
        Args:
            events: Full event stream from recording
            window_idx: Which window to extract (0 to windows_per_recording-1)
            
        Returns:
            EventData containing only events in the specified time window
        """
        if len(events.t) == 0:
            return events
        
        # If only 1 window, return all events
        if self.windows_per_recording == 1:
            return events
        
        t_min = events.t.min()
        t_max = events.t.max()
        duration = t_max - t_min
        
        if duration == 0:
            return events
        
        # Calculate window size based on overlap
        # With overlap=0.5 and N windows:
        #   - step_size = window_size * (1 - overlap)
        #   - Total span = (N-1) * step_size + window_size = duration
        #   - Solving: window_size = duration / (1 + (N-1) * (1 - overlap))
        effective_windows = 1.0 + (self.windows_per_recording - 1) * (1.0 - self.window_overlap)
        window_size = float(duration) / effective_windows
        
        # Step size between consecutive windows
        step_size = window_size * (1.0 - self.window_overlap)
        
        # Calculate window boundaries
        window_start = float(t_min) + window_idx * step_size
        window_end = window_start + window_size
        
        # Clamp to valid range (important for last window due to float precision)
        window_end = min(window_end, float(t_max) + 1)  # +1 to include t_max
        
        # Extract events in this window
        mask = (events.t >= window_start) & (events.t < window_end)
        
        if not np.any(mask):
            # No events in window - return empty
            return EventData(
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                p=np.array([], dtype=np.int8),
                t=np.array([], dtype=np.int64),
                height=events.height,
                width=events.width
            )
        
        # Shift timestamps to start at 0 for this window
        # Convert window_start to int64 to preserve precision for microsecond timestamps
        t_extracted = events.t[mask] - np.int64(window_start)
        
        return EventData(
            x=events.x[mask].copy(),
            y=events.y[mask].copy(),
            p=events.p[mask].copy(),
            t=t_extracted.copy(),  # Already int64 from subtraction
            height=events.height,
            width=events.width
        )
    
    def _labels_to_mask(
        self,
        labels: Dict[str, Any],
        events: EventData
    ) -> torch.Tensor:
        """Convert label dict to segmentation mask.

        Handles EBSSA format with object trajectories:
        - Obj.x, Obj.y: Object center coordinates over time
        - Obj.ts: Timestamps for each position
        """
        mask = torch.zeros(self.height, self.width, dtype=torch.long)

        # Scale factors from sensor resolution to output size
        scale_x = self.width / events.width
        scale_y = self.height / events.height

        # EBSSA format: Obj contains object trajectories
        if 'Obj' in labels:
            obj = labels['Obj']
            # Handle numpy structured array
            if hasattr(obj, 'dtype') and obj.dtype.names:
                obj_x = obj['x'].flatten() if 'x' in obj.dtype.names else None
                obj_y = obj['y'].flatten() if 'y' in obj.dtype.names else None
                obj_ts = obj['ts'].flatten() if 'ts' in obj.dtype.names else None
            elif isinstance(obj, tuple):
                # Tuple format: (x, y, id, ts, nObj)
                obj_x, obj_y = obj[0], obj[1]
                obj_ts = obj[3] if len(obj) > 3 else None
            else:
                obj_x, obj_y, obj_ts = None, None, None

            if obj_x is not None and obj_y is not None and len(obj_x) > 0:
                # Convert to numpy arrays if needed
                obj_x = np.asarray(obj_x).flatten()
                obj_y = np.asarray(obj_y).flatten()

                # Get event time range (convert to scalar for numpy comparison)
                if len(events.t) > 0:
                    t_min = float(events.t.min().item() if hasattr(events.t.min(), 'item') else events.t.min())
                    t_max = float(events.t.max().item() if hasattr(events.t.max(), 'item') else events.t.max())
                else:
                    t_min, t_max = 0.0, float('inf')

                # Find object positions within event time window
                if obj_ts is not None:
                    try:
                        # Handle nested arrays: recursively flatten and convert to float
                        obj_ts_flat = np.asarray(obj_ts).flatten()
                        # If still object dtype, try to extract numeric values
                        if obj_ts_flat.dtype == object:
                            obj_ts_flat = np.array([float(x) if np.isscalar(x) else float(np.asarray(x).flatten()[0]) for x in obj_ts_flat])
                        else:
                            obj_ts_flat = obj_ts_flat.astype(float)

                        if len(obj_ts_flat) > 0:
                            # Filter to positions within time window
                            time_mask = (obj_ts_flat >= t_min) & (obj_ts_flat <= t_max)
                            if np.any(time_mask):
                                pos_x = obj_x[time_mask]
                                pos_y = obj_y[time_mask]
                            else:
                                pos_x, pos_y = obj_x, obj_y
                        else:
                            pos_x, pos_y = obj_x, obj_y
                    except (ValueError, TypeError, IndexError):
                        # If timestamp filtering fails, use all positions
                        pos_x, pos_y = obj_x, obj_y
                else:
                    pos_x, pos_y = obj_x, obj_y

                # Create mask around object positions (use mean position for static mask)
                if len(pos_x) > 0:
                    # Ensure pos_x and pos_y are flat numeric arrays
                    try:
                        pos_x = np.asarray(pos_x).flatten().astype(float)
                        pos_y = np.asarray(pos_y).flatten().astype(float)
                    except (ValueError, TypeError):
                        # If conversion fails, skip mask creation
                        pass
                    else:
                        # Use all positions or subsample for efficiency
                        radius = 3  # Mask radius in output pixels
                        for i in range(0, len(pos_x), max(1, len(pos_x) // 100)):
                            x = int(pos_x[i] * scale_x)
                            y = int(pos_y[i] * scale_y)
                            # Create circular mask around object
                            y1 = max(0, y - radius)
                            y2 = min(self.height, y + radius + 1)
                            x1 = max(0, x - radius)
                            x2 = min(self.width, x + radius + 1)
                            mask[y1:y2, x1:x2] = 1

        # Fallback: Try bounding box format
        elif 'bbox' in labels:
            bbox = labels['bbox']
            if hasattr(bbox, 'ndim') and bbox.ndim == 1 and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                mask[y1:y2, x1:x2] = 1
        elif 'mask' in labels:
            mask_arr = labels['mask']
            mask = torch.from_numpy(mask_arr).long()
            if mask.shape != (self.height, self.width):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.height, self.width),
                    mode='nearest'
                ).squeeze().long()

        return mask
    
    def get_recording_info(self, index: int) -> Dict[str, Any]:
        """Get metadata for a recording."""
        rec = self.recordings[index]
        events, _ = self._load_sample(index)
        
        return {
            'name': rec['name'],
            'event_path': str(rec['event_path']),
            'label_path': str(rec['label_path']) if rec['label_path'] else None,
            'n_events': len(events),
            'duration_s': events.duration,
            'event_rate': events.event_rate,
        }


# =============================================================================
# N-MNIST DATASET
# =============================================================================


class NMNISTDataset(EventDataset):
    """
    Neuromorphic MNIST Dataset.
    
    Event-based variant of MNIST created by recording the original
    MNIST digits with an event camera performing saccades.
    
    Useful for testing and benchmarking SNN implementations.
    
    Paper Reference:
        Orchard et al. 2015 - "Converting Static Image Datasets to 
        Spiking Neuromorphic Datasets Using Saccades"
    
    Example:
        >>> dataset = NMNISTDataset(
        ...     root="./data/NMNIST",
        ...     train=True,
        ...     n_timesteps=10
        ... )
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        n_timesteps: int = 10,
        height: int = 34,
        width: int = 34,
        normalize: bool = True,
        polarity_channels: bool = True,
        transform: Optional[Callable] = None,
        augmentation: Optional[EventAugmentation] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize N-MNIST dataset.
        
        Args:
            root: Path to N-MNIST dataset root.
            train: Load training set if True, test set if False.
            n_timesteps: Number of temporal bins.
            height: Output height.
            width: Output width.
            normalize: Normalize voxel values.
            polarity_channels: Separate polarity channels.
            transform: Optional transform.
            augmentation: Optional augmentation.
            max_samples: Maximum samples to load.
        """
        super().__init__(
            n_timesteps=n_timesteps,
            height=height,
            width=width,
            normalize=normalize,
            polarity_channels=polarity_channels,
            transform=transform,
            augmentation=augmentation
        )
        
        self.root = Path(root)
        self.train = train
        
        # Find samples
        self.samples = self._find_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def _find_samples(self) -> List[Tuple[Path, int]]:
        """Find all sample files with labels."""
        samples = []
        
        split_dir = self.root / ("Train" if self.train else "Test")
        if not split_dir.exists():
            split_dir = self.root / ("train" if self.train else "test")
        
        # N-MNIST has directories for each digit (0-9)
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if digit_dir.exists():
                for f in sorted(digit_dir.glob("*.bin")):
                    samples.append((f, digit))
        
        # Also check for .npy format
        if len(samples) == 0:
            for digit in range(10):
                digit_dir = split_dir / str(digit)
                if digit_dir.exists():
                    for f in sorted(digit_dir.glob("*.npy")):
                        samples.append((f, digit))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_sample(self, index: int) -> Tuple[EventData, int]:
        """Load events and label for a sample."""
        filepath, label = self.samples[index]
        
        if filepath.suffix == '.bin':
            events = self._load_nmnist_bin(filepath)
        else:
            events = load_events_npy(filepath, height=34, width=34)
        
        return events, label
    
    def _load_nmnist_bin(self, filepath: Path) -> EventData:
        """Load N-MNIST binary format."""
        with open(filepath, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        
        # N-MNIST binary format: 5 bytes per event
        # byte 0: x (7 bits) + polarity (1 bit)
        # bytes 1-4: timestamp (microseconds, big endian)
        n_events = len(raw) // 5
        events = raw.reshape(n_events, 5)
        
        x = (events[:, 0] & 0x7F).astype(np.int32)
        y = events[:, 1].astype(np.int32)
        p = ((events[:, 0] >> 7) & 0x01).astype(np.int8)
        
        # Big-endian timestamp
        t = (events[:, 2].astype(np.int64) << 16 |
             events[:, 3].astype(np.int64) << 8 |
             events[:, 4].astype(np.int64))
        
        return EventData(x=x, y=y, p=p, t=t, height=34, width=34)


# =============================================================================
# SYNTHETIC DATASET
# =============================================================================


class SyntheticEventDataset(EventDataset):
    """
    Synthetic event dataset for testing.
    
    Generates random events with optional patterns.
    Useful for debugging and unit tests.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        n_events_per_sample: int = 1000,
        height: int = 128,
        width: int = 128,
        n_classes: int = 2,
        n_timesteps: int = 10,
        normalize: bool = True,
        polarity_channels: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            n_samples: Number of samples.
            n_events_per_sample: Events per sample.
            height: Image height.
            width: Image width.
            n_classes: Number of classes.
            n_timesteps: Temporal bins.
            normalize: Normalize voxels.
            polarity_channels: Separate polarity channels.
            seed: Random seed.
        """
        super().__init__(
            n_timesteps=n_timesteps,
            height=height,
            width=width,
            normalize=normalize,
            polarity_channels=polarity_channels
        )
        
        self.n_samples = n_samples
        self.n_events_per_sample = n_events_per_sample
        self.n_classes = n_classes
        self.seed = seed
        
        # Use local random state for reproducibility
        rng = np.random.RandomState(seed)
        
        # Pre-generate labels
        self.labels = rng.randint(0, n_classes, n_samples)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def _load_sample(self, index: int) -> Tuple[EventData, int]:
        """Generate synthetic events."""
        # Use local random state for reproducibility without affecting global state
        seed = index if self.seed is None else self.seed + index
        rng = np.random.RandomState(seed)
        
        n = self.n_events_per_sample
        
        x = rng.randint(0, self.width, n).astype(np.int32)
        y = rng.randint(0, self.height, n).astype(np.int32)
        p = rng.randint(0, 2, n).astype(np.int8)
        t = np.sort(rng.randint(0, 1000000, n)).astype(np.int64)
        
        events = EventData(
            x=x, y=y, p=p, t=t,
            height=self.height, width=self.width
        )
        
        label = int(self.labels[index])
        
        return events, label


# =============================================================================
# DATALOADER UTILITIES
# =============================================================================


def create_dataloader(
    dataset: EventDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create a DataLoader for event dataset.
    
    Args:
        dataset: EventDataset instance.
        batch_size: Batch size.
        shuffle: Shuffle samples.
        num_workers: Number of worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop incomplete last batch.
        collate_fn: Custom collate function.
    
    Returns:
        DataLoader instance.
    
    Example:
        >>> dataset = EBSSADataset(root="./data/EBSSA")
        >>> loader = create_dataloader(dataset, batch_size=4)
        >>> for voxels, labels in loader:
        ...     output = model(voxels)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    **kwargs
) -> EventDataset:
    """
    Factory function to create datasets by name.
    
    Args:
        name: Dataset name ("ebssa", "nmnist", "synthetic").
        root: Dataset root directory.
        split: Data split ("train", "val", "test").
        **kwargs: Additional arguments for dataset class.
    
    Returns:
        Dataset instance.
    
    Example:
        >>> dataset = get_dataset("ebssa", "./data/EBSSA", split="train")
    """
    name = name.lower()
    
    if name == "ebssa":
        return EBSSADataset(root=root, split=split, **kwargs)
    elif name == "nmnist":
        train = split == "train"
        return NMNISTDataset(root=root, train=train, **kwargs)
    elif name == "synthetic":
        return SyntheticEventDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data containers
    "EventData",
    
    # Conversion functions
    "events_to_voxel_grid",
    "events_to_frame",
    "events_to_time_surface",
    
    # File loaders
    "load_events",
    "load_events_mat",
    "load_events_h5",
    "load_events_npy",
    "load_labels_mat",
    
    # Augmentation
    "EventAugmentation",
    
    # Base class
    "EventDataset",
    
    # Datasets
    "EBSSADataset",
    "NMNISTDataset",
    "SyntheticEventDataset",
    
    # Utilities
    "create_dataloader",
    "get_dataset",
]