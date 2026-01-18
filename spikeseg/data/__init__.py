"""
Event Camera Data Loading for SpikeSEG.

This module provides:
    - EventData: Container for raw events (x, y, polarity, timestamp)
    - Conversion functions: events_to_voxel_grid, events_to_frame, etc.
    - File loaders: .mat, .h5, .npy formats
    - Dataset classes: EBSSADataset, NMNISTDataset, SyntheticEventDataset
    - DataLoader utilities: create_dataloader, get_dataset

Paper References:
    EBSSA Dataset (Afshar et al. 2020):
        - 84 labelled recordings of satellites, planets, stars
        - Resolution: 240×180 (DAVIS), 304×240 (ATIS)
        - Events: [x, y, p, t] in .mat files
    
    IGARSS 2023 Processing (Kirkland et al.):
        - Temporal buffering into 20 timesteps
        - 10 parsed event streams per buffer

Example:
    >>> from spikeseg.data import EBSSADataset, create_dataloader
    >>> 
    >>> dataset = EBSSADataset(
    ...     root="./data/EBSSA",
    ...     split="train",
    ...     n_timesteps=10,
    ...     height=128,
    ...     width=128
    ... )
    >>> 
    >>> loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    >>> 
    >>> for voxels, labels in loader:
    ...     # voxels: (B, T, C, H, W)
    ...     output = model(voxels)
"""

from .datasets import (
    # Data containers
    EventData,
    
    # Conversion functions
    events_to_voxel_grid,
    events_to_frame,
    events_to_time_surface,
    
    # File loaders
    load_events,
    load_events_mat,
    load_events_h5,
    load_events_npy,
    load_labels_mat,
    
    # Augmentation
    EventAugmentation,
    
    # Base class
    EventDataset,
    
    # Datasets
    EBSSADataset,
    NMNISTDataset,
    SyntheticEventDataset,
    
    # Utilities
    create_dataloader,
    get_dataset,
)

from .events import (
    # DoG filtering (Kheradpisheh 2018)
    create_dog_kernel,
    create_dog_filterbank,
    DoGFilter,
    
    # Normalization (SpykeTorch)
    local_normalization,
    LocalNormalization,
    
    # Intensity to latency encoding (rank-order coding)
    intensity_to_latency,
    intensity_to_latency_linear,
    IntensityToLatency,
    
    # Lateral inhibition
    lateral_inhibition,
    pointwise_inhibition,
    
    # Complete image-to-spike transform
    ImageToSpikeWave,
    
    # Event stream processing (EBSSA/IGARSS)
    EventBuffer,
    EventStreamProcessor,
    
    # Noise filtering
    filter_refractory_events,
    filter_isolated_events,
)

from .preprocessing import (
    # Gabor filters (alternative to DoG)
    create_gabor_kernel,
    create_gabor_filterbank,
    GaborFilterBank,
    
    # Adaptive thresholding (PENT - SpikeSEG)
    AdaptiveThreshold,
    LayerwiseAdaptiveThreshold,
    
    # LIF buffer (SpikeSEG)
    LIFBuffer,
    
    # Temporal buffering (SpikeSEG/IGARSS)
    TemporalBuffer,
    
    # Complete preprocessing pipelines
    SpykeTorchPreprocessor,
    SpikeSEGPreprocessor,
    
    # Spike augmentation
    SpikeAugmentation,
    
    # Normalization utilities
    normalize_events,
    normalize_timestamps,
    
    # Composable transforms
    Compose,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop,
    RandomCrop,
)


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
    
    # DoG filtering (Kheradpisheh 2018)
    "create_dog_kernel",
    "create_dog_filterbank",
    "DoGFilter",
    
    # Normalization (SpykeTorch)
    "local_normalization",
    "LocalNormalization",
    
    # Intensity to latency encoding
    "intensity_to_latency",
    "intensity_to_latency_linear",
    "IntensityToLatency",
    
    # Lateral inhibition
    "lateral_inhibition",
    "pointwise_inhibition",
    
    # Complete transforms
    "ImageToSpikeWave",
    
    # Event stream processing
    "EventBuffer",
    "EventStreamProcessor",
    
    # Noise filtering
    "filter_refractory_events",
    "filter_isolated_events",
    
    # Gabor filters (preprocessing.py)
    "create_gabor_kernel",
    "create_gabor_filterbank",
    "GaborFilterBank",
    
    # Adaptive thresholding (PENT - SpikeSEG)
    "AdaptiveThreshold",
    "LayerwiseAdaptiveThreshold",
    
    # LIF buffer (SpikeSEG)
    "LIFBuffer",
    
    # Temporal buffering (SpikeSEG/IGARSS)
    "TemporalBuffer",
    
    # Complete preprocessing pipelines
    "SpykeTorchPreprocessor",
    "SpikeSEGPreprocessor",
    
    # Spike augmentation
    "SpikeAugmentation",
    
    # Normalization utilities
    "normalize_events",
    "normalize_timestamps",
    
    # Composable transforms
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    "CenterCrop",
    "RandomCrop",
]
