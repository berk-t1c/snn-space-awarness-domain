"""
Visualization utilities for SpikeSEG.

This module provides paper-quality visualizations for:
    - Learned STDP filters (weight kernels)
    - Saliency maps and segmentation overlays
    - Spike raster plots and activity maps
    - Event camera data (polarity, time surfaces)
    - Training convergence and metrics
    - Feature activation maps
    - WTA winner statistics

Paper References:
    Kirkland et al. 2020 (SpikeSEG) - Saliency mapping visualizations
    Kirkland et al. 2022 (HULK-SMASH) - Instance segmentation visualizations
    Kirkland et al. 2023 (IGARSS) - Satellite detection visualizations
    Kheradpisheh et al. 2018 - STDP filter visualizations

Example:
    >>> from spikeseg.utils.visualization import (
    ...     plot_learned_filters, plot_saliency_overlay, plot_spike_raster
    ... )
    >>> 
    >>> # Visualize STDP-learned filters
    >>> fig = plot_learned_filters(model.encoder.conv2.weight)
    >>> fig.savefig("learned_filters.png", dpi=150)
    >>> 
    >>> # Overlay saliency on input
    >>> fig = plot_saliency_overlay(input_events, saliency_map, alpha=0.6)

Author: SpikeSEG Team
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass

import numpy as np

# Matplotlib imports with backend handling
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization functions will not work.")

# PyTorch imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for visualization style."""
    # Figure settings
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 150
    
    # Color settings
    cmap_filters: str = 'gray'
    cmap_saliency: str = 'hot'
    cmap_events: str = 'RdBu_r'  # Red=positive, Blue=negative polarity
    cmap_spikes: str = 'binary'
    cmap_activation: str = 'viridis'
    
    # Style settings
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    
    # Grid settings
    grid_spacing: float = 0.02
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.04
    
    # Animation settings
    fps: int = 10
    interval: int = 100


# Global default config
DEFAULT_CONFIG = VisualizationConfig()


def _try_style(style_names: List[str]) -> bool:
    """Try to apply a style from a list of alternatives."""
    for style_name in style_names:
        try:
            plt.style.use(style_name)
            return True
        except OSError:
            continue
    return False


def set_style(style: str = 'paper') -> None:
    """
    Set matplotlib style for paper-quality figures.
    
    Args:
        style: Style preset ('paper', 'presentation', 'dark', 'default').
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    if style == 'paper':
        # Try modern seaborn style names, fall back to older versions
        _try_style(['seaborn-v0_8-whitegrid', 'seaborn-whitegrid'])
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    elif style == 'presentation':
        _try_style(['seaborn-v0_8-talk', 'seaborn-talk'])
    elif style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize(x: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    
    if vmax - vmin < 1e-8:
        return np.zeros_like(x)
    
    return (x - vmin) / (vmax - vmin)


def _get_grid_dims(n: int, max_cols: int = 8) -> Tuple[int, int]:
    """Calculate grid dimensions for n items."""
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    return rows, cols


def _add_colorbar(
    fig: "Figure",
    ax: "Axes",
    im,
    label: str = "",
    fraction: float = 0.046,
    pad: float = 0.04
) -> None:
    """Add colorbar to axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    cbar = fig.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)


def _save_and_close(
    fig: "Figure",
    save_path: Optional[Union[str, Path]],
    dpi: int = 150,
    close: bool = False
) -> None:
    """Save figure and optionally close to free memory."""
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if close:
            plt.close(fig)


# =============================================================================
# FILTER VISUALIZATION
# =============================================================================

def plot_learned_filters(
    weights: Union[np.ndarray, "torch.Tensor"],
    title: str = "Learned STDP Filters",
    max_filters: int = 64,
    ncols: int = 8,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'gray',
    normalize_each: bool = True,
    show_colorbar: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize learned convolutional filters as a grid.
    
    This matches the filter visualization style in Kheradpisheh et al. 2018
    and SpikeSEG papers showing Gabor-like learned features.
    
    Args:
        weights: Filter weights, shape (out_ch, in_ch, kH, kW).
                 For multi-channel input, averages across input channels.
        title: Figure title.
        max_filters: Maximum number of filters to display.
        ncols: Number of columns in grid.
        figsize: Figure size (width, height).
        cmap: Colormap for filters.
        normalize_each: Normalize each filter individually.
        show_colorbar: Show colorbar for weight values.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    
    Example:
        >>> weights = model.encoder.conv2.weight  # (36, 4, 5, 5)
        >>> fig = plot_learned_filters(weights, title="Conv2 Features")
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    weights = _to_numpy(weights)
    
    # Handle different weight shapes
    if weights.ndim == 4:
        # (out_ch, in_ch, kH, kW) -> average over input channels
        weights = weights.mean(axis=1)
    elif weights.ndim == 3:
        pass  # Already (out_ch, kH, kW)
    else:
        raise ValueError(f"Expected 3D or 4D weights, got {weights.ndim}D")
    
    n_filters = min(weights.shape[0], max_filters)
    nrows, ncols = _get_grid_dims(n_filters, ncols)
    
    if figsize is None:
        figsize = (ncols * 1.2, nrows * 1.2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_filters == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        if i < n_filters:
            w = weights[i]
            if normalize_each:
                w = _normalize(w)
            im = ax.imshow(w, cmap=cmap, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'F{i}', fontsize=8, pad=2)
        else:
            ax.axis('off')
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if show_colorbar and n_filters > 0:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Weight')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_filter_evolution(
    weight_history: List[Union[np.ndarray, "torch.Tensor"]],
    filter_indices: List[int] = None,
    steps: List[int] = None,
    title: str = "Filter Evolution During Training",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize how filters evolve during STDP training.
    
    Args:
        weight_history: List of weight snapshots at different training steps.
        filter_indices: Which filters to show. Default: first 8.
        steps: Training step labels. Default: 0, 1, 2, ...
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    n_snapshots = len(weight_history)
    if n_snapshots == 0:
        raise ValueError("weight_history is empty")
    
    weights = [_to_numpy(w) for w in weight_history]
    if weights[0].ndim == 4:
        weights = [w.mean(axis=1) for w in weights]
    
    if filter_indices is None:
        filter_indices = list(range(min(8, weights[0].shape[0])))
    
    if steps is None:
        steps = list(range(n_snapshots))
    
    n_filters = len(filter_indices)
    
    if figsize is None:
        figsize = (n_snapshots * 1.5, n_filters * 1.5)
    
    fig, axes = plt.subplots(n_filters, n_snapshots, figsize=figsize)
    if n_filters == 1:
        axes = axes.reshape(1, -1)
    if n_snapshots == 1:
        axes = axes.reshape(-1, 1)
    
    for i, fi in enumerate(filter_indices):
        for j, (w, step) in enumerate(zip(weights, steps)):
            ax = axes[i, j]
            ax.imshow(_normalize(w[fi]), cmap='gray', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(f'Step {step}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'F{fi}', fontsize=9)
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# SALIENCY MAP VISUALIZATION
# =============================================================================

def plot_saliency_map(
    saliency: Union[np.ndarray, "torch.Tensor"],
    title: str = "Saliency Map",
    cmap: str = 'hot',
    figsize: Tuple[float, float] = (8, 6),
    colorbar: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize decoder saliency map.
    
    Args:
        saliency: Saliency map, shape (H, W) or (C, H, W) or (B, C, H, W).
        title: Figure title.
        cmap: Colormap.
        figsize: Figure size.
        colorbar: Show colorbar.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    saliency = _to_numpy(saliency)
    
    # Handle different shapes
    while saliency.ndim > 2:
        saliency = saliency[0] if saliency.shape[0] == 1 else saliency.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(saliency, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if colorbar:
        _add_colorbar(fig, ax, im, label='Saliency')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_saliency_overlay(
    input_image: Union[np.ndarray, "torch.Tensor"],
    saliency: Union[np.ndarray, "torch.Tensor"],
    alpha: float = 0.6,
    title: str = "Saliency Overlay",
    cmap_saliency: str = 'hot',
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Overlay saliency map on input image (SpikeSEG paper style).
    
    This matches Figure 2 style from SpikeSEG 2020 paper showing
    input -> saliency -> overlay visualization.
    
    Args:
        input_image: Input image/events, shape (H, W) or (C, H, W).
        saliency: Saliency map from decoder.
        alpha: Saliency overlay transparency.
        title: Figure title.
        cmap_saliency: Colormap for saliency.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure with 3 subplots: Input | Saliency | Overlay
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    input_image = _to_numpy(input_image)
    saliency = _to_numpy(saliency)
    
    # Handle shapes
    while input_image.ndim > 2:
        if input_image.shape[0] <= 3:
            input_image = input_image.transpose(1, 2, 0)
            if input_image.shape[2] == 1:
                input_image = input_image[:, :, 0]
        else:
            input_image = input_image[0]
    
    while saliency.ndim > 2:
        saliency = saliency[0] if saliency.shape[0] == 1 else saliency.sum(axis=0)
    
    # Normalize
    input_norm = _normalize(input_image)
    saliency_norm = _normalize(saliency)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Input
    axes[0].imshow(input_norm, cmap='gray', aspect='equal')
    axes[0].set_title('Input', fontsize=11)
    axes[0].axis('off')
    
    # Saliency
    im = axes[1].imshow(saliency_norm, cmap=cmap_saliency, aspect='equal')
    axes[1].set_title('Saliency Map', fontsize=11)
    axes[1].axis('off')
    _add_colorbar(fig, axes[1], im)
    
    # Overlay
    axes[2].imshow(input_norm, cmap='gray', aspect='equal')
    axes[2].imshow(saliency_norm, cmap=cmap_saliency, alpha=alpha, aspect='equal')
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_segmentation_result(
    input_image: Union[np.ndarray, "torch.Tensor"],
    prediction: Union[np.ndarray, "torch.Tensor"],
    ground_truth: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    class_names: Optional[List[str]] = None,
    title: str = "Segmentation Result",
    figsize: Tuple[float, float] = (15, 5),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize segmentation result with optional ground truth.
    
    Args:
        input_image: Input image.
        prediction: Predicted segmentation mask.
        ground_truth: Optional ground truth mask.
        class_names: Names for each class.
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    input_image = _to_numpy(input_image)
    prediction = _to_numpy(prediction)
    
    while input_image.ndim > 2:
        input_image = input_image[0] if input_image.shape[0] <= 3 else input_image.mean(axis=0)
    while prediction.ndim > 2:
        prediction = prediction.argmax(axis=0) if prediction.shape[0] > 1 else prediction[0]
    
    n_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Input
    axes[0].imshow(_normalize(input_image), cmap='gray')
    axes[0].set_title('Input', fontsize=11)
    axes[0].axis('off')
    
    # Prediction
    n_classes = int(prediction.max()) + 1
    # Use colormaps API (matplotlib 3.7+) with fallback
    try:
        cmap = plt.colormaps.get_cmap('tab10').resampled(n_classes)
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10', n_classes)
    axes[1].imshow(prediction, cmap=cmap, vmin=0, vmax=n_classes-1)
    axes[1].set_title('Prediction', fontsize=11)
    axes[1].axis('off')
    
    # Ground truth
    if ground_truth is not None:
        ground_truth = _to_numpy(ground_truth)
        while ground_truth.ndim > 2:
            ground_truth = ground_truth[0]
        axes[2].imshow(ground_truth, cmap=cmap, vmin=0, vmax=n_classes-1)
        axes[2].set_title('Ground Truth', fontsize=11)
        axes[2].axis('off')
    
    # Legend
    if class_names:
        patches = [mpatches.Patch(color=cmap(i), label=name) 
                   for i, name in enumerate(class_names)]
        fig.legend(handles=patches, loc='center right', fontsize=9)
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# SPIKE VISUALIZATION
# =============================================================================

def plot_spike_raster(
    spikes: Union[np.ndarray, "torch.Tensor"],
    title: str = "Spike Raster Plot",
    neuron_labels: Optional[List[str]] = None,
    time_unit: str = "timestep",
    max_neurons: int = 100,
    figsize: Tuple[float, float] = (12, 6),
    marker_size: float = 2.0,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Create spike raster plot showing spike times for each neuron.
    
    Args:
        spikes: Spike tensor, shape (T, N) or (T, C, H, W).
                For 4D, flattens spatial dimensions.
        title: Figure title.
        neuron_labels: Labels for neurons.
        time_unit: Label for time axis.
        max_neurons: Maximum neurons to display.
        figsize: Figure size.
        marker_size: Size of spike markers.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    spikes = _to_numpy(spikes)
    
    # Flatten to (T, N)
    if spikes.ndim > 2:
        T = spikes.shape[0]
        spikes = spikes.reshape(T, -1)
    
    T, N = spikes.shape
    
    # Limit neurons
    if N > max_neurons:
        # Sample evenly
        indices = np.linspace(0, N-1, max_neurons, dtype=int)
        spikes = spikes[:, indices]
        N = max_neurons
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find spike times and neuron indices
    spike_times, spike_neurons = np.where(spikes > 0)
    
    ax.scatter(spike_times, spike_neurons, s=marker_size, c='black', marker='|')
    
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel(f'Time ({time_unit})', fontsize=10)
    ax.set_ylabel('Neuron Index', fontsize=10)
    ax.set_title(title, fontsize=12)
    
    if neuron_labels:
        ax.set_yticks(range(len(neuron_labels)))
        ax.set_yticklabels(neuron_labels, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_membrane_traces(
    membrane: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 10.0,
    neuron_indices: Optional[List[int]] = None,
    title: str = "Membrane Potential Traces",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot membrane potential traces over time.
    
    Args:
        membrane: Membrane potential, shape (T, N) or (T, C, H, W).
        threshold: Firing threshold (drawn as dashed line).
        neuron_indices: Which neurons to plot. Default: first 5.
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    membrane = _to_numpy(membrane)
    
    # Flatten to (T, N)
    if membrane.ndim > 2:
        T = membrane.shape[0]
        membrane = membrane.reshape(T, -1)
    
    T, N = membrane.shape
    
    if neuron_indices is None:
        neuron_indices = list(range(min(5, N)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, ni in enumerate(neuron_indices):
        ax.plot(membrane[:, ni], label=f'Neuron {ni}', alpha=0.8)
    
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Time (timestep)', fontsize=10)
    ax.set_ylabel('Membrane Potential', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_spike_activity_map(
    spikes: Union[np.ndarray, "torch.Tensor"],
    title: str = "Spike Activity Map",
    cmap: str = 'hot',
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot spatial spike activity (spike count per location).
    
    Args:
        spikes: Spike tensor, shape (T, C, H, W) or (C, H, W) or (H, W).
        title: Figure title.
        cmap: Colormap.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    spikes = _to_numpy(spikes)
    
    # Sum over time and channels
    while spikes.ndim > 2:
        spikes = spikes.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(spikes, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    _add_colorbar(fig, ax, im, label='Spike Count')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# EVENT CAMERA VISUALIZATION
# =============================================================================

def plot_events_frame(
    events: Union[np.ndarray, "torch.Tensor"],
    title: str = "Event Frame",
    cmap: str = 'RdBu_r',
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize event camera data as a frame.
    
    Red = positive polarity (ON events)
    Blue = negative polarity (OFF events)
    
    Args:
        events: Event frame, shape (H, W) with polarity or (2, H, W) for pos/neg.
        title: Figure title.
        cmap: Colormap (RdBu_r for red=positive, blue=negative).
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    events = _to_numpy(events)
    
    # Handle (2, H, W) format (positive, negative channels)
    if events.ndim == 3 and events.shape[0] == 2:
        # Combine: positive - negative
        events = events[0] - events[1]
    elif events.ndim > 2:
        events = events.sum(axis=tuple(range(events.ndim - 2)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = max(abs(events.min()), abs(events.max()))
    im = ax.imshow(events, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    _add_colorbar(fig, ax, im, label='Polarity')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_time_surface(
    events: Union[np.ndarray, "torch.Tensor"],
    timestamps: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    tau: float = 0.1,
    title: str = "Time Surface",
    cmap: str = 'viridis',
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize events as a time surface (temporal activity map).
    
    Time surface shows recent activity with exponential decay.
    
    Args:
        events: Event coordinates and times, or pre-computed time surface.
        timestamps: Event timestamps (if events is coordinates).
        tau: Decay constant for exponential kernel.
        title: Figure title.
        cmap: Colormap.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    events = _to_numpy(events)
    
    # If already a 2D time surface
    if events.ndim == 2:
        time_surface = events
    else:
        # Assume last frame represents recent activity
        while events.ndim > 2:
            events = events[-1] if events.shape[0] > 1 else events[0]
        time_surface = events
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(time_surface, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    _add_colorbar(fig, ax, im, label='Recency')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_event_sequence(
    events: Union[np.ndarray, "torch.Tensor"],
    n_frames: int = 8,
    title: str = "Event Sequence",
    cmap: str = 'RdBu_r',
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize sequence of event frames over time.
    
    Args:
        events: Event sequence, shape (T, H, W) or (T, 2, H, W).
        n_frames: Number of frames to display.
        title: Figure title.
        cmap: Colormap.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    events = _to_numpy(events)
    
    T = events.shape[0]
    indices = np.linspace(0, T-1, min(n_frames, T), dtype=int)
    
    nrows, ncols = _get_grid_dims(len(indices), max_cols=8)
    
    if figsize is None:
        figsize = (ncols * 2.5, nrows * 2.5)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i, idx in enumerate(indices):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        frame = events[idx]
        if frame.ndim == 3 and frame.shape[0] == 2:
            frame = frame[0] - frame[1]
        elif frame.ndim > 2:
            frame = frame.sum(axis=0)
        
        vmax = max(abs(frame.min()), abs(frame.max()), 1e-8)
        ax.imshow(frame, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(f't={idx}', fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(len(indices), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# TRAINING METRICS VISUALIZATION
# =============================================================================

def plot_convergence_metric(
    convergence_history: List[float],
    threshold: float = 0.01,
    title: str = "Weight Convergence",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot STDP convergence metric over training.
    
    C = Σ w*(1-w) / n - approaches 0 as weights converge to 0 or 1.
    
    Args:
        convergence_history: List of convergence metric values.
        threshold: Convergence threshold (dashed line).
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(convergence_history, 'b-', linewidth=1.5, label='C = Σw(1-w)/n')
    ax.axhline(y=threshold, color='r', linestyle='--', 
               label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Update', fontsize=10)
    ax.set_ylabel('Convergence Metric', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weight_distribution(
    weights: Union[np.ndarray, "torch.Tensor"],
    title: str = "Weight Distribution",
    bins: int = 50,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot histogram of weight values.
    
    For converged STDP, weights should be bimodal (near 0 and 1).
    
    Args:
        weights: Weight tensor.
        title: Figure title.
        bins: Number of histogram bins.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    weights = _to_numpy(weights).flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(weights, bins=bins, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0.5, color='r', linestyle='--', label='w=0.5 (unconverged)')
    
    ax.set_xlabel('Weight Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wta_wins(
    win_counts: Union[np.ndarray, "torch.Tensor", List[int]],
    min_wins_threshold: int = 10,
    title: str = "WTA Winner Counts per Feature",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot histogram of WTA wins per feature.
    
    Shows how evenly features are being used during training.
    
    Args:
        win_counts: Number of wins per feature.
        min_wins_threshold: Threshold for "converged" feature.
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    win_counts = _to_numpy(win_counts)
    n_features = len(win_counts)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if w >= min_wins_threshold else 'red' for w in win_counts]
    bars = ax.bar(range(n_features), win_counts, color=colors, edgecolor='black', 
                  linewidth=0.5)
    
    ax.axhline(y=min_wins_threshold, color='orange', linestyle='--', 
               label=f'Min wins threshold ({min_wins_threshold})')
    
    n_converged = sum(1 for w in win_counts if w >= min_wins_threshold)
    ax.set_xlabel('Feature Index', fontsize=10)
    ax.set_ylabel('Win Count', fontsize=10)
    ax.set_title(f'{title}\n({n_converged}/{n_features} converged)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Plot multiple training metrics in subplots.
    
    Args:
        metrics: Dict of {metric_name: values_list}.
        title: Figure title.
        figsize: Figure size.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    n_metrics = len(metrics)
    nrows = (n_metrics + 1) // 2
    ncols = min(2, n_metrics)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(values, linewidth=1.5)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# FEATURE ACTIVATION VISUALIZATION
# =============================================================================

def plot_feature_activations(
    activations: Union[np.ndarray, "torch.Tensor"],
    title: str = "Feature Activations",
    max_features: int = 36,
    ncols: int = 6,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize feature map activations as a grid.
    
    Args:
        activations: Feature maps, shape (C, H, W) or (B, C, H, W).
        title: Figure title.
        max_features: Maximum features to show.
        ncols: Number of columns.
        figsize: Figure size.
        cmap: Colormap.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    activations = _to_numpy(activations)
    
    # Handle batch dimension
    if activations.ndim == 4:
        activations = activations[0]  # Take first sample
    
    C, H, W = activations.shape
    n_features = min(C, max_features)
    nrows, ncols = _get_grid_dims(n_features, ncols)
    
    if figsize is None:
        figsize = (ncols * 2, nrows * 2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        if i < n_features:
            ax.imshow(activations[i], cmap=cmap, aspect='equal')
            ax.set_title(f'F{i}', fontsize=8, pad=2)
            ax.axis('off')
        else:
            ax.axis('off')
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# SATELLITE DETECTION VISUALIZATION (IGARSS 2023)
# =============================================================================

def plot_satellite_detection(
    events: Union[np.ndarray, "torch.Tensor"],
    detections: Union[np.ndarray, "torch.Tensor"],
    title: str = "Satellite Detection",
    figsize: Tuple[float, float] = (10, 8),
    marker_size: float = 100,
    save_path: Optional[Union[str, Path]] = None
) -> "Figure":
    """
    Visualize satellite detection results (IGARSS 2023 style).
    
    Args:
        events: Event frame background.
        detections: Detection locations, shape (N, 2) for (y, x) or mask.
        title: Figure title.
        figsize: Figure size.
        marker_size: Size of detection markers.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")
    
    events = _to_numpy(events)
    detections = _to_numpy(detections)
    
    # Handle event shape
    while events.ndim > 2:
        events = events.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show events as background
    ax.imshow(events, cmap='gray', aspect='equal')
    
    # Plot detections
    if detections.ndim == 2 and detections.shape[1] == 2:
        # (N, 2) format: (y, x) coordinates
        y_coords, x_coords = detections[:, 0], detections[:, 1]
        ax.scatter(x_coords, y_coords, s=marker_size, c='red', marker='o',
                   facecolors='none', linewidths=2, label='Detection')
    elif detections.ndim == 2:
        # Mask format
        y_coords, x_coords = np.where(detections > 0)
        ax.scatter(x_coords, y_coords, s=marker_size/4, c='red', marker='.',
                   label='Detection')
    
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "VisualizationConfig",
    "set_style",
    
    # Filter visualization
    "plot_learned_filters",
    "plot_filter_evolution",
    
    # Saliency visualization
    "plot_saliency_map",
    "plot_saliency_overlay",
    "plot_segmentation_result",
    
    # Spike visualization
    "plot_spike_raster",
    "plot_membrane_traces",
    "plot_spike_activity_map",
    
    # Event camera visualization
    "plot_events_frame",
    "plot_time_surface",
    "plot_event_sequence",
    
    # Training metrics
    "plot_convergence_metric",
    "plot_weight_distribution",
    "plot_wta_wins",
    "plot_training_metrics",
    
    # Feature visualization
    "plot_feature_activations",
    
    # Satellite detection
    "plot_satellite_detection",
]