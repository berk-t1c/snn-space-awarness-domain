#!/usr/bin/env python3
"""
Comprehensive STDP Training Script for SpikeSEG.

This script implements unsupervised layer-wise training using Spike-Timing
Dependent Plasticity (STDP) for the SpikeSEG spiking neural network.

Key Features:
    - Layer-wise training (Conv1 → Conv2 → Conv3)
    - STDP learning with Winner-Take-All (WTA) competition
    - Adaptive thresholds for homeostasis
    - Convergence monitoring per neuron/feature
    - Checkpoint saving and resuming
    - Comprehensive metrics logging
    - Signal handling for graceful shutdown
    - Memory-efficient event data processing

Training Procedure (from papers):
    1. Initialize Conv1 with DoG/Gabor filters (fixed, no training)
    2. Train Conv2 using STDP until convergence (~1 epoch for EBSSA)
    3. Train Conv3 (classification) using STDP
    4. Attach decoder with tied weights

Paper References:
    Kheradpisheh et al. 2018 - "STDP-based spiking deep convolutional neural 
    networks for object recognition"
    
    Kirkland et al. 2023 (IGARSS) - "Neuromorphic sensing and processing for 
    space domain awareness"
        - α⁺ = 0.04, α⁻ = 0.03
        - Weight initialization: 0.8 ± 0.01
        - Features converge within 1 epoch

Example:
    >>> python train.py --config configs/ebssa.yaml --output runs/exp1
    >>> python train.py --resume runs/exp1/checkpoint_latest.pt

Author: SpikeSEG Team
"""

from __future__ import annotations

import os
import sys
import time
import json
import signal
import atexit
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, TYPE_CHECKING
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum

import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Type checking imports (not imported at runtime)
if TYPE_CHECKING:
    from spikeseg.learning.stdp import STDPLearner

# Add project root to path (scripts/train.py -> scripts -> project_root)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class ConfigurationError(TrainingError):
    """Raised for configuration errors."""
    pass


class CheckpointError(TrainingError):
    """Raised for checkpoint save/load errors."""
    pass


class ConvergenceError(TrainingError):
    """Raised when training fails to converge."""
    pass


class DataLoadingError(TrainingError):
    """Raised for data loading errors."""
    pass


# =============================================================================
# ENUMS
# =============================================================================


class TrainingPhase(Enum):
    """Training phases for layer-wise STDP."""
    INITIALIZATION = "initialization"
    CONV1 = "conv1_training"
    CONV2 = "conv2_training"
    CONV3 = "conv3_training"
    VALIDATION = "validation"
    COMPLETED = "completed"


class WTAMode(Enum):
    """Winner-Take-All modes."""
    GLOBAL = "global"   # One winner per feature map
    LOCAL = "local"     # Winners in spatial neighborhoods
    BOTH = "both"       # Combined global and local


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class STDPParams:
    """
    STDP learning rule parameters.
    
    Paper Reference (IGARSS 2023):
        "STDP parameters: α⁺ = 0.04, α⁻ = 0.03"
    """
    lr_plus: float = 0.04       # Potentiation learning rate
    lr_minus: float = 0.03      # Depression learning rate
    weight_min: float = 0.0     # Minimum weight bound
    weight_max: float = 1.0     # Maximum weight bound
    weight_init_mean: float = 0.8  # Initial weight mean
    weight_init_std: float = 0.01  # Initial weight std
    use_soft_bounds: bool = True   # Use multiplicative soft bounds
    
    def validate(self) -> None:
        """Validate parameters."""
        if self.lr_plus <= 0:
            raise ConfigurationError(f"lr_plus must be positive, got {self.lr_plus}")
        if self.lr_minus <= 0:
            raise ConfigurationError(f"lr_minus must be positive, got {self.lr_minus}")
        if self.weight_min >= self.weight_max:
            raise ConfigurationError(
                f"weight_min ({self.weight_min}) must be < weight_max ({self.weight_max})"
            )
        if not (self.weight_min <= self.weight_init_mean <= self.weight_max):
            raise ConfigurationError(
                f"weight_init_mean ({self.weight_init_mean}) must be in "
                f"[{self.weight_min}, {self.weight_max}]"
            )


@dataclass
class WTAParams:
    """
    Winner-Take-All parameters for lateral inhibition.
    
    Controls competition between neurons to enforce sparse coding.
    """
    mode: str = "global"         # 'global', 'local', or 'both'
    local_radius: int = 2        # Radius for local inhibition
    enable_homeostasis: bool = True  # Adaptive thresholds
    target_rate: float = 0.1     # Target firing rate for homeostasis
    homeostasis_lr: float = 0.001  # Threshold adaptation rate
    threshold_min: float = 1.0   # Minimum adaptive threshold
    threshold_max: float = 100.0 # Maximum adaptive threshold
    
    def validate(self) -> None:
        """Validate parameters."""
        valid_modes = {'global', 'local', 'both'}
        if self.mode not in valid_modes:
            raise ConfigurationError(
                f"WTA mode must be one of {valid_modes}, got '{self.mode}'"
            )
        if self.local_radius < 0:
            raise ConfigurationError(
                f"local_radius must be non-negative, got {self.local_radius}"
            )
        if not (0 < self.target_rate < 1):
            raise ConfigurationError(
                f"target_rate must be in (0, 1), got {self.target_rate}"
            )


@dataclass
class HomeostasisParams:
    """
    Adaptive threshold (homeostasis) parameters.
    
    Paper Reference (Lee et al. 2018 / Diehl & Cook 2015):
        "In the event of a post-neuronal spike in a convolutional feature map,
        we uniformly increase the firing threshold of all the post-units 
        constituting the feature map. In the period of non-firing, the firing 
        threshold of the feature map exponentially decays over time."
    
    This prevents any single kernel from dominating the learning.
    """
    enabled: bool = True           # Enable adaptive thresholds
    theta_rest: float = 10.0       # Resting threshold (initial value)
    theta_plus: float = 0.05       # Threshold increase per spike
    tau_theta: float = 1e4         # Decay time constant (higher = slower)
    theta_max: float = 50.0        # Maximum threshold cap
    
    # Dead neuron recovery
    dead_neuron_recovery: bool = True   # Enable weight perturbation for dead neurons
    dead_threshold: float = 0.01        # Firing rate below this = dead
    recovery_boost: float = 0.1         # Weight perturbation magnitude
    
    def validate(self) -> None:
        """Validate parameters."""
        if self.theta_rest <= 0:
            raise ConfigurationError(
                f"theta_rest must be positive, got {self.theta_rest}"
            )
        if self.theta_plus < 0:
            raise ConfigurationError(
                f"theta_plus must be non-negative, got {self.theta_plus}"
            )
        if self.tau_theta <= 0:
            raise ConfigurationError(
                f"tau_theta must be positive, got {self.tau_theta}"
            )


@dataclass  
class ConvergenceParams:
    """
    Convergence monitoring parameters.
    
    Determines when a layer has learned sufficient features.
    """
    min_wins_per_neuron: int = 10  # Minimum WTA wins to be considered converged
    target_ratio: float = 0.95    # Target fraction of converged neurons
    patience: int = 5             # Epochs without improvement before stopping
    delta_threshold: float = 1e-4 # Weight change threshold for convergence
    check_interval: int = 100     # Check convergence every N samples
    
    def validate(self) -> None:
        """Validate parameters."""
        if self.min_wins_per_neuron < 1:
            raise ConfigurationError(
                f"min_wins_per_neuron must be >= 1, got {self.min_wins_per_neuron}"
            )
        if not (0 < self.target_ratio <= 1):
            raise ConfigurationError(
                f"target_ratio must be in (0, 1], got {self.target_ratio}"
            )


@dataclass
class DataParams:
    """
    Data loading parameters.
    """
    dataset: str = "ebssa"       # Dataset name
    data_root: str = "./data"    # Data directory
    sensor: str = "all"          # Sensor type: "DAVIS", "ATIS", or "all"
    include_unlabelled: bool = True  # Include unlabelled data (for unsupervised STDP)
    batch_size: int = 1          # Batch size (typically 1 for online STDP)
    num_workers: int = 4         # DataLoader workers
    pin_memory: bool = True      # Pin memory for faster GPU transfer
    
    # Event/spike parameters
    timestep_ms: float = 100.0   # Simulation timestep in milliseconds
    n_timesteps: int = 10        # Number of timesteps per sample
    input_height: int = 128      # Input height
    input_width: int = 128       # Input width
    input_channels: int = 1      # Input channels (1 for ON/OFF combined)
    
    # Processing
    normalize: bool = True       # Normalize event counts
    shuffle_train: bool = True   # Shuffle training data
    shuffle_val: bool = False    # Shuffle validation data
    train_ratio: float = 1.0     # Train/val split ratio (1.0 = use all for training)
    
    # Sliding window sampling - extract multiple samples per recording
    windows_per_recording: int = 1   # Number of windows per recording (1=original behavior)
    window_overlap: float = 0.5      # Overlap between consecutive windows (0.0-0.9)
    
    # Augmentation (safe options for satellites)
    augmentation: Optional[Dict[str, Any]] = None  # Augmentation config dict


@dataclass
class ModelParams:
    """
    Model architecture parameters.
    
    Paper Reference (IGARSS 2023):
        "5×5 convolution kernel" for Conv1/Conv2
        "7×7 final classification kernel"
        "4 features in layer 1, 36 features in layer 2"
    """
    n_classes: int = 2           # Number of output classes (2 for EBSSA binary)
    conv1_channels: int = 4      # Conv1 output channels
    conv2_channels: int = 36     # Conv2 output channels
    kernel_sizes: Tuple[int, int, int] = (5, 5, 7)  # Kernel sizes for each layer
    
    # Pooling configuration
    pool1_kernel: int = 2        # Pool1 kernel size
    pool1_stride: int = 2        # Pool1 stride
    pool2_kernel: int = 2        # Pool2 kernel size
    pool2_stride: int = 2        # Pool2 stride
    
    # Layer-wise leak (IGARSS 2023: "90% and 10% of threshold")
    thresholds: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    leaks: Tuple[float, float, float] = (9.0, 1.0, 0.0)  # 90%, 10%, 0%
    
    # Initialization
    use_dog_filters: bool = True # Use DoG filters for Conv1
    
    def __post_init__(self):
        # Convert lists to tuples (from YAML)
        if isinstance(self.kernel_sizes, list):
            self.kernel_sizes = tuple(self.kernel_sizes)
        if isinstance(self.thresholds, list):
            self.thresholds = tuple(self.thresholds)
        if isinstance(self.leaks, list):
            self.leaks = tuple(self.leaks)


@dataclass
class CheckpointParams:
    """
    Checkpointing parameters.
    """
    save_dir: str = "checkpoints"  # Subdirectory for checkpoints
    save_interval: int = 1         # Save every N epochs
    keep_last_n: int = 3           # Keep last N checkpoints
    save_best: bool = True         # Save best model separately
    save_on_interrupt: bool = True # Save on keyboard interrupt


@dataclass
class LoggingParams:
    """
    Logging parameters.
    """
    log_dir: str = "logs"         # Subdirectory for logs
    log_level: str = "INFO"       # Logging level
    log_interval: int = 100       # Log every N samples
    tensorboard: bool = True      # Enable TensorBoard
    wandb: bool = False           # Enable Weights & Biases
    wandb_project: str = "spikeseg"  # W&B project name
    print_model_summary: bool = True  # Print model summary at start


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Combines all parameter groups into single config object.
    """
    # Experiment settings
    experiment_name: str = "spikeseg_training"
    output_dir: str = "./runs"
    seed: int = 42
    device: str = "cuda"
    
    # Training duration
    max_epochs: int = 10
    max_samples_per_epoch: Optional[int] = None  # None = full dataset
    
    # Layer-wise training control
    train_conv1: bool = False  # Conv1 typically uses fixed DoG filters
    train_conv2: bool = True   # Train Conv2 via STDP
    train_conv3: bool = True   # Train Conv3 via STDP
    
    # Component configurations
    stdp: STDPParams = field(default_factory=STDPParams)
    wta: WTAParams = field(default_factory=WTAParams)
    homeostasis: HomeostasisParams = field(default_factory=HomeostasisParams)
    convergence: ConvergenceParams = field(default_factory=ConvergenceParams)
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    checkpoint: CheckpointParams = field(default_factory=CheckpointParams)
    logging: LoggingParams = field(default_factory=LoggingParams)
    
    # Memory optimization
    clear_cache_interval: int = 1000  # Clear CUDA cache every N samples
    gradient_checkpointing: bool = False  # Not used for STDP but kept for consistency
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        self.stdp.validate()
        self.wta.validate()
        self.homeostasis.validate()
        self.convergence.validate()
        
        if self.max_epochs < 1:
            raise ConfigurationError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must be non-negative, got {self.seed}")
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file.
        
        Returns:
            TrainingConfig instance.
        
        Raises:
            ConfigurationError: If file not found or invalid.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}") from e
        
        # Parse nested configs
        if 'stdp' in data and isinstance(data['stdp'], dict):
            data['stdp'] = STDPParams(**data['stdp'])
        if 'wta' in data and isinstance(data['wta'], dict):
            data['wta'] = WTAParams(**data['wta'])
        if 'homeostasis' in data and isinstance(data['homeostasis'], dict):
            data['homeostasis'] = HomeostasisParams(**data['homeostasis'])
        if 'convergence' in data and isinstance(data['convergence'], dict):
            data['convergence'] = ConvergenceParams(**data['convergence'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataParams(**data['data'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelParams(**data['model'])
        if 'checkpoint' in data and isinstance(data['checkpoint'], dict):
            data['checkpoint'] = CheckpointParams(**data['checkpoint'])
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingParams(**data['logging'])
        
        return cls(**data)
    
    @classmethod
    def from_paper(cls, paper: str = "igarss2023", **overrides) -> "TrainingConfig":
        """
        Create configuration from paper parameters.
        
        Args:
            paper: Paper identifier ('igarss2023', 'kheradpisheh2018').
            **overrides: Parameter overrides.
        
        Returns:
            TrainingConfig with paper parameters.
        """
        if paper.lower() == "igarss2023":
            config = cls(
                stdp=STDPParams(
                    lr_plus=0.04,
                    lr_minus=0.03,
                    weight_init_mean=0.8,
                    weight_init_std=0.01
                ),
                model=ModelParams(
                    conv1_channels=4,
                    conv2_channels=36,
                    kernel_sizes=(5, 5, 7),
                    leaks=(9.0, 1.0, 0.0),  # 90%, 10%, 0%
                    use_dog_filters=True
                ),
                max_epochs=1,  # "Features converge within 1 epoch"
            )
        elif paper.lower() == "kheradpisheh2018":
            config = cls(
                stdp=STDPParams(
                    lr_plus=0.004,
                    lr_minus=0.003,
                    weight_init_mean=0.8,
                    weight_init_std=0.02
                ),
                model=ModelParams(
                    conv1_channels=30,
                    conv2_channels=250,
                    kernel_sizes=(5, 17, 17),
                    leaks=(0.0, 0.0, 0.0),  # IF neurons
                    use_dog_filters=True
                )
            )
        else:
            raise ConfigurationError(
                f"Unknown paper '{paper}'. Available: 'igarss2023', 'kheradpisheh2018'"
            )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# LOGGING
# =============================================================================


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI color codes for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def setup_logging(
    output_dir: Path,
    config: LoggingParams,
    experiment_name: str
) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    
    Args:
        output_dir: Base output directory.
        config: Logging configuration.
        experiment_name: Name for log files.
    
    Returns:
        Configured root logger for spikeseg.
    """
    # Create log directory
    log_dir = output_dir / config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("spikeseg")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class LayerStats:
    """Statistics for a single layer during training."""
    name: str
    n_features: int
    wins: np.ndarray = field(default_factory=lambda: np.array([]))
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_sparsity: float = 0.0
    n_spikes: int = 0
    n_updates: int = 0
    convergence_ratio: float = 0.0
    
    def __post_init__(self):
        if len(self.wins) == 0:
            self.wins = np.zeros(self.n_features, dtype=np.int32)
    
    def update_weights(self, weights: torch.Tensor) -> None:
        """Update weight statistics."""
        w = weights.detach().cpu().float().numpy()
        self.weight_mean = float(np.mean(w))
        self.weight_std = float(np.std(w))
        self.weight_sparsity = float(np.mean(np.abs(w) < 0.01))
    
    def record_winners(self, winners: List) -> None:
        """Record WTA winners.
        
        Args:
            winners: List of feature indices (int) OR tuples (feature_idx, y, x)
        """
        for w in winners:
            # Handle both int and tuple formats
            feature_idx = w[0] if isinstance(w, (tuple, list)) else w
            if 0 <= feature_idx < self.n_features:
                self.wins[feature_idx] += 1
    
    def compute_convergence(self, min_wins: int) -> float:
        """Compute fraction of converged features."""
        self.convergence_ratio = float(np.mean(self.wins >= min_wins))
        return self.convergence_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'n_features': self.n_features,
            'weight_mean': self.weight_mean,
            'weight_std': self.weight_std,
            'weight_sparsity': self.weight_sparsity,
            'n_spikes': self.n_spikes,
            'n_updates': self.n_updates,
            'convergence_ratio': self.convergence_ratio,
            'converged_features': int((self.wins >= 10).sum()),
            'max_wins': int(self.wins.max()) if len(self.wins) > 0 else 0,
        }


class MetricsTracker:
    """
    Tracks comprehensive training metrics.
    
    Features:
        - Per-layer statistics
        - Epoch-level summaries
        - Best model tracking
        - Export to JSON
    """
    
    def __init__(self, layer_configs: List[Tuple[str, int]]):
        """
        Initialize metrics tracker.
        
        Args:
            layer_configs: List of (layer_name, n_features) tuples.
        """
        self.layers: Dict[str, LayerStats] = {}
        for name, n_feat in layer_configs:
            self.layers[name] = LayerStats(name=name, n_features=n_feat)
        
        self.epoch_history: List[Dict[str, Any]] = []
        self.best_convergence: Dict[str, float] = {name: 0.0 for name, _ in layer_configs}
        self.start_time = time.time()
        
        self.logger = logging.getLogger("spikeseg.metrics")
    
    def update_layer(
        self,
        layer_name: str,
        weights: torch.Tensor,
        winners: Optional[List[int]] = None,
        n_spikes: int = 0,
        n_updates: int = 0
    ) -> None:
        """Update metrics for a layer."""
        if layer_name not in self.layers:
            self.logger.warning(f"Unknown layer: {layer_name}")
            return
        
        stats = self.layers[layer_name]
        stats.update_weights(weights)
        stats.n_spikes += n_spikes
        stats.n_updates += n_updates
        
        if winners:
            stats.record_winners(winners)
    
    def compute_convergence(self, layer_name: str, min_wins: int) -> float:
        """Compute convergence ratio for a layer."""
        if layer_name not in self.layers:
            return 0.0
        
        ratio = self.layers[layer_name].compute_convergence(min_wins)
        
        if ratio > self.best_convergence[layer_name]:
            self.best_convergence[layer_name] = ratio
        
        return ratio
    
    def record_epoch(
        self,
        epoch: int,
        layer_name: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record epoch summary."""
        stats = self.layers[layer_name]
        elapsed = time.time() - self.start_time
        
        summary = {
            'epoch': epoch,
            'layer': layer_name,
            'elapsed_seconds': elapsed,
            **stats.to_dict()
        }
        
        if extra:
            summary.update(extra)
        
        self.epoch_history.append(summary)
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall training summary."""
        return {
            'total_time_seconds': time.time() - self.start_time,
            'layers': {name: s.to_dict() for name, s in self.layers.items()},
            'best_convergence': self.best_convergence,
            'n_epochs': len(self.epoch_history),
            'epoch_history': self.epoch_history[-10:],  # Last 10 epochs
        }
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        self.logger.info(f"Metrics saved: {path}")
    
    def reset_layer(self, layer_name: str) -> None:
        """Reset metrics for a layer."""
        if layer_name in self.layers:
            n_feat = self.layers[layer_name].n_features
            self.layers[layer_name] = LayerStats(name=layer_name, n_features=n_feat)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
        - Save model, optimizer, and training state
        - Keep only last N checkpoints
        - Save best model separately
        - Automatic latest symlink
    """
    
    def __init__(self, output_dir: Path, config: CheckpointParams):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Base output directory.
            config: Checkpoint configuration.
        """
        self.checkpoint_dir = output_dir / config.save_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = config.keep_last_n
        self.save_best = config.save_best
        self.best_metric = 0.0
        
        self.saved_checkpoints: List[Path] = []
        self.logger = logging.getLogger("spikeseg.checkpoint")
    
    def save(
        self,
        model: nn.Module,
        epoch: int,
        phase: TrainingPhase,
        layer_name: str,
        metrics: Dict[str, Any],
        config: TrainingConfig,
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save.
            epoch: Current epoch.
            phase: Current training phase.
            layer_name: Current layer being trained.
            metrics: Current metrics dict.
            config: Training configuration.
            is_best: Whether this is the best model so far.
            extra_state: Additional state to save.
        
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'phase': phase.value,
            'layer_name': layer_name,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch{epoch:03d}_{layer_name}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)
        self.logger.info(f"Saved checkpoint: {path.name}")
        
        # Update latest symlink
        latest_link = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.name)
        
        # Save best if applicable
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path.name}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return path
    
    def load(
        self,
        path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint. If None, loads latest.
            device: Device to load tensors to.
        
        Returns:
            Checkpoint dict, or None if not found.
        """
        if path is None:
            path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        path = Path(path)
        
        # Resolve symlink
        if path.is_symlink():
            path = path.resolve()
        
        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {path}")
            return None
        
        try:
            checkpoint = torch.load(path, map_location=device or 'cpu')
            self.logger.info(f"Loaded checkpoint: {path.name}")
            return checkpoint
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e
    
    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only last N."""
        # Don't delete best or latest
        regular = [
            p for p in self.saved_checkpoints
            if p.name not in ('checkpoint_best.pt', 'checkpoint_latest.pt')
        ]
        
        while len(regular) > self.keep_last_n:
            old = regular.pop(0)
            if old.exists():
                old.unlink()
                self.logger.debug(f"Removed old checkpoint: {old.name}")
            if old in self.saved_checkpoints:
                self.saved_checkpoints.remove(old)


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================


class GracefulShutdown:
    """
    Context manager for graceful shutdown on SIGINT/SIGTERM.
    
    Allows training loop to save checkpoint before exiting.
    """
    
    def __init__(self):
        self._shutdown_requested = False
        self._original_sigint = None
        self._original_sigterm = None
        self.logger = logging.getLogger("spikeseg.shutdown")
    
    def __enter__(self) -> "GracefulShutdown":
        self._original_sigint = signal.signal(signal.SIGINT, self._handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        return False
    
    def _handler(self, signum: int, frame) -> None:
        if self._shutdown_requested:
            self.logger.error("Forced shutdown. Exiting immediately!")
            sys.exit(1)
        
        self._shutdown_requested = True
        self.logger.warning(
            "Shutdown requested. Finishing current batch... "
            "(Press Ctrl+C again to force quit)"
        )
    
    @property
    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested


# =============================================================================
# ADAPTIVE THRESHOLD MANAGER (HOMEOSTASIS)
# =============================================================================


class AdaptiveThresholdManager:
    """
    Manages adaptive firing thresholds for homeostasis.
    
    Paper Reference (Lee et al. 2018 / Diehl & Cook 2015):
        - On spike: Increase threshold of entire feature map
        - On non-firing: Exponentially decay threshold toward rest
        
    This prevents any single kernel from dominating the learning by
    making frequently-firing neurons harder to activate.
    """
    
    def __init__(
        self,
        n_channels: int,
        theta_rest: float = 10.0,
        theta_plus: float = 0.05,
        tau_theta: float = 1e4,
        theta_max: float = 50.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            n_channels: Number of feature maps.
            theta_rest: Resting (initial) threshold.
            theta_plus: Threshold increase per spike.
            tau_theta: Decay time constant (higher = slower decay).
            theta_max: Maximum allowed threshold.
            device: Device for tensors.
        """
        self.n_channels = n_channels
        self.theta_rest = theta_rest
        self.theta_plus = theta_plus
        self.tau_theta = tau_theta
        self.theta_max = theta_max
        self.device = device or torch.device('cpu')
        
        # Initialize thresholds at rest
        self.thresholds = torch.full(
            (n_channels,), theta_rest, 
            device=self.device, dtype=torch.float32
        )
        
        # Track spike counts per channel for monitoring
        self.spike_counts = torch.zeros(n_channels, device=self.device)
        self.total_samples = 0
        
        self.logger = logging.getLogger("spikeseg.homeostasis")
    
    def update_on_spikes(self, spikes: torch.Tensor) -> None:
        """
        Update thresholds based on which feature maps spiked.
        
        Args:
            spikes: Spike tensor (out_channels, H, W) or (batch, out_channels, H, W).
        """
        # Sum spikes per channel
        if spikes.dim() == 4:
            # (batch, C, H, W) -> (C,)
            channel_spikes = (spikes.sum(dim=(0, 2, 3)) > 0).float()
        elif spikes.dim() == 3:
            # (C, H, W) -> (C,)
            channel_spikes = (spikes.sum(dim=(1, 2)) > 0).float()
        else:
            return
        
        # Move to same device
        channel_spikes = channel_spikes.to(self.device)
        
        # Increase threshold for channels that spiked
        self.thresholds += self.theta_plus * channel_spikes
        self.thresholds.clamp_(max=self.theta_max)
        
        # Track statistics
        self.spike_counts += channel_spikes
        self.total_samples += 1
    
    def decay(self, dt: float = 1.0) -> None:
        """
        Apply exponential decay toward resting threshold.
        
        Args:
            dt: Time step (typically 1.0 per sample).
        """
        # θ(t) = θ_rest + (θ - θ_rest) * exp(-dt/τ)
        decay_factor = np.exp(-dt / self.tau_theta)
        self.thresholds = self.theta_rest + (self.thresholds - self.theta_rest) * decay_factor
    
    def get_threshold(self, channel: Optional[int] = None) -> torch.Tensor:
        """Get current threshold(s)."""
        if channel is not None:
            return self.thresholds[channel]
        return self.thresholds
    
    def get_firing_rates(self) -> torch.Tensor:
        """Get average firing rate per channel."""
        if self.total_samples == 0:
            return torch.zeros(self.n_channels, device=self.device)
        return self.spike_counts / self.total_samples
    
    def get_dead_neurons(self, min_rate: float = 0.01) -> torch.Tensor:
        """
        Identify channels that rarely fire (dead neurons).
        
        Args:
            min_rate: Minimum firing rate threshold.
        
        Returns:
            Boolean mask of dead channels.
        """
        rates = self.get_firing_rates()
        return rates < min_rate
    
    def reset(self) -> None:
        """Reset thresholds to resting values."""
        self.thresholds.fill_(self.theta_rest)
        self.spike_counts.zero_()
        self.total_samples = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get threshold statistics."""
        return {
            'mean_threshold': float(self.thresholds.mean().item()),
            'min_threshold': float(self.thresholds.min().item()),
            'max_threshold': float(self.thresholds.max().item()),
            'n_dead_neurons': int(self.get_dead_neurons().sum().item()),
            'mean_firing_rate': float(self.get_firing_rates().mean().item()),
        }


# =============================================================================
# STDP TRAINER
# =============================================================================


class STDPTrainer:
    """
    STDP Trainer for SpikeSEG.
    
    Implements unsupervised layer-wise training using Spike-Timing
    Dependent Plasticity with Winner-Take-All competition.
    
    The training procedure follows the papers:
        1. Initialize Conv1 with fixed filters (DoG/Gabor)
        2. Train Conv2 features via STDP until convergence
        3. Train Conv3 (classification) via STDP
        4. Attach decoder with tied weights
    
    Features:
        - Layer-wise training
        - STDP weight updates with soft bounds
        - WTA lateral inhibition
        - Adaptive thresholds (homeostasis)
        - Convergence monitoring
        - Comprehensive logging
        - Checkpoint save/resume
    
    Example:
        >>> config = TrainingConfig.from_paper("igarss2023")
        >>> trainer = STDPTrainer(config)
        >>> summary = trainer.train()
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize STDP trainer.
        
        Args:
            config: Training configuration.
            model: Pre-created SpikeSEG model (optional).
            train_loader: Training data loader (optional).
            val_loader: Validation data loader (optional).
        """
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging first
        self.logger = setup_logging(
            self.output_dir,
            config.logging,
            config.experiment_name
        )
        
        self._log_header()
        
        # Setup device
        self.device = self._setup_device()
        
        # Set random seeds
        self._set_seeds()
        
        # Save config
        config.to_yaml(self.output_dir / "config.yaml")
        self.logger.info(f"Config saved: {self.output_dir / 'config.yaml'}")
        
        # Initialize components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create model if not provided
        if self.model is None:
            self.model = self._create_model()
        
        # Create data loaders if not provided
        if self.train_loader is None:
            self._setup_data_loaders()
        
        # Initialize metrics tracker
        layer_configs = [
            ('conv1', config.model.conv1_channels),
            ('conv2', config.model.conv2_channels),
            ('conv3', config.model.n_classes),
        ]
        self.metrics = MetricsTracker(layer_configs)
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            self.output_dir,
            config.checkpoint
        )
        
        # Training state
        self.current_epoch = 0
        self.current_phase = TrainingPhase.INITIALIZATION
        self.current_layer: Optional[str] = None
        self.global_step = 0
        
        # Adaptive threshold managers (homeostasis) for each layer
        # Paper: prevents any single kernel from dominating learning
        self.threshold_managers: Dict[str, AdaptiveThresholdManager] = {}
        homeostasis_cfg = config.homeostasis
        if homeostasis_cfg.enabled:
            for layer_name, n_channels in layer_configs:
                self.threshold_managers[layer_name] = AdaptiveThresholdManager(
                    n_channels=n_channels,
                    theta_rest=homeostasis_cfg.theta_rest,
                    theta_plus=homeostasis_cfg.theta_plus,
                    tau_theta=homeostasis_cfg.tau_theta,
                    theta_max=homeostasis_cfg.theta_max,
                    device=self.device
                )
            self.logger.info(
                f"Adaptive thresholds enabled "
                f"(θ_rest={homeostasis_cfg.theta_rest}, "
                f"Δθ={homeostasis_cfg.theta_plus}, "
                f"τ={homeostasis_cfg.tau_theta})"
            )
        
        # Dead neuron recovery settings from homeostasis config
        self.dead_neuron_recovery = homeostasis_cfg.dead_neuron_recovery
        self.dead_threshold = homeostasis_cfg.dead_threshold
        self.recovery_boost = homeostasis_cfg.recovery_boost
        
        # STDP learner (uses learning/stdp.py module)
        self.stdp_learner: Optional[STDPLearner] = None
        try:
            from spikeseg.learning.stdp import STDPLearner, STDPConfig as STDPLearnerConfig
            stdp_cfg = STDPLearnerConfig(
                lr_plus=config.stdp.lr_plus,
                lr_minus=config.stdp.lr_minus,
                weight_min=config.stdp.weight_min,
                weight_max=config.stdp.weight_max,
            )
            self.stdp_learner = STDPLearner(stdp_cfg)
            self.logger.info("Using STDPLearner from learning module")
        except ImportError:
            self.logger.warning("STDPLearner not available, using simplified STDP")
        
        # TensorBoard writer
        self.writer = None
        if config.logging.tensorboard:
            self._setup_tensorboard()
        
        # Log setup summary
        self._log_setup_summary()
    
    def _log_header(self) -> None:
        """Log training header."""
        self.logger.info("=" * 70)
        self.logger.info("  SpikeSEG STDP Training")
        self.logger.info("=" * 70)
        self.logger.info(f"  Experiment: {self.config.experiment_name}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info("=" * 70)
    
    def _setup_device(self) -> torch.device:
        """Setup and validate training device."""
        requested = self.config.device
        
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                return torch.device("cpu")
            
            # Parse device ID if specified
            if ":" in requested:
                device_id = int(requested.split(":")[1])
                if device_id >= torch.cuda.device_count():
                    self.logger.warning(
                        f"CUDA device {device_id} not available, using device 0"
                    )
                    device_id = 0
                device = torch.device(f"cuda:{device_id}")
            else:
                device = torch.device("cuda:0")
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            self.logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            
            return device
        else:
            self.logger.info("Using CPU")
            return torch.device("cpu")
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Random seed: {seed}")
    
    def _create_model(self) -> nn.Module:
        """Create SpikeSEG model from configuration."""
        self.logger.info("Creating SpikeSEG model...")
        
        # Import here to avoid circular imports
        from spikeseg.models import SpikeSEG, EncoderConfig, LayerConfig
        
        cfg = self.config.model
        
        # Create encoder config
        # NOTE: Pooling parameters from config.yaml are now properly applied
        # IGARSS 2023 doesn't specify pooling, but Kheradpisheh 2018 uses 7x7/6 for Pool1
        enc_config = EncoderConfig(
            input_channels=self.config.data.input_channels,
            conv1=LayerConfig(
                out_channels=cfg.conv1_channels,
                kernel_size=cfg.kernel_sizes[0],
                threshold=cfg.thresholds[0],
                leak=cfg.leaks[0]
            ),
            conv2=LayerConfig(
                out_channels=cfg.conv2_channels,
                kernel_size=cfg.kernel_sizes[1],
                threshold=cfg.thresholds[1],
                leak=cfg.leaks[1]
            ),
            conv3=LayerConfig(
                out_channels=cfg.n_classes,
                kernel_size=cfg.kernel_sizes[2],
                threshold=cfg.thresholds[2],
                leak=cfg.leaks[2]
            ),
            # Pooling configuration from config.yaml
            pool1_kernel_size=cfg.pool1_kernel,
            pool1_stride=cfg.pool1_stride,
            pool2_kernel_size=cfg.pool2_kernel,
            pool2_stride=cfg.pool2_stride,
        )
        
        model = SpikeSEG(config=enc_config, device=self.device)
        
        # Initialize weights for STDP
        self._initialize_weights(model)
        
        # Initialize Conv1 with DoG filters if configured
        if cfg.use_dog_filters:
            self._initialize_dog_filters(model)
            if not self.config.train_conv1:
                model.freeze_layer('conv1')
                self.logger.info("Conv1 frozen (using fixed DoG filters)")
        
        model.to(self.device)
        
        return model
    
    def _initialize_weights(self, model: nn.Module) -> None:
        """Initialize trainable layer weights for STDP."""
        stdp = self.config.stdp
        
        for layer_name in ['conv2', 'conv3']:
            layer = getattr(model.encoder, layer_name)
            
            with torch.no_grad():
                layer.conv.weight.normal_(stdp.weight_init_mean, stdp.weight_init_std)
                layer.conv.weight.clamp_(stdp.weight_min, stdp.weight_max)
        
        self.logger.info(
            f"Weights initialized: mean={stdp.weight_init_mean:.2f}, "
            f"std={stdp.weight_init_std:.3f}"
        )
    
    def _initialize_dog_filters(self, model: nn.Module) -> None:
        """Initialize Conv1 with Difference of Gaussians filters."""
        try:
            from spikeseg.core.functional import create_dog_filters
            
            k = self.config.model.kernel_sizes[0]
            n_in = self.config.data.input_channels
            n_out = self.config.model.conv1_channels
            
            # Create base DoG filters [2, 1, k, k] (ON/OFF center-surround)
            dog = create_dog_filters(size=k, sigma_center=1.0, sigma_surround=2.0)
            n_dog = dog.shape[0]  # 2 (ON and OFF)
            
            # Expand to required number of output filters
            if n_out <= n_dog:
                filters = dog[:n_out]
            else:
                # Duplicate and scale
                repeats = (n_out + n_dog - 1) // n_dog
                filters = dog.repeat(repeats, 1, 1, 1)[:n_out]
            
            # Now filters is [n_out, 1, k, k], need [n_out, n_in, k, k]
            if n_in > 1:
                # Replicate the filter across input channels
                # Each output filter responds to all input channels equally
                filters = filters.repeat(1, n_in, 1, 1)
            
            with torch.no_grad():
                model.encoder.conv1.conv.weight.copy_(filters)
            
            self.logger.info(f"DoG filters initialized: {filters.shape}")
            
        except ImportError as e:
            self.logger.warning(f"Could not import DoG filters: {e}")
            self.logger.warning("Using random initialization for Conv1")
    
    def _setup_data_loaders(self) -> None:
        """Setup data loaders using spikeseg.data module."""
        self.logger.info("Setting up data loaders...")

        try:
            from spikeseg.data import (
                get_dataset, create_dataloader,
                EBSSADataset, SyntheticEventDataset
            )

            data_cfg = self.config.data

            # Try to load real dataset first
            if data_cfg.dataset.lower() == "ebssa":
                try:
                    # Setup augmentation if configured
                    from spikeseg.data import EventAugmentation
                    aug = None
                    aug_cfg = data_cfg.augmentation
                    if aug_cfg and aug_cfg.get('enabled', False):
                        aug = EventAugmentation(
                            flip_horizontal=aug_cfg.get('flip_horizontal', False),
                            flip_polarity=aug_cfg.get('flip_polarity', False),
                            drop_rate=aug_cfg.get('drop_rate', 0.0),
                            noise_rate=aug_cfg.get('noise_rate', 0.0),
                            random_crop=aug_cfg.get('random_crop', None)
                        )
                        self.logger.info(f"Augmentation enabled: flip_h={aug_cfg.get('flip_horizontal')}, "
                                        f"flip_p={aug_cfg.get('flip_polarity')}, "
                                        f"drop={aug_cfg.get('drop_rate')}, noise={aug_cfg.get('noise_rate')}")
                    
                    # Get sliding window parameters
                    windows_per_recording = getattr(data_cfg, 'windows_per_recording', 1)
                    window_overlap = getattr(data_cfg, 'window_overlap', 0.5)
                    
                    train_dataset = EBSSADataset(
                        root=data_cfg.data_root,
                        split="train",
                        sensor=data_cfg.sensor,
                        n_timesteps=data_cfg.n_timesteps,
                        height=data_cfg.input_height,
                        width=data_cfg.input_width,
                        polarity_channels=(data_cfg.input_channels == 2),
                        normalize=data_cfg.normalize,
                        include_unlabelled=data_cfg.include_unlabelled,
                        train_ratio=data_cfg.train_ratio,
                        augmentation=aug,
                        windows_per_recording=windows_per_recording,
                        window_overlap=window_overlap
                    )
                    val_dataset = EBSSADataset(
                        root=data_cfg.data_root,
                        split="val",
                        sensor=data_cfg.sensor,
                        n_timesteps=data_cfg.n_timesteps,
                        height=data_cfg.input_height,
                        width=data_cfg.input_width,
                        polarity_channels=(data_cfg.input_channels == 2),
                        normalize=data_cfg.normalize,
                        include_unlabelled=False,  # Val uses only labelled
                        train_ratio=data_cfg.train_ratio,
                        augmentation=None,  # No augmentation for validation
                        windows_per_recording=1,  # No sliding window for validation
                        window_overlap=0.0
                    )
                    n_recordings = len(train_dataset.recordings)
                    self.logger.info(f"Loaded EBSSA dataset: {n_recordings} recordings × {windows_per_recording} windows = {len(train_dataset)} train samples, {len(val_dataset)} val")
                except Exception as e:
                    self.logger.warning(f"Failed to load EBSSA dataset: {e}")
                    self.logger.info("Falling back to synthetic dataset for testing")
                    train_dataset = SyntheticEventDataset(
                        n_samples=100,
                        n_events_per_sample=1000,
                        height=data_cfg.input_height,
                        width=data_cfg.input_width,
                        n_timesteps=data_cfg.n_timesteps,
                        polarity_channels=(data_cfg.input_channels == 2),
                        normalize=data_cfg.normalize,
                        seed=self.config.seed
                    )
                    val_dataset = SyntheticEventDataset(
                        n_samples=20,
                        n_events_per_sample=1000,
                        height=data_cfg.input_height,
                        width=data_cfg.input_width,
                        n_timesteps=data_cfg.n_timesteps,
                        polarity_channels=(data_cfg.input_channels == 2),
                        normalize=data_cfg.normalize,
                        seed=self.config.seed + 1000
                    )

            elif data_cfg.dataset.lower() == "synthetic":
                train_dataset = SyntheticEventDataset(
                    n_samples=data_cfg.batch_size * 100,
                    n_events_per_sample=1000,
                    height=data_cfg.input_height,
                    width=data_cfg.input_width,
                    n_timesteps=data_cfg.n_timesteps,
                    polarity_channels=(data_cfg.input_channels == 2),
                    normalize=data_cfg.normalize,
                    seed=self.config.seed
                )
                val_dataset = SyntheticEventDataset(
                    n_samples=data_cfg.batch_size * 20,
                    n_events_per_sample=1000,
                    height=data_cfg.input_height,
                    width=data_cfg.input_width,
                    n_timesteps=data_cfg.n_timesteps,
                    polarity_channels=(data_cfg.input_channels == 2),
                    normalize=data_cfg.normalize,
                    seed=self.config.seed + 1000
                )
                self.logger.info(f"Using synthetic dataset: {len(train_dataset)} train, {len(val_dataset)} val")

            else:
                # Try generic dataset loading
                train_dataset = get_dataset(
                    name=data_cfg.dataset,
                    root=data_cfg.data_root,
                    split="train",
                    n_timesteps=data_cfg.n_timesteps,
                    height=data_cfg.input_height,
                    width=data_cfg.input_width
                )
                val_dataset = get_dataset(
                    name=data_cfg.dataset,
                    root=data_cfg.data_root,
                    split="val",
                    n_timesteps=data_cfg.n_timesteps,
                    height=data_cfg.input_height,
                    width=data_cfg.input_width
                )

            # Create data loaders
            self.train_loader = create_dataloader(
                train_dataset,
                batch_size=data_cfg.batch_size,
                shuffle=data_cfg.shuffle_train,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory
            )
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=data_cfg.batch_size,
                shuffle=data_cfg.shuffle_val,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory
            )

            self.logger.info(f"Data loaders ready: batch_size={data_cfg.batch_size}")

        except ImportError as e:
            self.logger.warning(f"Could not import data module: {e}")
            self.logger.warning("Using dummy data generator")
            self.train_loader = None
            self.val_loader = None
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_dir = self.output_dir / "tensorboard"
            self.writer = SummaryWriter(tb_dir)
            self.logger.info(f"TensorBoard: {tb_dir}")
            
        except ImportError:
            self.logger.warning("TensorBoard not available")
            self.writer = None
    
    def _log_setup_summary(self) -> None:
        """Log training setup summary."""
        cfg = self.config
        
        self.logger.info("-" * 50)
        self.logger.info("Configuration Summary:")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Max epochs: {cfg.max_epochs}")
        self.logger.info(f"  Model: {cfg.model.conv1_channels} → "
                        f"{cfg.model.conv2_channels} → {cfg.model.n_classes}")
        self.logger.info(f"  Kernels: {cfg.model.kernel_sizes}")
        self.logger.info(f"  STDP: α⁺={cfg.stdp.lr_plus}, α⁻={cfg.stdp.lr_minus}")
        self.logger.info(f"  WTA mode: {cfg.wta.mode}")
        
        if self.model is not None:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"  Parameters: {total:,} ({trainable:,} trainable)")
        
        self.logger.info("-" * 50)
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def train(self, resume_from: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run full training procedure.
        
        Args:
            resume_from: Path to checkpoint to resume from.
        
        Returns:
            Training summary dict.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("  Starting STDP Training")
        self.logger.info("=" * 70 + "\n")
        
        # Resume if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        
        # Determine layers to train
        layers_to_train = []
        if self.config.train_conv1:
            layers_to_train.append('conv1')
        if self.config.train_conv2:
            layers_to_train.append('conv2')
        if self.config.train_conv3:
            layers_to_train.append('conv3')
        
        self.logger.info(f"Layers to train: {layers_to_train}")
        
        # Training with graceful shutdown
        with GracefulShutdown() as shutdown:
            try:
                for layer_name in layers_to_train:
                    if shutdown.should_stop:
                        self.logger.warning("Training interrupted")
                        break
                    
                    self.current_layer = layer_name
                    self.current_phase = TrainingPhase[layer_name.upper()]
                    
                    self.logger.info("\n" + "=" * 60)
                    self.logger.info(f"  Training Layer: {layer_name}")
                    self.logger.info("=" * 60 + "\n")
                    
                    # Train layer
                    converged = self._train_layer(layer_name, shutdown)
                    
                    # Freeze after training
                    self.model.freeze_layer(layer_name)
                    self.logger.info(f"Layer {layer_name} frozen")
                    
                    if converged:
                        self.logger.info(f"✓ Layer {layer_name} converged!")
                    else:
                        self.logger.warning(f"✗ Layer {layer_name} did not fully converge")
                
                self.current_phase = TrainingPhase.COMPLETED
                
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                self.logger.error(traceback.format_exc())
                
                # Emergency checkpoint
                if self.config.checkpoint.save_on_interrupt:
                    self._save_checkpoint(emergency=True)
                
                raise TrainingError(f"Training failed: {e}") from e
        
        # Final summary
        summary = self.metrics.get_summary()
        self.metrics.save(self.output_dir / "metrics_final.json")
        
        self._log_final_summary(summary)
        
        # Cleanup
        if self.writer:
            self.writer.close()
        
        return summary
    
    def _train_layer(self, layer_name: str, shutdown: GracefulShutdown) -> bool:
        """
        Train a single layer using STDP with homeostasis.
        
        Args:
            layer_name: Name of layer to train.
            shutdown: Shutdown handler.
        
        Returns:
            True if converged, False otherwise.
        """
        layer = getattr(self.model.encoder, layer_name)
        conv_cfg = self.config.convergence
        
        # Reset metrics for this layer
        self.metrics.reset_layer(layer_name)
        
        # Reset threshold manager for this layer
        if layer_name in self.threshold_managers:
            thresh_mgr = self.threshold_managers[layer_name]
            thresh_mgr.reset()
            self.logger.info(
                f"Reset adaptive thresholds for {layer_name}: "
                f"θ_rest={thresh_mgr.theta_rest}, θ_plus={thresh_mgr.theta_plus}, "
                f"θ_max={thresh_mgr.theta_max}, actual_mean={thresh_mgr.thresholds.mean().item():.2f}"
            )
        
        best_convergence = 0.0
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            if shutdown.should_stop:
                break
            
            self.current_epoch = epoch
            epoch_start = time.time()
            
            self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            
            # Reset model state
            self.model.reset_state()
            
            # Run training epoch
            epoch_stats = self._train_epoch(layer_name, shutdown)
            
            if shutdown.should_stop:
                break
            
            # Compute convergence
            convergence = self.metrics.compute_convergence(
                layer_name,
                conv_cfg.min_wins_per_neuron
            )
            
            # Epoch timing
            epoch_time = time.time() - epoch_start
            samples_per_sec = epoch_stats['n_samples'] / max(epoch_time, 1e-6)
            
            # Log epoch summary
            self.logger.info(
                f"  Epoch complete: {epoch_stats['n_samples']} samples, "
                f"{epoch_stats['n_spikes']} spikes, "
                f"{convergence*100:.1f}% converged, "
                f"{epoch_time:.1f}s ({samples_per_sec:.1f} samples/s)"
            )
            
            # Record metrics
            self.metrics.record_epoch(
                epoch,
                layer_name,
                extra={
                    'epoch_time': epoch_time,
                    'samples_per_sec': samples_per_sec,
                    **epoch_stats
                }
            )
            
            # TensorBoard logging
            if self.writer:
                self._log_to_tensorboard(epoch, layer_name, convergence, epoch_stats)
            
            # Checkpointing
            is_best = convergence > best_convergence
            if is_best:
                best_convergence = convergence
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % self.config.checkpoint.save_interval == 0:
                self._save_checkpoint(is_best=is_best)
            
            # Early stopping checks
            if convergence >= conv_cfg.target_ratio:
                self.logger.info(
                    f"Target convergence reached: {convergence*100:.1f}%"
                )
                return True
            
            if patience_counter >= conv_cfg.patience:
                self.logger.warning(
                    f"No improvement for {conv_cfg.patience} epochs. "
                    f"Best: {best_convergence*100:.1f}%"
                )
                break
        
        return best_convergence >= conv_cfg.target_ratio
    
    def _train_epoch(
        self,
        layer_name: str,
        shutdown: GracefulShutdown
    ) -> Dict[str, Any]:
        """
        Run single training epoch.
        
        Args:
            layer_name: Layer to train.
            shutdown: Shutdown handler.
        
        Returns:
            Epoch statistics dict.
        """
        stats = {
            'n_samples': 0,
            'n_spikes': 0,
            'n_updates': 0,
            'n_winners': 0,
        }
        
        # Use training data or dummy data
        if self.train_loader is not None:
            data_iter = iter(self.train_loader)
        else:
            data_iter = self._dummy_data_generator()
        
        for batch_idx, batch in enumerate(data_iter):
            if shutdown.should_stop:
                break
            
            # Check sample limit
            if (self.config.max_samples_per_epoch is not None and
                stats['n_samples'] >= self.config.max_samples_per_epoch):
                break
            
            # Process batch
            result = self._train_step(batch, layer_name)
            
            # Update stats
            stats['n_samples'] += 1
            stats['n_spikes'] += result['n_spikes']
            stats['n_updates'] += result['n_updates']
            stats['n_winners'] += len(result.get('winners', []))
            
            # Update metrics
            layer = getattr(self.model.encoder, layer_name)
            self.metrics.update_layer(
                layer_name,
                layer.conv.weight,
                winners=result.get('winners'),
                n_spikes=result['n_spikes'],
                n_updates=result['n_updates']
            )
            
            # Periodic logging
            if (batch_idx + 1) % self.config.logging.log_interval == 0:
                self._log_batch_progress(batch_idx, stats, layer_name)
            
            # Memory management
            if (batch_idx + 1) % self.config.clear_cache_interval == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            self.global_step += 1
        
        return stats
    
    def _train_step(
        self,
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        layer_name: str
    ) -> Dict[str, Any]:
        """
        Single STDP training step with homeostasis.
        
        Implements full paper mechanism:
            1. Forward pass to get spikes
            2. WTA competition to find winners
            3. STDP weight update for winners
            4. Adaptive threshold update (homeostasis)
            5. Dead neuron recovery (if enabled)
        
        Args:
            batch: Input batch (tensor or dict with 'spikes' key).
            layer_name: Layer to train.
        
        Returns:
            Step result dict.
        """
        # Extract input tensor
        if isinstance(batch, dict):
            x = batch.get('spikes', batch.get('events', batch.get('input')))
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        if x is None:
            return {'n_spikes': 0, 'n_updates': 0, 'winners': [], 'homeostasis': {}}
        
        x = x.to(self.device)

        # Handle different input dimensions:
        # - Dataset returns: (T, C, H, W) voxel grid per sample
        # - DataLoader batches to: (B, T, C, H, W)
        # - Encoder expects: (T, B, C, H, W) or (B, C, H, W) for single frame
        if x.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
        elif x.dim() == 5:
            # (B, T, C, H, W) -> (T, B, C, H, W)
            # DataLoader batches along dim 0, but encoder expects time on dim 0
            x = x.permute(1, 0, 2, 3, 4)
        
        result = {
            'n_spikes': 0,
            'n_updates': 0,
            'winners': [],
            'homeostasis': {}
        }
        
        with torch.no_grad():
            # ============================================================
            # BUILD ADAPTIVE THRESHOLDS FOR FORWARD PASS
            # Paper (Diehl & Cook 2015, Lee et al. 2018):
            # "On spike: increase threshold of entire feature map"
            # This prevents any single kernel from dominating the learning
            # ============================================================
            layer_thresholds = {}
            for lname, threshold_mgr in self.threshold_managers.items():
                layer_thresholds[lname] = threshold_mgr.get_threshold()
            
            # Forward pass through encoder WITH adaptive thresholds
            enc_output = self.model.encode(x, layer_thresholds=layer_thresholds)
            
            # Get layer spikes
            if hasattr(enc_output, 'layer_spikes') and layer_name in enc_output.layer_spikes:
                post_spikes = enc_output.layer_spikes[layer_name]
            else:
                post_spikes = enc_output.classification_spikes
            
            # Handle 5D (T, B, C, H, W) output
            if post_spikes.dim() == 5:
                post_spikes = post_spikes.sum(dim=0)  # Sum over time
            
            result['n_spikes'] = int(post_spikes.sum().item())
            
            # ============================================================
            # ADAPTIVE THRESHOLD UPDATE (Homeostasis)
            # Paper: "In the event of a post-neuronal spike in a convolutional
            # feature map, we uniformly increase the firing threshold"
            # Now these updated thresholds will be USED in the next forward pass!
            # ============================================================
            if layer_name in self.threshold_managers:
                threshold_mgr = self.threshold_managers[layer_name]
                
                # Update thresholds based on which channels spiked
                threshold_mgr.update_on_spikes(post_spikes)
                
                # Apply decay between samples
                threshold_mgr.decay(dt=1.0)
                
                # Record stats
                result['homeostasis'] = threshold_mgr.get_stats()
            
            # Get spike times from encoder output (new feature)
            layer_spike_times = getattr(enc_output, 'layer_spike_times', None) or {}
            post_spike_times = layer_spike_times.get(layer_name)

            # Find WTA winners using spike times for proper first-spike competition
            winners = self._find_wta_winners(post_spikes, post_spike_times)
            result['winners'] = [(w[0], w[1], w[2]) for w in winners]  # Ensure tuple format

            # Apply STDP if we have winners
            if winners:
                # Get pre-synaptic spike times
                pre_spike_times = self._get_pre_spike_times(layer_spike_times, layer_name)

                if pre_spike_times is not None and post_spike_times is not None:
                    result['n_updates'] = self._apply_stdp_update(
                        layer_name, pre_spike_times, post_spike_times, winners
                    )
            
            # ============================================================
            # DEAD NEURON RECOVERY
            # If some feature maps never fire, add noise to their weights
            # ============================================================
            if self.dead_neuron_recovery and layer_name in self.threshold_managers:
                threshold_mgr = self.threshold_managers[layer_name]
                dead_mask = threshold_mgr.get_dead_neurons(self.dead_threshold)
                n_dead = int(dead_mask.sum().item())
                
                if n_dead > 0 and threshold_mgr.total_samples > 100:  # Wait for warmup
                    layer = getattr(self.model.encoder, layer_name)
                    weights = layer.conv.weight
                    
                    # Add small perturbation to dead neuron weights
                    for c in range(dead_mask.shape[0]):
                        if dead_mask[c]:
                            noise = torch.randn_like(weights[c]) * self.recovery_boost
                            weights[c] = torch.clamp(
                                weights[c] + noise,
                                self.config.stdp.weight_min,
                                self.config.stdp.weight_max
                            )
                    
                    result['n_recovered'] = n_dead
        
        return result
    
    def _get_pre_spikes(
        self,
        input_spikes: torch.Tensor,
        enc_output,
        layer_name: str
    ) -> Optional[torch.Tensor]:
        """Get pre-synaptic spikes for a layer."""
        if layer_name == 'conv1':
            return input_spikes

        layer_spikes = enc_output.layer_spikes if hasattr(enc_output, 'layer_spikes') else {}

        if layer_name == 'conv2':
            # Pre = output of pool1 or conv1
            pre = layer_spikes.get('pool1', layer_spikes.get('conv1'))
        else:  # conv3
            # Pre = output of pool2 or conv2
            pre = layer_spikes.get('pool2', layer_spikes.get('conv2'))

        if pre is not None and pre.dim() == 5:
            pre = pre.sum(dim=0)  # Sum over time

        return pre

    def _get_pre_spike_times(
        self,
        layer_spike_times: Dict[str, torch.Tensor],
        layer_name: str
    ) -> Optional[torch.Tensor]:
        """
        Get pre-synaptic spike times for STDP.

        Args:
            layer_spike_times: Dict mapping layer name to spike times tensor.
            layer_name: Name of the post-synaptic layer.

        Returns:
            Pre-synaptic spike times tensor (B, C, H, W), or None.
        """
        if layer_name == 'conv1':
            # For conv1, pre-synaptic is input - we don't have spike times
            # Return None to fall back to activity-based STDP
            return None

        if layer_name == 'conv2':
            # Pre = output of pool1 (or conv1 if no pooling)
            return layer_spike_times.get('pool1', layer_spike_times.get('conv1'))
        else:  # conv3
            # Pre = output of pool2 (or conv2 if no pooling)
            return layer_spike_times.get('pool2', layer_spike_times.get('conv2'))
    
    def _find_wta_winners(
        self,
        spikes: torch.Tensor,
        spike_times: Optional[torch.Tensor] = None
    ) -> List[Tuple[int, int, int]]:
        """
        Find Winner-Take-All winners using PROPER competition (VECTORIZED).

        Paper Reference (Kheradpisheh 2018):
            "When a neuron fires, in a specific location, it inhibits other
            neurons in that location belonging to other neuronal maps (i.e.,
            resets their potentials to zero) and does not allow them to fire
            until the next image is shown. Also, it PREVENTS OTHER NEURONS,
            AT ALL LOCATIONS, IN ITS OWN MAP TO FIRE."

        This implements BOTH competitions:
            1. INTRA-MAP (global): Only ONE neuron per feature map wins
               (the first to fire across all spatial locations)
            2. INTER-MAP (local): At each location, only ONE feature wins

        CRITICAL FIX: Previous implementation only did inter-map competition,
        allowing multiple spatial locations to update the same feature map's
        weights. This caused all features to converge to averaged/same patterns.

        Args:
            spikes: Spike tensor (B, C, H, W).
            spike_times: First spike times (B, C, H, W). -1 = never fired.

        Returns:
            List of (feature_idx, y, x) tuples for winning neurons.
            At most ONE winner per feature map (intra-map competition).
        """
        wta_cfg = self.config.wta

        if spikes.dim() != 4:
            # Fallback for non-4D tensors
            spike_counts = spikes.sum() if spikes.dim() < 4 else spikes.sum(dim=(0, 2, 3))
            winners_flat = torch.nonzero(spike_counts > 0, as_tuple=True)[0]
            return [(int(w), 0, 0) for w in winners_flat.tolist()]

        B, C, H, W = spikes.shape
        device = spikes.device

        if spike_times is not None and wta_cfg.mode in ('global', 'both'):
            # VECTORIZED first-spike WTA with PROPER intra-map competition
            # spike_times: (B, C, H, W), -1 means never fired

            # Use first batch element (typical for online STDP)
            times = spike_times[0]  # (C, H, W)

            # Replace -1 (never fired) with inf so argmin works correctly
            times_for_argmin = torch.where(
                times >= 0,
                times,
                torch.tensor(float('inf'), device=device, dtype=times.dtype)
            )  # (C, H, W)

            # ================================================================
            # INTRA-MAP COMPETITION (CRITICAL FIX)
            # For each feature map, find the FIRST neuron to fire
            # (minimum spike time across all spatial locations)
            # ================================================================
            times_flat = times_for_argmin.view(C, -1)  # (C, H*W)

            # Find earliest spike time for each feature map
            min_time_per_feature, min_idx_per_feature = times_flat.min(dim=1)  # (C,), (C,)

            # Which features actually fired? (min time < inf)
            feature_fired = min_time_per_feature < float('inf')  # (C,)

            # Get winning feature indices
            winning_feature_indices = torch.nonzero(feature_fired, as_tuple=True)[0]  # (N_winners,)

            if len(winning_feature_indices) == 0:
                return []

            # For each winning feature, get the spatial location of the first spike
            winners = []
            for feat_idx in winning_feature_indices.tolist():
                flat_idx = min_idx_per_feature[feat_idx].item()
                winner_y = flat_idx // W
                winner_x = flat_idx % W
                winners.append((feat_idx, winner_y, winner_x))

            # ================================================================
            # INTER-MAP COMPETITION (optional, for mode='both')
            # At each winning location, only keep the feature that fired first
            # This handles cases where multiple features have their first spike
            # at the same spatial location
            # ================================================================
            if wta_cfg.mode == 'both' and len(winners) > 1:
                # Group winners by location
                location_to_winners = {}
                for feat, y, x in winners:
                    loc = (y, x)
                    spike_time = times_for_argmin[feat, y, x].item()
                    if loc not in location_to_winners:
                        location_to_winners[loc] = []
                    location_to_winners[loc].append((feat, spike_time))

                # Keep only the earliest spike at each location
                filtered_winners = []
                for (y, x), feat_times in location_to_winners.items():
                    # Sort by spike time and keep earliest
                    feat_times.sort(key=lambda ft: ft[1])
                    earliest_feat = feat_times[0][0]
                    filtered_winners.append((earliest_feat, y, x))

                winners = filtered_winners

        else:
            # VECTORIZED fallback: use spike count with intra-map competition
            # Use first batch element
            spikes_b = spikes[0]  # (C, H, W)

            # For each feature map, find the location with most spikes
            spikes_flat = spikes_b.view(C, -1)  # (C, H*W)
            max_spikes_per_feature, max_idx_per_feature = spikes_flat.max(dim=1)  # (C,), (C,)

            # Which features actually fired?
            feature_fired = max_spikes_per_feature > 0  # (C,)

            # Get winning feature indices
            winning_feature_indices = torch.nonzero(feature_fired, as_tuple=True)[0]

            if len(winning_feature_indices) == 0:
                return []

            # For each winning feature, get the spatial location
            winners = []
            for feat_idx in winning_feature_indices.tolist():
                flat_idx = max_idx_per_feature[feat_idx].item()
                winner_y = flat_idx // W
                winner_x = flat_idx % W
                winners.append((feat_idx, winner_y, winner_x))

        return winners
    
    def _apply_stdp_update(
        self,
        layer_name: str,
        pre_spike_times: torch.Tensor,
        post_spike_times: torch.Tensor,
        winners: List[Tuple[int, int, int]]
    ) -> int:
        """
        Apply STDP weight update using proper spike timing (VECTORIZED).

        Paper STDP Rule (Kheradpisheh 2018, Equation 3):
            Δw_ij = a⁺ · w_ij · (1 - w_ij)    if t_pre ≤ t_post  (LTP)
            Δw_ij = -a⁻ · w_ij · (1 - w_ij)   if t_pre > t_post  (LTD)

        Key insight: "The exact time difference does not affect the weight
        change, but only its sign is considered."

        This is a fully vectorized implementation that processes all winners
        and all weights in parallel using PyTorch tensor operations.
        ~10-100x faster than the loop-based version.

        Args:
            layer_name: Name of layer to update.
            pre_spike_times: Pre-synaptic first spike times (B, C, H, W).
                            -1 means neuron never fired.
            post_spike_times: Post-synaptic first spike times (B, C, H, W).
                             -1 means neuron never fired.
            winners: List of (feature_idx, y, x) tuples for winning neurons.

        Returns:
            Number of weights updated.
        """
        if not winners:
            return 0

        layer = getattr(self.model.encoder, layer_name)
        stdp_cfg = self.config.stdp

        with torch.no_grad():
            weights = layer.conv.weight  # (out_ch, in_ch, kH, kW)
            out_ch, in_ch, kH, kW = weights.shape
            pad = kH // 2  # Assuming same padding

            # Get input dimensions
            if pre_spike_times.dim() == 4:
                B, C_in, H_in, W_in = pre_spike_times.shape
            else:
                self.logger.warning(f"Unexpected pre_spike_times dim: {pre_spike_times.dim()}")
                return 0

            device = weights.device
            
            # Pad pre_spike_times for easy receptive field extraction
            # Use inf for padded regions (they never spike, so always LTD)
            pre_padded = torch.nn.functional.pad(
                pre_spike_times[0],  # Use first batch element (C_in, H, W)
                (pad, pad, pad, pad),
                mode='constant',
                value=float('inf')
            )  # (C_in, H_in + 2*pad, W_in + 2*pad)

            # Accumulate weight updates across all winners
            delta_w = torch.zeros_like(weights)
            n_updates = 0

            # Group winners by feature map for efficiency
            from collections import defaultdict
            winners_by_feat = defaultdict(list)
            for feat, y, x in winners:
                if feat < out_ch:
                    winners_by_feat[feat].append((y, x))

            # Process each feature map's winners
            for feat, locations in winners_by_feat.items():
                if not locations:
                    continue

                # Get weights for this feature map
                w = weights[feat]  # (in_ch, kH, kW)
                
                # Compute soft bounds once for this feature
                soft_bounds = w * (1.0 - w)  # (in_ch, kH, kW)

                # Process all winners for this feature
                for winner_y, winner_x in locations:
                    # Get post-synaptic spike time
                    t_post = post_spike_times[0, feat, winner_y, winner_x]
                    
                    if t_post < 0:
                        continue  # Winner didn't fire

                    # Extract receptive field from padded pre_spike_times
                    # After padding, the receptive field for (y, x) starts at (y, x)
                    rf_y_start = winner_y
                    rf_x_start = winner_x
                    pre_rf = pre_padded[:, rf_y_start:rf_y_start + kH, rf_x_start:rf_x_start + kW]
                    # Shape: (C_in, kH, kW) - matches weight shape

                    # Vectorized STDP computation
                    # LTP: pre fires before or at post (t_pre <= t_post AND t_pre >= 0)
                    # LTD: pre fires after post OR pre never fires (t_pre > t_post OR t_pre < 0)
                    # Since we used inf for padding and never-fired neurons, t_pre > t_post handles both
                    
                    ltp_mask = (pre_rf <= t_post) & (pre_rf >= 0)  # Pre fired before/at post
                    ltd_mask = ~ltp_mask  # Everything else (including inf and -1)

                    # Compute delta for this winner
                    delta_ltp = stdp_cfg.lr_plus * soft_bounds * ltp_mask.float()
                    delta_ltd = -stdp_cfg.lr_minus * soft_bounds * ltd_mask.float()
                    
                    delta_w[feat] += delta_ltp + delta_ltd
                    n_updates += in_ch * kH * kW

            # NOTE: With proper intra-map competition (fixed in _find_wta_winners),
            # there should be at most ONE winner per feature map.
            # This averaging is kept as a safety check but should rarely trigger.
            for feat, locations in winners_by_feat.items():
                n_winners = len(locations)
                if n_winners > 1:
                    self.logger.debug(f"Multiple winners for feature {feat}: {n_winners} (unexpected)")
                    delta_w[feat] /= n_winners

            # Apply updates and clamp
            weights.add_(delta_w)
            weights.clamp_(stdp_cfg.weight_min, stdp_cfg.weight_max)

        return n_updates
    
    # =========================================================================
    # LOGGING AND CHECKPOINTING
    # =========================================================================
    
    def _log_batch_progress(
        self,
        batch_idx: int,
        stats: Dict[str, Any],
        layer_name: str
    ) -> None:
        """Log batch progress with homeostasis info."""
        convergence = self.metrics.compute_convergence(
            layer_name,
            self.config.convergence.min_wins_per_neuron
        )
        
        layer_stats = self.metrics.layers[layer_name]
        max_wins = layer_stats.wins.max() if len(layer_stats.wins) > 0 else 0
        
        # Base message
        msg = (
            f"  [{batch_idx + 1}] samples={stats['n_samples']}, "
            f"spikes={stats['n_spikes']}, "
            f"conv={convergence*100:.1f}%, "
            f"max_wins={max_wins}"
        )
        
        # Add homeostasis info if available
        if layer_name in self.threshold_managers:
            thresh_mgr = self.threshold_managers[layer_name]
            thresh_stats = thresh_mgr.get_stats()
            msg += f", θ_mean={thresh_stats['mean_threshold']:.2f}"
            if thresh_stats['n_dead_neurons'] > 0:
                msg += f", dead={thresh_stats['n_dead_neurons']}"
        
        self.logger.info(msg)
    
    def _log_to_tensorboard(
        self,
        epoch: int,
        layer_name: str,
        convergence: float,
        stats: Dict[str, Any]
    ) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        prefix = f"{layer_name}/"
        
        self.writer.add_scalar(f"{prefix}convergence", convergence, epoch)
        self.writer.add_scalar(f"{prefix}n_spikes", stats['n_spikes'], epoch)
        self.writer.add_scalar(f"{prefix}n_samples", stats['n_samples'], epoch)
        
        # Weight histograms
        layer = getattr(self.model.encoder, layer_name)
        self.writer.add_histogram(
            f"{prefix}weights",
            layer.conv.weight.detach().cpu(),
            epoch
        )
        
        # Weight convergence metric: w*(1-w)
        weights = layer.conv.weight.detach()
        weight_convergence = (weights * (1.0 - weights)).mean()
        self.writer.add_scalar(f"{prefix}weight_convergence", weight_convergence, epoch)
        
        # Homeostasis metrics
        if layer_name in self.threshold_managers:
            thresh_mgr = self.threshold_managers[layer_name]
            thresh_stats = thresh_mgr.get_stats()
            
            self.writer.add_scalar(f"{prefix}threshold/mean", thresh_stats['mean_threshold'], epoch)
            self.writer.add_scalar(f"{prefix}threshold/min", thresh_stats['min_threshold'], epoch)
            self.writer.add_scalar(f"{prefix}threshold/max", thresh_stats['max_threshold'], epoch)
            self.writer.add_scalar(f"{prefix}n_dead_neurons", thresh_stats['n_dead_neurons'], epoch)
            self.writer.add_scalar(f"{prefix}firing_rate", thresh_stats['mean_firing_rate'], epoch)
            
            # Threshold histogram
            self.writer.add_histogram(
                f"{prefix}thresholds",
                thresh_mgr.thresholds.detach().cpu(),
                epoch
            )
    
    def _save_checkpoint(self, is_best: bool = False, emergency: bool = False) -> None:
        """Save training checkpoint."""
        prefix = "emergency_" if emergency else ""
        
        self.checkpoint_mgr.save(
            model=self.model,
            epoch=self.current_epoch,
            phase=self.current_phase,
            layer_name=self.current_layer or "unknown",
            metrics=self.metrics.get_summary(),
            config=self.config,
            is_best=is_best,
            extra_state={
                'global_step': self.global_step,
            }
        )
    
    def _resume_from_checkpoint(self, path: Path) -> None:
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_mgr.load(path, device=self.device)
        
        if checkpoint is None:
            self.logger.warning(f"Could not load checkpoint: {path}")
            return
        
        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.current_phase = TrainingPhase(checkpoint['phase'])
        self.current_layer = checkpoint.get('layer_name')
        
        if 'extra_state' in checkpoint:
            self.global_step = checkpoint['extra_state'].get('global_step', 0)
        
        self.logger.info(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"phase {self.current_phase.value}, "
            f"layer {self.current_layer}"
        )
    
    def _log_final_summary(self, summary: Dict[str, Any]) -> None:
        """Log final training summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("  Training Complete!")
        self.logger.info("=" * 70)
        self.logger.info(f"  Total time: {summary['total_time_seconds']:.1f}s")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info("")
        
        for layer_name, layer_metrics in summary['layers'].items():
            self.logger.info(
                f"  {layer_name}: {layer_metrics['convergence_ratio']*100:.1f}% converged "
                f"({layer_metrics['converged_features']}/{layer_metrics['n_features']} features)"
            )
        
        self.logger.info("=" * 70 + "\n")
    
    def _dummy_data_generator(self, n_samples: int = 100) -> Iterator[torch.Tensor]:
        """Generate dummy data for testing."""
        h = self.config.data.input_height
        w = self.config.data.input_width
        c = self.config.data.input_channels
        
        self.logger.warning("Using dummy data generator (no real data provided)")
        
        for _ in range(n_samples):
            # Random sparse spike pattern
            spikes = torch.zeros(1, c, h, w)
            n_events = np.random.randint(50, 200)
            
            ys = np.random.randint(0, h, n_events)
            xs = np.random.randint(0, w, n_events)
            cs = np.random.randint(0, c, n_events)
            
            for y, x, ch in zip(ys, xs, cs):
                spikes[0, ch, y, x] = 1.0
            
            yield spikes


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SpikeSEG STDP Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML config file'
    )
    
    parser.add_argument(
        '--paper',
        type=str,
        default=None,
        choices=['igarss2023', 'kheradpisheh2018'],
        help='Use paper configuration'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./runs',
        help='Output directory'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Experiment name'
    )
    
    # Resume
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Maximum epochs per layer'
    )
    
    parser.add_argument(
        '--n-classes',
        type=int,
        default=None,
        help='Number of output classes'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'cuda:0', 'cuda:1'],
        help='Training device'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Load or create configuration
    if args.config is not None:
        config = TrainingConfig.from_yaml(args.config)
    elif args.paper is not None:
        config = TrainingConfig.from_paper(args.paper)
    else:
        config = TrainingConfig()
    
    # Apply CLI overrides
    config.output_dir = args.output
    
    if args.name:
        config.experiment_name = args.name
    else:
        config.experiment_name = f"spikeseg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.n_classes is not None:
        config.model.n_classes = args.n_classes
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.log_level is not None:
        config.logging.log_level = args.log_level
    if args.no_tensorboard:
        config.logging.tensorboard = False
    
    try:
        # Create trainer
        trainer = STDPTrainer(config)
        
        # Run training
        resume_path = Path(args.resume) if args.resume else None
        summary = trainer.train(resume_from=resume_path)
        
        print("\n✓ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return 130
        
    except TrainingError as e:
        print(f"\n✗ Training error: {e}")
        return 1
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())