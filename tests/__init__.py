"""
SpikeSEG Training Module.

This module provides comprehensive STDP-based training for SpikeSEG
spiking neural networks.

Components:
    - STDPTrainer: Main trainer class for layer-wise STDP training
    - TrainingConfig: Complete training configuration
    - MetricsTracker: Comprehensive metrics tracking
    - CheckpointManager: Checkpoint save/load management

Example:
    >>> from spikeseg.training import STDPTrainer, TrainingConfig
    >>> 
    >>> # Create from paper configuration
    >>> config = TrainingConfig.from_paper("igarss2023")
    >>> trainer = STDPTrainer(config)
    >>> summary = trainer.train()
    >>> 
    >>> # Or from YAML config
    >>> config = TrainingConfig.from_yaml("configs/ebssa.yaml")
    >>> trainer = STDPTrainer(config, train_loader=my_loader)
    >>> summary = trainer.train()
"""

from .train import (
    # Exceptions
    TrainingError,
    ConfigurationError,
    CheckpointError,
    ConvergenceError,
    DataLoadingError,
    
    # Enums
    TrainingPhase,
    WTAMode,
    
    # Configuration classes
    STDPParams,
    WTAParams,
    ConvergenceParams,
    DataParams,
    ModelParams,
    CheckpointParams,
    LoggingParams,
    TrainingConfig,
    
    # Metrics
    LayerStats,
    MetricsTracker,
    
    # Core classes
    CheckpointManager,
    GracefulShutdown,
    STDPTrainer,
    
    # Functions
    setup_logging,
)

__all__ = [
    # Exceptions
    "TrainingError",
    "ConfigurationError", 
    "CheckpointError",
    "ConvergenceError",
    "DataLoadingError",
    
    # Enums
    "TrainingPhase",
    "WTAMode",
    
    # Configuration
    "STDPParams",
    "WTAParams",
    "ConvergenceParams",
    "DataParams",
    "ModelParams",
    "CheckpointParams",
    "LoggingParams",
    "TrainingConfig",
    
    # Metrics
    "LayerStats",
    "MetricsTracker",
    
    # Core
    "CheckpointManager",
    "GracefulShutdown",
    "STDPTrainer",
    
    # Functions
    "setup_logging",
]