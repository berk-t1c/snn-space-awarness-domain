"""
Configuration management for SpikeSEG.

This module provides utilities for loading, validating, and managing
configuration from YAML files.

Example:
    >>> from spikeseg.config import load_config, get_default_config_path
    >>> 
    >>> # Load from default location
    >>> config = load_config()
    >>> 
    >>> # Load from specific file
    >>> config = load_config("configs/experiments/satellite.yaml")
    >>> 
    >>> # Load with overrides
    >>> config = load_config(overrides={"max_epochs": 20, "seed": 123})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigFileNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


# =============================================================================
# PATH UTILITIES
# =============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root (parent of 'spikeseg' package).
    """
    return Path(__file__).resolve().parent.parent


def get_default_config_path() -> Path:
    """
    Get the default configuration file path.
    
    Returns:
        Path to configs/config.yaml
    """
    return get_project_root() / "configs" / "config.yaml"


def get_config_dir() -> Path:
    """
    Get the configuration directory.
    
    Returns:
        Path to configs/ directory.
    """
    return get_project_root() / "configs"


# =============================================================================
# CONFIG LOADING
# =============================================================================


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        path: Path to YAML file.
    
    Returns:
        Dictionary of configuration values.
    
    Raises:
        ConfigFileNotFoundError: If file doesn't exist.
        ConfigError: If YAML parsing fails.
    """
    path = Path(path)
    
    if not path.exists():
        raise ConfigFileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {path}: {e}") from e


def save_yaml(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration.
        override: Override configuration.
    
    Returns:
        Merged configuration.
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        path: Path to config file. If None, uses default config.
        overrides: Dictionary of values to override.
    
    Returns:
        Complete configuration dictionary.
    
    Example:
        >>> config = load_config()  # Load default
        >>> config = load_config("configs/experiments/satellite.yaml")
        >>> config = load_config(overrides={"max_epochs": 5})
    """
    if path is None:
        path = get_default_config_path()
    
    config = load_yaml(path)
    
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


# =============================================================================
# CONFIG DATACLASSES (lightweight versions for quick access)
# =============================================================================


@dataclass
class STDPParams:
    """STDP learning parameters (IGARSS 2023 / EBSSA)."""
    lr_plus: float = 0.04    # α⁺ potentiation rate (IGARSS 2023: 10x higher)
    lr_minus: float = 0.03   # α⁻ depression rate (IGARSS 2023: 10x higher)
    weight_min: float = 0.0
    weight_max: float = 1.0
    weight_init_mean: float = 0.8
    weight_init_std: float = 0.01  # IGARSS 2023: 0.01
    use_soft_bounds: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "STDPParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelParams:
    """Model architecture parameters (IGARSS 2023 / EBSSA)."""
    n_classes: int = 2       # Binary segmentation for EBSSA
    conv1_channels: int = 4
    conv2_channels: int = 36
    kernel_sizes: tuple = (5, 5, 7)
    # Pool1/Pool2: 2×2 kernel, stride 2 (IGARSS 2023 standard pooling)
    pool1_kernel: int = 2
    pool1_stride: int = 2
    pool2_kernel: int = 2
    pool2_stride: int = 2
    thresholds: tuple = (10.0, 10.0, 10.0)  # Adaptive via PENT
    leaks: tuple = (9.0, 1.0, 0.0)  # 90%, 10%, 0% of threshold
    use_dog_filters: bool = True
    
    def __post_init__(self):
        # Convert lists to tuples (from YAML)
        if isinstance(self.kernel_sizes, list):
            self.kernel_sizes = tuple(self.kernel_sizes)
        if isinstance(self.thresholds, list):
            self.thresholds = tuple(self.thresholds)
        if isinstance(self.leaks, list):
            self.leaks = tuple(self.leaks)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataParams:
    """Data loading parameters."""
    dataset: str = "ebssa"
    data_root: str = "./data"
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    timestep_ms: float = 100.0
    n_timesteps: int = 20
    input_height: int = 128
    input_width: int = 128
    input_channels: int = 1
    normalize: bool = True
    shuffle_train: bool = True
    shuffle_val: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_stdp_params(config: Optional[Dict[str, Any]] = None) -> STDPParams:
    """
    Get STDP parameters from config.
    
    Args:
        config: Config dict. If None, loads default.
    
    Returns:
        STDPParams instance.
    """
    if config is None:
        config = load_config()
    return STDPParams.from_dict(config.get("stdp", {}))


def get_model_params(config: Optional[Dict[str, Any]] = None) -> ModelParams:
    """
    Get model parameters from config.
    
    Args:
        config: Config dict. If None, loads default.
    
    Returns:
        ModelParams instance.
    """
    if config is None:
        config = load_config()
    return ModelParams.from_dict(config.get("model", {}))


def get_data_params(config: Optional[Dict[str, Any]] = None) -> DataParams:
    """
    Get data parameters from config.
    
    Args:
        config: Config dict. If None, loads default.
    
    Returns:
        DataParams instance.
    """
    if config is None:
        config = load_config()
    return DataParams.from_dict(config.get("data", {}))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    # Path utilities
    "get_project_root",
    "get_default_config_path",
    "get_config_dir",
    # Loading/saving
    "load_yaml",
    "save_yaml",
    "merge_configs",
    "load_config",
    # Dataclasses
    "STDPParams",
    "ModelParams",
    "DataParams",
    # Convenience
    "get_stdp_params",
    "get_model_params",
    "get_data_params",
]