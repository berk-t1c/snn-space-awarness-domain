"""
Utility functions for SpikeSEG.

This module contains:
- Logging setup and formatting
- TensorBoard integration
- Metrics tracking and export
- Progress tracking with ETA
- Visualization tools (future)

Example:
    >>> from spikeseg.utils import setup_logger, TensorBoardLogger
    >>> 
    >>> # Setup logging
    >>> logger = setup_logger("spikeseg", log_dir="./logs")
    >>> logger.info("Training started")
    >>> 
    >>> # TensorBoard
    >>> tb = TensorBoardLogger("./runs/exp1")
    >>> tb.log_scalar("loss", 0.5, step=100)
"""

from .logging import (
    # Colors
    ANSIColors,
    
    # Formatters
    ColoredFormatter,
    DetailedFormatter,
    
    # Logger setup
    setup_logger,
    get_logger,
    log_level,
    
    # TensorBoard
    TensorBoardLogger,
    
    # Metrics
    MetricValue,
    MetricsLogger,
    
    # Progress
    ProgressTracker,
    
    # Utilities
    log_dict,
    log_separator,
    log_header,
)


__all__ = [
    # Colors
    "ANSIColors",
    
    # Formatters
    "ColoredFormatter",
    "DetailedFormatter",
    
    # Logger setup
    "setup_logger",
    "get_logger",
    "log_level",
    
    # TensorBoard
    "TensorBoardLogger",
    
    # Metrics
    "MetricValue",
    "MetricsLogger",
    
    # Progress
    "ProgressTracker",
    
    # Utilities
    "log_dict",
    "log_separator",
    "log_header",
]