"""
Logging utilities for SpikeSEG.

This module provides:
    - ColoredFormatter: ANSI-colored console output
    - setup_logger: Configure logging with file and console handlers
    - TensorBoardLogger: TensorBoard integration wrapper
    - MetricsLogger: Structured metrics logging
    - ProgressTracker: Training progress tracking
    - LogContext: Context manager for temporary log settings

Features:
    - Colored console output by log level
    - File logging with detailed format
    - TensorBoard scalar, histogram, and image logging
    - JSON/CSV metrics export
    - Progress bars and ETA tracking

Example:
    >>> from spikeseg.utils.logging import setup_logger, TensorBoardLogger
    >>> 
    >>> # Setup basic logging
    >>> logger = setup_logger("spikeseg", log_dir="./logs", level="INFO")
    >>> logger.info("Training started")
    >>> 
    >>> # TensorBoard logging
    >>> tb = TensorBoardLogger("./runs/exp1")
    >>> tb.log_scalar("loss", 0.5, step=100)
    >>> tb.log_histogram("weights", weight_tensor, step=100)

Author: SpikeSEG Team
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import numpy as np


# =============================================================================
# TYPE HINTS
# =============================================================================

# Avoid importing torch at module level for flexibility
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
SIMPLE_LOG_FORMAT = '%(levelname)s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT = '%H:%M:%S'


# =============================================================================
# ANSI COLOR CODES
# =============================================================================

class ANSIColors:
    """ANSI escape codes for terminal colors."""
    
    # Reset
    RESET = '\033[0m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    @classmethod
    def colorize(cls, text: str, color: str, bold: bool = False) -> str:
        """Apply color to text."""
        prefix = cls.BOLD if bold else ''
        return f"{prefix}{color}{text}{cls.RESET}"
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports color."""
        # Check for NO_COLOR environment variable
        if os.environ.get('NO_COLOR'):
            return False
        
        # Check if stdout is a tty
        if not hasattr(sys.stdout, 'isatty'):
            return False
        
        if not sys.stdout.isatty():
            return False
        
        # Check TERM environment variable
        term = os.environ.get('TERM', '')
        if term in ('dumb', ''):
            return False
        
        return True


# =============================================================================
# COLORED FORMATTER
# =============================================================================


class ColoredFormatter(logging.Formatter):
    """
    Logging formatter with ANSI color codes for different log levels.
    
    Colors:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Magenta (bold)
    
    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ColoredFormatter())
        >>> logger.addHandler(handler)
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: ANSIColors.CYAN,
        logging.INFO: ANSIColors.GREEN,
        logging.WARNING: ANSIColors.YELLOW,
        logging.ERROR: ANSIColors.RED,
        logging.CRITICAL: ANSIColors.MAGENTA,
    }
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: Optional[bool] = None
    ):
        """
        Initialize colored formatter.
        
        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_colors: Force color on/off. None = auto-detect.
        """
        if fmt is None:
            fmt = '%(asctime)s | %(levelname)s | %(message)s'
        if datefmt is None:
            datefmt = TIME_FORMAT
        
        super().__init__(fmt, datefmt)
        
        if use_colors is None:
            self.use_colors = ANSIColors.supports_color()
        else:
            self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original levelname
        original_levelname = record.levelname
        
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, '')
            if record.levelno >= logging.CRITICAL:
                record.levelname = f"{ANSIColors.BOLD}{color}{record.levelname:8s}{ANSIColors.RESET}"
            else:
                record.levelname = f"{color}{record.levelname:8s}{ANSIColors.RESET}"
        else:
            record.levelname = f"{record.levelname:8s}"
        
        # Format message
        result = super().format(record)
        
        # Restore original
        record.levelname = original_levelname
        
        return result


class DetailedFormatter(logging.Formatter):
    """
    Detailed formatter for file logging with function name and line number.
    
    Format: timestamp | level | logger | function:line | message
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None
    ):
        if fmt is None:
            fmt = DETAILED_LOG_FORMAT
        if datefmt is None:
            datefmt = DATE_FORMAT
        super().__init__(fmt, datefmt)


# =============================================================================
# LOGGER SETUP
# =============================================================================


def setup_logger(
    name: str = "spikeseg",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    level: Union[str, int] = "INFO",
    console: bool = True,
    file: bool = True,
    use_colors: Optional[bool] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Logger name. Default "spikeseg".
        log_dir: Directory for log files. Default None (current directory).
        log_file: Log file name. Default "{name}_{timestamp}.log".
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Enable console output.
        file: Enable file output.
        use_colors: Force console colors on/off. None = auto-detect.
        propagate: Propagate to parent loggers.
    
    Returns:
        Configured logger.
    
    Example:
        >>> logger = setup_logger("myapp", log_dir="./logs", level="DEBUG")
        >>> logger.info("Application started")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Convert level string to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = propagate
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(use_colors=use_colors)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(DetailedFormatter())
        logger.addHandler(file_handler)
        
        logger.debug(f"Log file: {log_path}")
    
    return logger


def get_logger(name: str = "spikeseg") -> logging.Logger:
    """
    Get an existing logger by name.
    
    If the logger doesn't exist, returns a logger with default settings.
    
    Args:
        name: Logger name.
    
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If no handlers, add a basic console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger


@contextmanager
def log_level(logger: logging.Logger, level: Union[str, int]):
    """
    Context manager to temporarily change log level.
    
    Args:
        logger: Logger to modify.
        level: Temporary log level.
    
    Example:
        >>> with log_level(logger, "DEBUG"):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This won't be shown")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


# =============================================================================
# TENSORBOARD LOGGER
# =============================================================================


class TensorBoardLogger:
    """
    TensorBoard logging wrapper with convenient methods.
    
    Wraps torch.utils.tensorboard.SummaryWriter with:
        - Automatic step tracking
        - Image normalization
        - Histogram binning
        - Hyperparameter logging
    
    Example:
        >>> tb = TensorBoardLogger("./runs/exp1")
        >>> tb.log_scalar("train/loss", 0.5, step=100)
        >>> tb.log_histogram("model/weights", weights, step=100)
        >>> tb.log_image("samples/input", image, step=100)
        >>> tb.close()
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
            comment: Comment to append to log directory name.
            flush_secs: Flush interval in seconds.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = None
        self._step = 0
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                comment=comment,
                flush_secs=flush_secs
            )
            self._available = True
        except ImportError:
            warnings.warn(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            self._available = False
    
    @property
    def available(self) -> bool:
        """Check if TensorBoard is available."""
        return self._available
    
    @property
    def step(self) -> int:
        """Get current global step."""
        return self._step
    
    @step.setter
    def step(self, value: int) -> None:
        """Set current global step."""
        self._step = value
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Data identifier (e.g., "train/loss").
            value: Scalar value to log.
            step: Global step. Uses internal counter if None.
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log multiple scalars under same main tag.
        
        Args:
            main_tag: Group identifier.
            tag_scalar_dict: Dict of {tag: value}.
            step: Global step.
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, "torch.Tensor"],
        step: Optional[int] = None,
        bins: str = 'tensorflow'
    ) -> None:
        """
        Log a histogram of values.
        
        Args:
            tag: Data identifier.
            values: Values to create histogram from.
            step: Global step.
            bins: Binning method ('tensorflow', 'auto', 'fd', etc.).
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        # Convert torch tensor to numpy
        if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_image(
        self,
        tag: str,
        img_tensor: Union[np.ndarray, "torch.Tensor"],
        step: Optional[int] = None,
        dataformats: str = 'CHW'
    ) -> None:
        """
        Log an image.
        
        Args:
            tag: Data identifier.
            img_tensor: Image tensor. Shape depends on dataformats.
            step: Global step.
            dataformats: Format of image tensor ('CHW', 'HWC', 'HW', etc.).
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        # Convert torch tensor
        if TORCH_AVAILABLE and isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
        
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)
    
    def log_images(
        self,
        tag: str,
        img_tensor: Union[np.ndarray, "torch.Tensor"],
        step: Optional[int] = None,
        dataformats: str = 'NCHW'
    ) -> None:
        """
        Log multiple images as a grid.
        
        Args:
            tag: Data identifier.
            img_tensor: Batch of images.
            step: Global step.
            dataformats: Format ('NCHW', 'NHWC').
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        if TORCH_AVAILABLE and isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
        
        self.writer.add_images(tag, img_tensor, step, dataformats=dataformats)
    
    def log_figure(
        self,
        tag: str,
        figure,
        step: Optional[int] = None,
        close: bool = True
    ) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            tag: Data identifier.
            figure: matplotlib figure object.
            step: Global step.
            close: Close figure after logging.
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        self.writer.add_figure(tag, figure, step, close=close)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None
    ) -> None:
        """
        Log text.
        
        Args:
            tag: Data identifier.
            text: Text string.
            step: Global step.
        """
        if not self._available:
            return
        
        if step is None:
            step = self._step
        
        self.writer.add_text(tag, text, step)
    
    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float]
    ) -> None:
        """
        Log hyperparameters and associated metrics.
        
        Args:
            hparam_dict: Dict of hyperparameters.
            metric_dict: Dict of metrics.
        """
        if not self._available:
            return
        
        # Filter to serializable types
        hparams = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool)):
                hparams[k] = v
            else:
                hparams[k] = str(v)
        
        self.writer.add_hparams(hparams, metric_dict)
    
    def log_graph(
        self,
        model: "torch.nn.Module",
        input_to_model: Optional["torch.Tensor"] = None
    ) -> None:
        """
        Log model graph.
        
        Args:
            model: PyTorch model.
            input_to_model: Example input tensor.
        """
        if not self._available or not TORCH_AVAILABLE:
            return
        
        try:
            self.writer.add_graph(model, input_to_model)
        except Exception as e:
            warnings.warn(f"Could not log model graph: {e}")
    
    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._available:
            self.writer.flush()
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self._available:
            self.writer.close()
    
    def __enter__(self) -> "TensorBoardLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        return f"TensorBoardLogger(log_dir={self.log_dir}, available={self._available}, step={self._step})"


# =============================================================================
# METRICS LOGGER
# =============================================================================


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsLogger:
    """
    Structured metrics logging with history tracking.
    
    Features:
        - Track metric history
        - Compute running statistics
        - Export to JSON/CSV
        - Integration with TensorBoard
    
    Example:
        >>> metrics = MetricsLogger()
        >>> metrics.log("loss", 0.5, step=100)
        >>> metrics.log("accuracy", 0.95, step=100)
        >>> print(metrics.get_latest("loss"))
        >>> metrics.save_json("metrics.json")
    """
    
    def __init__(
        self,
        tensorboard: Optional[TensorBoardLogger] = None,
        history_size: int = 10000
    ):
        """
        Initialize metrics logger.
        
        Args:
            tensorboard: Optional TensorBoard logger for forwarding.
            history_size: Maximum history entries per metric.
        """
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._tensorboard = tensorboard
        self._history_size = history_size
        self._step = 0
    
    @property
    def step(self) -> int:
        """Get current global step."""
        return self._step
    
    @step.setter
    def step(self, value: int) -> None:
        """Set current global step."""
        self._step = value
    
    def log(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Global step. Uses internal counter if None.
            tags: Optional metadata tags.
        """
        if step is None:
            step = self._step
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        entry = MetricValue(
            value=value,
            step=step,
            tags=tags or {}
        )
        self._metrics[name].append(entry)
        
        # Trim history if needed
        if len(self._metrics[name]) > self._history_size:
            self._metrics[name] = self._metrics[name][-self._history_size:]
        
        # Forward to TensorBoard
        if self._tensorboard is not None:
            self._tensorboard.log_scalar(name, value, step)
    
    def log_dict(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dict of {name: value}.
            step: Global step.
            prefix: Prefix to add to all metric names.
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.log(full_name, value, step)
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if name not in self._metrics or len(self._metrics[name]) == 0:
            return None
        return self._metrics[name][-1].value
    
    def get_history(
        self,
        name: str,
        last_n: Optional[int] = None
    ) -> List[MetricValue]:
        """
        Get metric history.
        
        Args:
            name: Metric name.
            last_n: Return only last N entries.
        
        Returns:
            List of MetricValue entries.
        """
        if name not in self._metrics:
            return []
        
        history = self._metrics[name]
        if last_n is not None:
            history = history[-last_n:]
        return history
    
    def get_values(
        self,
        name: str,
        last_n: Optional[int] = None
    ) -> List[float]:
        """Get just the values from metric history."""
        return [m.value for m in self.get_history(name, last_n)]
    
    def get_mean(
        self,
        name: str,
        last_n: Optional[int] = None
    ) -> Optional[float]:
        """Get mean of recent values."""
        values = self.get_values(name, last_n)
        if not values:
            return None
        return sum(values) / len(values)
    
    def get_std(
        self,
        name: str,
        last_n: Optional[int] = None
    ) -> Optional[float]:
        """Get standard deviation of recent values."""
        values = self.get_values(name, last_n)
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_min(self, name: str) -> Optional[float]:
        """Get minimum value for a metric."""
        values = self.get_values(name)
        return min(values) if values else None
    
    def get_max(self, name: str) -> Optional[float]:
        """Get maximum value for a metric."""
        values = self.get_values(name)
        return max(values) if values else None
    
    def get_all_names(self) -> List[str]:
        """Get all metric names."""
        return list(self._metrics.keys())
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert all metrics to dictionary."""
        return {
            name: [asdict(m) for m in entries]
            for name, entries in self._metrics.items()
        }
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_csv(self, path: Union[str, Path]) -> None:
        """Save metrics to CSV file."""
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value', 'step', 'timestamp'])
            for name, entries in self._metrics.items():
                for m in entries:
                    writer.writerow([name, m.value, m.step, m.timestamp])
    
    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear metric history.
        
        Args:
            name: Metric to clear. If None, clears all.
        """
        if name is None:
            self._metrics.clear()
        elif name in self._metrics:
            del self._metrics[name]
    
    def __repr__(self) -> str:
        return f"MetricsLogger(metrics={list(self._metrics.keys())}, step={self._step})"


# =============================================================================
# PROGRESS TRACKER
# =============================================================================


class ProgressTracker:
    """
    Training progress tracker with ETA estimation.
    
    Features:
        - Track iterations and epochs
        - Estimate remaining time
        - Format progress strings
        - Calculate throughput
    
    Example:
        >>> progress = ProgressTracker(total_epochs=10, samples_per_epoch=1000)
        >>> for epoch in range(10):
        ...     for batch in dataloader:
        ...         progress.update(batch_size=32)
        ...         print(progress.format())
    """
    
    def __init__(
        self,
        total_epochs: int,
        samples_per_epoch: int,
        start_epoch: int = 0
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_epochs: Total number of epochs.
            samples_per_epoch: Samples per epoch.
            start_epoch: Starting epoch (for resume).
        """
        self.total_epochs = total_epochs
        self.samples_per_epoch = samples_per_epoch
        self.total_samples = total_epochs * samples_per_epoch
        
        self.current_epoch = start_epoch
        self.current_sample = 0
        self.global_sample = start_epoch * samples_per_epoch
        
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        
        self._recent_times: List[float] = []
        self._recent_counts: List[int] = []
    
    def update(self, batch_size: int = 1) -> None:
        """
        Update progress after processing a batch.
        
        Args:
            batch_size: Number of samples in batch.
        """
        self.current_sample += batch_size
        self.global_sample += batch_size
        
        self._recent_times.append(time.time())
        self._recent_counts.append(batch_size)
        
        # Keep only recent entries
        if len(self._recent_times) > 100:
            self._recent_times = self._recent_times[-50:]
            self._recent_counts = self._recent_counts[-50:]
    
    def new_epoch(self, epoch: Optional[int] = None) -> None:
        """
        Start a new epoch.
        
        Args:
            epoch: Epoch number. If None, increments current.
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        self.current_sample = 0
        self.epoch_start_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def epoch_elapsed(self) -> float:
        """Elapsed time in current epoch."""
        return time.time() - self.epoch_start_time
    
    @property
    def progress_ratio(self) -> float:
        """Overall progress as ratio [0, 1]."""
        if self.total_samples == 0:
            return 0.0
        return self.global_sample / self.total_samples
    
    @property
    def epoch_progress_ratio(self) -> float:
        """Current epoch progress as ratio [0, 1]."""
        if self.samples_per_epoch == 0:
            return 0.0
        return min(1.0, self.current_sample / self.samples_per_epoch)
    
    @property
    def throughput(self) -> float:
        """Samples per second (recent average)."""
        if len(self._recent_times) < 2:
            return 0.0
        
        time_delta = self._recent_times[-1] - self._recent_times[0]
        if time_delta <= 0:
            return 0.0
        
        sample_count = sum(self._recent_counts)
        return sample_count / time_delta
    
    @property
    def eta(self) -> float:
        """Estimated time remaining in seconds."""
        throughput = self.throughput
        if throughput <= 0:
            return float('inf')
        
        remaining = self.total_samples - self.global_sample
        return remaining / throughput
    
    @property
    def epoch_eta(self) -> float:
        """Estimated time remaining in current epoch."""
        throughput = self.throughput
        if throughput <= 0:
            return float('inf')
        
        remaining = self.samples_per_epoch - self.current_sample
        return max(0, remaining / throughput)
    
    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds == float('inf'):
            return "??:??:??"
        
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def format(self, width: int = 50) -> str:
        """
        Format progress as string with bar.
        
        Args:
            width: Width of progress bar.
        
        Returns:
            Formatted progress string.
        """
        # Progress bar
        filled = int(width * self.epoch_progress_ratio)
        bar = '█' * filled + '░' * (width - filled)
        
        # Stats
        pct = self.epoch_progress_ratio * 100
        throughput = self.throughput
        eta = self.format_time(self.epoch_eta)
        
        return (
            f"Epoch {self.current_epoch + 1}/{self.total_epochs} "
            f"[{bar}] {pct:5.1f}% "
            f"| {throughput:.1f} samples/s "
            f"| ETA: {eta}"
        )
    
    def format_summary(self) -> str:
        """Format overall progress summary."""
        pct = self.progress_ratio * 100
        elapsed = self.format_time(self.elapsed)
        eta = self.format_time(self.eta)
        
        return (
            f"Overall: {pct:.1f}% | "
            f"Elapsed: {elapsed} | "
            f"ETA: {eta} | "
            f"{self.throughput:.1f} samples/s"
        )
    
    def __repr__(self) -> str:
        return (
            f"ProgressTracker(epoch={self.current_epoch}/{self.total_epochs}, "
            f"sample={self.current_sample}/{self.samples_per_epoch})"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def log_dict(
    logger: logging.Logger,
    data: Dict[str, Any],
    prefix: str = "",
    level: int = logging.INFO
) -> None:
    """
    Log dictionary items as separate log entries.
    
    Args:
        logger: Logger to use.
        data: Dictionary to log.
        prefix: Prefix for each line.
        level: Log level.
    """
    for key, value in data.items():
        if isinstance(value, float):
            logger.log(level, f"{prefix}{key}: {value:.6f}")
        else:
            logger.log(level, f"{prefix}{key}: {value}")


def log_separator(
    logger: logging.Logger,
    char: str = "=",
    width: int = 70,
    level: int = logging.INFO
) -> None:
    """Log a separator line."""
    logger.log(level, char * width)


def log_header(
    logger: logging.Logger,
    title: str,
    char: str = "=",
    width: int = 70,
    level: int = logging.INFO
) -> None:
    """Log a header with title."""
    logger.log(level, char * width)
    logger.log(level, f"  {title}")
    logger.log(level, char * width)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

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
    
    # Constants
    "DEFAULT_LOG_FORMAT",
    "DETAILED_LOG_FORMAT",
    "SIMPLE_LOG_FORMAT",
]