"""
Pytest configuration and fixtures for SpikeSEG tests.

This module provides reusable fixtures for testing all SpikeSEG components:
    - Device handling (CPU/CUDA)
    - Tensor generation (spikes, events, weights)
    - Neuron instances (LIF, IF)
    - Layer instances (SpikingConv2d, SpikingPool2d)
    - Model instances (Encoder, Decoder, SpikeSEG)
    - Learning components (STDP, WTA)
    - Training configuration
    - Temporary directories for checkpoints

Usage:
    Tests automatically have access to all fixtures. Use them as function arguments:
    
    >>> def test_lif_neuron(lif_neuron, random_membrane):
    ...     spikes, membrane = lif_neuron(random_membrane)
    ...     assert spikes.shape == random_membrane.shape

Fixture Naming Conventions:
    - *_config: Configuration dataclass instances
    - *_params: Parameter dictionaries
    - random_*: Randomly generated tensors
    - sample_*: Fixed sample data for reproducible tests
    - mock_*: Mock objects for isolation testing
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass

import pytest
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests requiring CUDA (deselect with '-m \"not cuda\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip CUDA tests if CUDA is not available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    if not cuda_available:
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


# =============================================================================
# DEVICE FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def cuda_available(torch_available) -> bool:
    """Check if CUDA is available."""
    if not torch_available:
        return False
    import torch
    return torch.cuda.is_available()


@pytest.fixture
def device(torch_available, cuda_available):
    """Get the best available device (CUDA if available, else CPU)."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.device("cuda" if cuda_available else "cpu")


@pytest.fixture
def cpu_device(torch_available):
    """Force CPU device for deterministic tests."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.device("cpu")


@pytest.fixture
def cuda_device(cuda_available):
    """Get CUDA device (skip if not available)."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch
    return torch.device("cuda")


# =============================================================================
# RANDOM SEED FIXTURES
# =============================================================================


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def set_seed(seed, torch_available):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if torch_available:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# =============================================================================
# TENSOR SHAPE FIXTURES
# =============================================================================


@pytest.fixture
def batch_size() -> int:
    """Default batch size."""
    return 2


@pytest.fixture
def in_channels() -> int:
    """Default input channels."""
    return 1


@pytest.fixture
def out_channels() -> int:
    """Default output channels."""
    return 4


@pytest.fixture
def height() -> int:
    """Default spatial height."""
    return 32


@pytest.fixture
def width() -> int:
    """Default spatial width."""
    return 32


@pytest.fixture
def n_timesteps() -> int:
    """Default number of timesteps."""
    return 10


@pytest.fixture
def kernel_size() -> int:
    """Default convolution kernel size."""
    return 5


# =============================================================================
# TENSOR FIXTURES
# =============================================================================


@pytest.fixture
def random_input(torch_available, set_seed, batch_size, in_channels, height, width, device):
    """Random input tensor (B, C, H, W)."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.rand(batch_size, in_channels, height, width, device=device)


@pytest.fixture
def random_membrane(torch_available, set_seed, batch_size, out_channels, height, width, device):
    """Random membrane potential tensor."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.rand(batch_size, out_channels, height, width, device=device) * 15.0


@pytest.fixture
def random_spikes(torch_available, set_seed, batch_size, out_channels, height, width, device):
    """Random binary spike tensor."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return (torch.rand(batch_size, out_channels, height, width, device=device) > 0.9).float()


@pytest.fixture
def random_weights(torch_available, set_seed, out_channels, in_channels, kernel_size, device):
    """Random weight tensor for convolution (out_ch, in_ch, kH, kW)."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.rand(out_channels, in_channels, kernel_size, kernel_size, device=device)


@pytest.fixture
def temporal_spikes(torch_available, set_seed, n_timesteps, batch_size, out_channels, height, width, device):
    """Random spike tensor with time dimension (T, B, C, H, W)."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return (torch.rand(n_timesteps, batch_size, out_channels, height, width, device=device) > 0.95).float()


@pytest.fixture
def sample_events(torch_available, batch_size, in_channels, height, width, device):
    """Sample event-like input (sparse, binary)."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    # Create sparse event pattern (like satellite trajectory)
    events = torch.zeros(batch_size, in_channels, height, width, device=device)
    # Add diagonal line of events
    for i in range(min(height, width)):
        events[:, :, i, i] = 1.0
    return events


# =============================================================================
# NEURON FIXTURES
# =============================================================================


@pytest.fixture
def lif_params() -> Dict[str, Any]:
    """Default LIF neuron parameters."""
    return {
        "threshold": 10.0,
        "leak_factor": 0.9,
        "leak_mode": "subtractive",
    }


@pytest.fixture
def if_params() -> Dict[str, Any]:
    """Default IF neuron parameters."""
    return {
        "threshold": 10.0,
    }


@pytest.fixture
def lif_neuron(torch_available, lif_params):
    """Create a LIF neuron instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.core import LIFNeuron
    return LIFNeuron(**lif_params)


@pytest.fixture
def if_neuron(torch_available, if_params):
    """Create an IF neuron instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.core import IFNeuron
    return IFNeuron(**if_params)


# =============================================================================
# LAYER FIXTURES
# =============================================================================


@pytest.fixture
def conv_layer_params(in_channels, out_channels, kernel_size) -> Dict[str, Any]:
    """Parameters for SpikingConv2d layer."""
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "threshold": 10.0,
        "leak_factor": 0.9,
        "padding": kernel_size // 2,
    }


@pytest.fixture
def spiking_conv2d(torch_available, conv_layer_params, device):
    """Create a SpikingConv2d layer."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.core import SpikingConv2d
    layer = SpikingConv2d(**conv_layer_params)
    return layer.to(device)


@pytest.fixture
def spiking_pool2d(torch_available, device):
    """Create a SpikingPool2d layer."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.core import SpikingPool2d
    layer = SpikingPool2d(kernel_size=2, stride=2)
    return layer.to(device)


# =============================================================================
# STDP FIXTURES
# =============================================================================


@pytest.fixture
def stdp_config_dict() -> Dict[str, Any]:
    """STDP configuration as dictionary."""
    return {
        "lr_plus": 0.04,
        "lr_minus": 0.03,
        "weight_min": 0.0,
        "weight_max": 1.0,
        "weight_init_mean": 0.8,
        "weight_init_std": 0.01,
    }


@pytest.fixture
def stdp_config(torch_available, stdp_config_dict):
    """Create STDPConfig instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.learning import STDPConfig
    return STDPConfig(**stdp_config_dict)


@pytest.fixture
def stdp_learner(torch_available, stdp_config):
    """Create STDPLearner instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.learning import STDPLearner
    return STDPLearner(stdp_config)


@pytest.fixture
def igarss_stdp_config(torch_available):
    """STDP config with exact IGARSS 2023 parameters."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.learning import STDPConfig
    return STDPConfig.from_paper("igarss2023")


# =============================================================================
# WTA FIXTURES
# =============================================================================


@pytest.fixture
def wta_config_dict(out_channels, height, width) -> Dict[str, Any]:
    """WTA configuration as dictionary."""
    return {
        "mode": "global",
        "enable_homeostasis": True,
        "target_rate": 0.1,
        "homeostasis_lr": 0.001,
    }


@pytest.fixture
def wta_config(torch_available, wta_config_dict):
    """Create WTAConfig instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.learning import WTAConfig
    return WTAConfig(**wta_config_dict)


@pytest.fixture
def wta_inhibition(torch_available, wta_config, out_channels, height, width, device):
    """Create WTAInhibition instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.learning import WTAInhibition
    return WTAInhibition(
        config=wta_config,
        n_channels=out_channels,
        spatial_shape=(height, width),
        device=device
    )


# =============================================================================
# ENCODER FIXTURES
# =============================================================================


@pytest.fixture
def layer_config_params() -> Dict[str, Any]:
    """Default layer configuration parameters."""
    return {
        "out_channels": 4,
        "kernel_size": 5,
        "threshold": 10.0,
        "leak": 9.0,
    }


@pytest.fixture
def encoder_config(torch_available, in_channels):
    """Create EncoderConfig with small architecture for testing."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import EncoderConfig, LayerConfig
    return EncoderConfig(
        input_channels=in_channels,
        conv1=LayerConfig(out_channels=2, kernel_size=5, threshold=10.0, leak=9.0),
        conv2=LayerConfig(out_channels=4, kernel_size=5, threshold=10.0, leak=1.0),
        conv3=LayerConfig(out_channels=1, kernel_size=5, threshold=10.0, leak=0.0),
        pool_size=2,
    )


@pytest.fixture
def encoder(torch_available, encoder_config, device):
    """Create SpikeSEGEncoder instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import SpikeSEGEncoder
    return SpikeSEGEncoder(encoder_config).to(device)


@pytest.fixture
def igarss_encoder_config(torch_available):
    """Encoder config with IGARSS 2023 paper parameters."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import EncoderConfig
    return EncoderConfig.from_paper("igarss2023")


# =============================================================================
# DECODER FIXTURES
# =============================================================================


@pytest.fixture
def decoder_config(torch_available):
    """Create DecoderConfig with defaults."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import DecoderConfig
    return DecoderConfig()


@pytest.fixture
def decoder(torch_available, encoder, device):
    """Create SpikeSEGDecoder from encoder."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import SpikeSEGDecoder
    return SpikeSEGDecoder.from_encoder(encoder).to(device)


# =============================================================================
# FULL MODEL FIXTURES
# =============================================================================


@pytest.fixture
def spikeseg_model(torch_available, encoder_config, device):
    """Create complete SpikeSEG model."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import SpikeSEG
    return SpikeSEG(encoder_config).to(device)


@pytest.fixture
def small_spikeseg(torch_available, device):
    """Create minimal SpikeSEG for fast testing."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from spikeseg.models import SpikeSEG, EncoderConfig, LayerConfig
    
    config = EncoderConfig(
        input_channels=1,
        conv1=LayerConfig(out_channels=2, kernel_size=3, threshold=10.0, leak=9.0),
        conv2=LayerConfig(out_channels=4, kernel_size=3, threshold=10.0, leak=1.0),
        conv3=LayerConfig(out_channels=1, kernel_size=3, threshold=10.0, leak=0.0),
        pool_size=2,
    )
    return SpikeSEG(config).to(device)


# =============================================================================
# TRAINING FIXTURES
# =============================================================================


@pytest.fixture
def training_config_dict() -> Dict[str, Any]:
    """Training configuration as dictionary."""
    return {
        "experiment_name": "test_experiment",
        "output_dir": "./test_runs",
        "seed": 42,
        "device": "cpu",
        "max_epochs": 2,
        "max_samples_per_epoch": 10,
        "train_conv1": False,
        "train_conv2": True,
        "train_conv3": True,
    }


@pytest.fixture
def training_config(torch_available, training_config_dict, tmp_path):
    """Create TrainingConfig instance."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    
    # TrainingConfig is in scripts/train.py, not exposed as module yet
    # Return the dict for now, tests can use it directly
    training_config_dict["output_dir"] = str(tmp_path / "runs")
    return training_config_dict


@pytest.fixture
def quick_training_config(torch_available, tmp_path):
    """Minimal training config for fast tests."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    
    # TrainingConfig is in scripts/train.py, not exposed as module yet
    # Return dict for now
    return {
        "experiment_name": "quick_test",
        "output_dir": str(tmp_path / "runs"),
        "seed": 42,
        "device": "cpu",
        "max_epochs": 1,
        "max_samples_per_epoch": 5,
        "train_conv1": False,
        "train_conv2": True,
        "train_conv3": False,
    }


# =============================================================================
# DATALOADER FIXTURES
# =============================================================================


@pytest.fixture
def dummy_dataset(torch_available, set_seed, height, width):
    """Create a dummy dataset for testing."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    from torch.utils.data import TensorDataset
    
    n_samples = 20
    data = (torch.rand(n_samples, 1, height, width) > 0.9).float()
    labels = torch.zeros(n_samples, 1, height, width)
    
    return TensorDataset(data, labels)


@pytest.fixture
def dummy_dataloader(torch_available, dummy_dataset):
    """Create a DataLoader from dummy dataset."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    from torch.utils.data import DataLoader
    
    return DataLoader(dummy_dataset, batch_size=1, shuffle=False)


# =============================================================================
# FILE/PATH FIXTURES
# =============================================================================


@pytest.fixture
def tmp_checkpoint_dir(tmp_path) -> Path:
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def tmp_log_dir(tmp_path) -> Path:
    """Temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def sample_config_path(tmp_path) -> Path:
    """Create a sample config YAML file."""
    config_content = """
experiment_name: "test"
output_dir: "./test_runs"
seed: 42
device: "cpu"
max_epochs: 1
max_samples_per_epoch: 5

stdp:
  lr_plus: 0.04
  lr_minus: 0.03

model:
  n_classes: 1
  conv1_channels: 2
  conv2_channels: 4
  kernel_sizes: [3, 3, 3]

homeostasis:
  enabled: true
  theta_rest: 10.0
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def project_root_path() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root_path) -> Path:
    """Get configs directory."""
    return project_root_path / "configs"


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_encoder_output(torch_available, batch_size, out_channels, height, width, device):
    """Create mock EncoderOutput for decoder testing."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    from spikeseg.models import EncoderOutput, PoolingIndices
    
    # Create mock pooling indices
    pool1_size = (height // 2, width // 2)
    pool2_size = (height // 4, width // 4)
    
    pool1_indices = torch.zeros(batch_size, out_channels, *pool1_size, dtype=torch.long, device=device)
    pool2_indices = torch.zeros(batch_size, out_channels, *pool2_size, dtype=torch.long, device=device)
    
    pooling_indices = PoolingIndices(
        pool1_indices=pool1_indices,
        pool2_indices=pool2_indices,
        pool1_output_size=(batch_size, out_channels, *pool1_size),
        pool2_output_size=(batch_size, out_channels, *pool2_size),
    )
    
    # Create mock spikes
    classification_spikes = torch.zeros(batch_size, 1, *pool2_size, device=device)
    
    return EncoderOutput(
        classification_spikes=classification_spikes,
        pooling_indices=pooling_indices,
        layer_spikes={},
        layer_membranes={},
    )


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def assert_tensor_equal(torch_available):
    """Utility function to assert tensor equality."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    
    def _assert_equal(a, b, rtol=1e-5, atol=1e-8):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), \
            f"Tensors not equal:\n{a}\nvs\n{b}"
    
    return _assert_equal


@pytest.fixture
def assert_spikes_valid(torch_available):
    """Utility function to validate spike tensor."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    
    def _assert_valid(spikes):
        # Check binary
        assert torch.all((spikes == 0) | (spikes == 1)), \
            "Spikes must be binary (0 or 1)"
        # Check dtype
        assert spikes.dtype in (torch.float32, torch.float64), \
            f"Spikes should be float, got {spikes.dtype}"
    
    return _assert_valid


@pytest.fixture
def count_parameters(torch_available):
    """Utility to count model parameters."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    
    def _count(model, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    return _count


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================


@pytest.fixture
def timer():
    """Simple timer context manager for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
    
    return Timer


@pytest.fixture
def memory_tracker(torch_available, cuda_available):
    """Track CUDA memory usage."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch
    
    class MemoryTracker:
        def __init__(self):
            self.start_mem = 0
            self.end_mem = 0
        
        def __enter__(self):
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
            return self
        
        def __exit__(self, *args):
            self.end_mem = torch.cuda.memory_allocated()
            self.peak_mem = torch.cuda.max_memory_allocated()
        
        @property
        def used(self):
            return self.end_mem - self.start_mem
        
        @property
        def peak(self):
            return self.peak_mem
    
    return MemoryTracker