"""
Tests for spiking neuron models.

This module tests:
    - spike_function: Threshold comparison for spike generation
    - BaseNeuron: Abstract base class
    - IFNeuron: Integrate-and-Fire neuron
    - LIFNeuron: Leaky Integrate-and-Fire neuron (subtractive & multiplicative)
    - create_neuron: Factory function

Run with: pytest tests/test_neurons.py -v
"""

import pytest
import torch
import torch.nn as nn
import math


# =============================================================================
# SPIKE FUNCTION TESTS
# =============================================================================


class TestSpikeFunction:
    """Test the spike_function threshold operation."""
    
    def test_basic_threshold(self, torch_available, device):
        """Test basic threshold behavior."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.tensor([5.0, 10.0, 15.0], device=device)
        spikes = spike_function(membrane, threshold=10.0)
        
        expected = torch.tensor([0.0, 1.0, 1.0], device=device)
        assert torch.equal(spikes, expected)
    
    def test_exact_threshold(self, torch_available, device):
        """Test spike at exact threshold value."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.tensor([9.99, 10.0, 10.01], device=device)
        spikes = spike_function(membrane, threshold=10.0)
        
        # >= threshold, so 10.0 should spike
        expected = torch.tensor([0.0, 1.0, 1.0], device=device)
        assert torch.equal(spikes, expected)
    
    def test_all_below_threshold(self, torch_available, device):
        """Test no spikes when all below threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.rand(10, device=device) * 5.0  # All < 10
        spikes = spike_function(membrane, threshold=10.0)
        
        assert spikes.sum() == 0
    
    def test_all_above_threshold(self, torch_available, device):
        """Test all spike when all above threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.rand(10, device=device) * 5.0 + 10.0  # All >= 10
        spikes = spike_function(membrane, threshold=10.0)
        
        assert spikes.sum() == 10
    
    def test_output_dtype_float(self, torch_available, device):
        """Test output is float tensor."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.tensor([15.0], device=device)
        spikes = spike_function(membrane, threshold=10.0)
        
        assert spikes.dtype == torch.float32 or spikes.dtype == torch.float64
    
    def test_output_binary(self, torch_available, device, assert_spikes_valid):
        """Test output is binary (0 or 1)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.randn(100, device=device) * 20.0
        spikes = spike_function(membrane, threshold=10.0)
        
        assert_spikes_valid(spikes)
    
    def test_preserves_shape(self, torch_available, device):
        """Test output shape matches input shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        shapes = [(10,), (4, 5), (2, 3, 4), (1, 4, 32, 32)]
        
        for shape in shapes:
            membrane = torch.randn(shape, device=device) * 20.0
            spikes = spike_function(membrane, threshold=10.0)
            
            assert spikes.shape == membrane.shape
    
    def test_zero_threshold(self, torch_available, device):
        """Test with zero threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.tensor([-1.0, 0.0, 1.0], device=device)
        spikes = spike_function(membrane, threshold=0.0)
        
        # >= 0, so 0.0 and 1.0 should spike
        expected = torch.tensor([0.0, 1.0, 1.0], device=device)
        assert torch.equal(spikes, expected)
    
    def test_negative_membrane(self, torch_available, device):
        """Test with negative membrane potentials."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import spike_function
        
        membrane = torch.tensor([-5.0, -1.0, 0.0, 5.0], device=device)
        spikes = spike_function(membrane, threshold=1.0)
        
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        assert torch.equal(spikes, expected)


# =============================================================================
# BASE NEURON TESTS
# =============================================================================


class TestBaseNeuron:
    """Test BaseNeuron abstract class."""
    
    def test_is_nn_module(self, torch_available):
        """Test BaseNeuron inherits from nn.Module."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import BaseNeuron
        
        assert issubclass(BaseNeuron, nn.Module)
    
    def test_threshold_buffer(self, torch_available, device):
        """Test threshold is stored as buffer."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import BaseNeuron
        
        # Create concrete subclass for testing
        class ConcreteNeuron(BaseNeuron):
            def forward(self, input_current, membrane):
                return input_current, membrane
        
        neuron = ConcreteNeuron(threshold=5.0).to(device)
        
        assert hasattr(neuron, 'threshold')
        assert neuron.threshold.item() == 5.0
        assert 'threshold' in dict(neuron.named_buffers())
    
    def test_threshold_not_parameter(self, torch_available):
        """Test threshold is not a trainable parameter."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import BaseNeuron
        
        class ConcreteNeuron(BaseNeuron):
            def forward(self, input_current, membrane):
                return input_current, membrane
        
        neuron = ConcreteNeuron(threshold=5.0)
        
        # Should have no trainable parameters
        assert len(list(neuron.parameters())) == 0
    
    def test_reset_state(self, torch_available, device):
        """Test reset_state creates zero tensor."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import BaseNeuron
        
        class ConcreteNeuron(BaseNeuron):
            def forward(self, input_current, membrane):
                return input_current, membrane
        
        neuron = ConcreteNeuron(threshold=5.0)
        
        shape = (2, 4, 8, 8)
        membrane = neuron.reset_state(shape, device)
        
        assert membrane.shape == shape
        assert membrane.device == device
        assert membrane.sum() == 0.0
    
    def test_forward_not_implemented(self, torch_available):
        """Test forward raises NotImplementedError in base class."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import BaseNeuron
        
        neuron = BaseNeuron(threshold=5.0)
        
        with pytest.raises(NotImplementedError):
            neuron(torch.zeros(1), torch.zeros(1))


# =============================================================================
# IF NEURON TESTS
# =============================================================================


class TestIFNeuron:
    """Test Integrate-and-Fire neuron."""
    
    def test_creation(self, torch_available, device):
        """Test IF neuron creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(device)
        
        assert neuron.threshold.item() == 10.0
    
    def test_default_threshold(self, torch_available):
        """Test IF neuron default threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron()
        
        assert neuron.threshold.item() == 1.0
    
    def test_integration(self, torch_available, device):
        """Test membrane potential integration (accumulation)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([3.0], device=device)
        
        # First step: 0 + 3 = 3
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(3.0)
        
        # Second step: 3 + 3 = 6
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(6.0)
        
        # Third step: 6 + 3 = 9
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(9.0)
    
    def test_spike_at_threshold(self, torch_available, device):
        """Test spike generation when threshold is reached."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(device)
        
        membrane = torch.tensor([9.0], device=device)
        input_current = torch.tensor([2.0], device=device)
        
        # 9 + 2 = 11 >= 10, should spike
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 1.0
    
    def test_reset_after_spike(self, torch_available, device):
        """Test membrane resets to 0 after spiking."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(device)
        
        membrane = torch.tensor([9.0], device=device)
        input_current = torch.tensor([5.0], device=device)
        
        # 9 + 5 = 14 >= 10, should spike and reset
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 1.0
        assert membrane.item() == 0.0  # Reset to 0
    
    def test_no_spike_below_threshold(self, torch_available, device):
        """Test no spike when below threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(device)
        
        membrane = torch.tensor([5.0], device=device)
        input_current = torch.tensor([4.0], device=device)
        
        # 5 + 4 = 9 < 10, should not spike
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 0.0
        assert membrane.item() == pytest.approx(9.0)
    
    def test_batch_processing(self, torch_available, device, batch_size, out_channels, height, width):
        """Test IF neuron with batched input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=0.5).to(device)
        
        membrane = torch.zeros(batch_size, out_channels, height, width, device=device)
        input_current = torch.rand(batch_size, out_channels, height, width, device=device)
        
        spikes, new_membrane = neuron(input_current, membrane)
        
        assert spikes.shape == membrane.shape
        assert new_membrane.shape == membrane.shape
    
    def test_no_leak(self, torch_available, device):
        """Test IF neuron has no leak (pure integration)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=100.0).to(device)  # High threshold, no spikes
        
        membrane = torch.tensor([50.0], device=device)
        input_current = torch.tensor([0.0], device=device)  # No input
        
        # With no input and no leak, membrane should stay the same
        _, new_membrane = neuron(input_current, membrane)
        
        assert new_membrane.item() == pytest.approx(50.0)
    
    def test_repr(self, torch_available):
        """Test string representation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0)
        repr_str = repr(neuron)
        
        assert 'IFNeuron' in repr_str
        assert '10' in repr_str


# =============================================================================
# LIF NEURON TESTS - SUBTRACTIVE MODE
# =============================================================================


class TestLIFNeuronSubtractive:
    """Test LIF neuron with subtractive leak (IGARSS 2023 style)."""
    
    def test_creation(self, torch_available, device):
        """Test LIF neuron creation with subtractive mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(
            threshold=10.0,
            leak_factor=0.9,
            leak_leak_mode="subtractive"
        ).to(device)
        
        assert neuron.threshold.item() == 10.0
        assert neuron.leak_factor == 0.9
        assert neuron.leak_mode == "subtractive"
        assert neuron.leak.item() == pytest.approx(9.0)  # 0.9 * 10.0
    
    def test_leak_calculation(self, torch_available, device):
        """Test leak is calculated as leak_factor * threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # Test various combinations
        test_cases = [
            (10.0, 0.9, 9.0),   # IGARSS Layer 1
            (10.0, 0.1, 1.0),   # IGARSS Layer 2
            (10.0, 0.0, 0.0),   # No leak
            (20.0, 0.5, 10.0),  # Custom
        ]
        
        for threshold, leak_factor, expected_leak in test_cases:
            neuron = LIFNeuron(threshold=threshold, leak_factor=leak_factor, leak_mode="subtractive")
            assert neuron.leak.item() == pytest.approx(expected_leak)
    
    def test_subtractive_dynamics(self, torch_available, device):
        """Test V(t) = V(t-1) + I(t) - λ dynamics."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # λ = 2.0 (leak_factor=0.2, threshold=10.0)
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.2, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([5.0], device=device)
        input_current = torch.tensor([3.0], device=device)
        
        # V = 5 + 3 - 2 = 6
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(6.0)
    
    def test_membrane_clamp_non_negative(self, torch_available, device):
        """Test membrane is clamped to non-negative."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # High leak that would make membrane negative
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.9, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([1.0], device=device)
        input_current = torch.tensor([0.0], device=device)
        
        # V = 1 + 0 - 9 = -8, but clamped to 0
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == 0.0
    
    def test_igarss_layer1_dynamics(self, torch_available, device):
        """Test IGARSS 2023 Layer 1 dynamics (90% leak)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # IGARSS Layer 1: λ = 90% of threshold
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.9, leak_mode="subtractive").to(device)
        
        # With high leak, membrane decays quickly
        membrane = torch.tensor([9.0], device=device)
        input_current = torch.tensor([0.0], device=device)
        
        # V = 9 + 0 - 9 = 0
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == 0.0
    
    def test_igarss_layer2_dynamics(self, torch_available, device):
        """Test IGARSS 2023 Layer 2 dynamics (10% leak)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # IGARSS Layer 2: λ = 10% of threshold
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        # With low leak, membrane persists longer
        membrane = torch.tensor([5.0], device=device)
        input_current = torch.tensor([0.0], device=device)
        
        # V = 5 + 0 - 1 = 4
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(4.0)
    
    def test_spike_and_reset(self, torch_available, device):
        """Test spike generation and reset in subtractive mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([9.0], device=device)
        input_current = torch.tensor([5.0], device=device)
        
        # V = 9 + 5 - 1 = 13 >= 10, spike and reset
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 1.0
        assert membrane.item() == 0.0
    
    def test_repr_subtractive(self, torch_available):
        """Test string representation shows subtractive mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.9, leak_mode="subtractive")
        repr_str = repr(neuron)
        
        assert 'LIFNeuron' in repr_str
        assert 'subtractive' in repr_str
        assert 'λ=' in repr_str or 'leak' in repr_str.lower()


# =============================================================================
# LIF NEURON TESTS - MULTIPLICATIVE MODE
# =============================================================================


class TestLIFNeuronMultiplicative:
    """Test LIF neuron with multiplicative leak (snnTorch style)."""
    
    def test_creation(self, torch_available, device):
        """Test LIF neuron creation with multiplicative mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(
            threshold=1.0,
            leak_factor=0.1,
            leak_leak_mode="multiplicative"
        ).to(device)
        
        assert neuron.threshold.item() == 1.0
        assert neuron.leak_factor == 0.1
        assert neuron.leak_mode == "multiplicative"
        assert neuron.beta.item() == pytest.approx(0.9)  # 1 - 0.1
    
    def test_beta_calculation(self, torch_available, device):
        """Test β = 1 - leak_factor."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        test_cases = [
            (0.0, 1.0),   # No leak, β=1
            (0.1, 0.9),   # Typical, β=0.9
            (0.5, 0.5),   # Half decay
            (1.0, 0.0),   # Full decay
        ]
        
        for leak_factor, expected_beta in test_cases:
            neuron = LIFNeuron(threshold=1.0, leak_factor=leak_factor, leak_mode="multiplicative")
            assert neuron.beta.item() == pytest.approx(expected_beta)
    
    def test_multiplicative_dynamics(self, torch_available, device):
        """Test V(t) = β * V(t-1) + I(t) dynamics."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # β = 0.8
        neuron = LIFNeuron(threshold=100.0, leak_factor=0.2, leak_mode="multiplicative").to(device)
        
        membrane = torch.tensor([10.0], device=device)
        input_current = torch.tensor([2.0], device=device)
        
        # V = 0.8 * 10 + 2 = 10
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(10.0)
    
    def test_exponential_decay(self, torch_available, device):
        """Test membrane decays exponentially with no input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # β = 0.9
        neuron = LIFNeuron(threshold=100.0, leak_factor=0.1, leak_mode="multiplicative").to(device)
        
        membrane = torch.tensor([10.0], device=device)
        input_current = torch.tensor([0.0], device=device)
        
        # Track decay over multiple steps
        for i in range(5):
            _, membrane = neuron(input_current, membrane)
        
        # V = 10 * (0.9)^5 ≈ 5.9
        expected = 10.0 * (0.9 ** 5)
        assert membrane.item() == pytest.approx(expected, rel=0.01)
    
    def test_snntorch_style(self, torch_available, device):
        """Test typical snnTorch configuration."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # Typical snnTorch: β = 0.9 (leak_factor = 0.1)
        neuron = LIFNeuron(threshold=1.0, leak_factor=0.1, leak_mode="multiplicative").to(device)
        
        membrane = torch.tensor([0.0], device=device)
        input_current = torch.tensor([0.5], device=device)
        
        # Step 1: V = 0.9*0 + 0.5 = 0.5
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(0.5)
        
        # Step 2: V = 0.9*0.5 + 0.5 = 0.95
        _, membrane = neuron(input_current, membrane)
        assert membrane.item() == pytest.approx(0.95)
        
        # Step 3: V = 0.9*0.95 + 0.5 = 1.355 >= 1.0, spike!
        spikes, membrane = neuron(input_current, membrane)
        assert spikes.item() == 1.0
        assert membrane.item() == 0.0  # Reset
    
    def test_repr_multiplicative(self, torch_available):
        """Test string representation shows multiplicative mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=1.0, leak_factor=0.1, leak_mode="multiplicative")
        repr_str = repr(neuron)
        
        assert 'LIFNeuron' in repr_str
        assert 'multiplicative' in repr_str
        assert 'β=' in repr_str or 'beta' in repr_str.lower()


# =============================================================================
# LIF NEURON TESTS - COMMON
# =============================================================================


class TestLIFNeuronCommon:
    """Test common LIF neuron behavior across modes."""
    
    def test_zero_leak_like_if(self, torch_available, device):
        """Test LIF with zero leak behaves like IF."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron, IFNeuron
        
        lif = LIFNeuron(threshold=10.0, leak_factor=0.0, leak_mode="subtractive").to(device)
        if_neuron = IFNeuron(threshold=10.0).to(device)
        
        membrane_lif = torch.tensor([5.0], device=device)
        membrane_if = torch.tensor([5.0], device=device)
        input_current = torch.tensor([3.0], device=device)
        
        # Both should behave the same with zero leak
        _, mem_lif = lif(input_current, membrane_lif)
        _, mem_if = if_neuron(input_current, membrane_if)
        
        assert mem_lif.item() == pytest.approx(mem_if.item())
    
    def test_batch_processing(self, torch_available, device, batch_size, out_channels, height, width):
        """Test LIF neuron with batched input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=0.5, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.zeros(batch_size, out_channels, height, width, device=device)
        input_current = torch.rand(batch_size, out_channels, height, width, device=device)
        
        spikes, new_membrane = neuron(input_current, membrane)
        
        assert spikes.shape == membrane.shape
        assert new_membrane.shape == membrane.shape
    
    def test_invalid_mode(self, torch_available):
        """Test LIF neuron rejects invalid mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        with pytest.raises(ValueError):
            LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="invalid_mode")
    
    def test_mode_case_sensitive(self, torch_available):
        """Test leak_mode parameter is case-sensitive (only lowercase accepted)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # Lowercase works
        n1 = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive")
        n2 = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="multiplicative")
        
        assert n1.leak_mode == "subtractive"
        assert n2.leak_mode == "multiplicative"
        
        # Uppercase should fail
        with pytest.raises(ValueError):
            LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="SUBTRACTIVE")
        
        with pytest.raises(ValueError):
            LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="Multiplicative")
    
    def test_device_transfer(self, torch_available, device):
        """Test neuron can be moved to device."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive")
        neuron = neuron.to(device)
        
        assert neuron.threshold.device == device
        assert neuron.leak.device == device


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateNeuron:
    """Test create_neuron factory function."""
    
    def test_create_if(self, torch_available):
        """Test creating IF neuron via factory."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron, IFNeuron
        
        neuron = create_neuron("if", threshold=10.0)
        
        assert isinstance(neuron, IFNeuron)
        assert neuron.threshold.item() == 10.0
    
    def test_create_lif_subtractive(self, torch_available):
        """Test creating LIF neuron with subtractive mode via factory."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron, LIFNeuron
        
        neuron = create_neuron(
            "lif",
            threshold=10.0,
            leak_factor=0.9,
            leak_leak_mode="subtractive"
        )
        
        assert isinstance(neuron, LIFNeuron)
        assert neuron.threshold.item() == 10.0
        assert neuron.leak_factor == 0.9
        assert neuron.leak_mode == "subtractive"
    
    def test_create_lif_multiplicative(self, torch_available):
        """Test creating LIF neuron with multiplicative mode via factory."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron, LIFNeuron
        
        neuron = create_neuron(
            "lif",
            threshold=1.0,
            leak_factor=0.1,
            leak_leak_mode="multiplicative"
        )
        
        assert isinstance(neuron, LIFNeuron)
        assert neuron.leak_mode == "multiplicative"
    
    def test_case_insensitive(self, torch_available):
        """Test factory is case-insensitive for neuron_type."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron, IFNeuron, LIFNeuron
        
        # IF variations
        assert isinstance(create_neuron("IF", threshold=10.0), IFNeuron)
        assert isinstance(create_neuron("If", threshold=10.0), IFNeuron)
        assert isinstance(create_neuron("if", threshold=10.0), IFNeuron)
        
        # LIF variations
        assert isinstance(create_neuron("LIF", threshold=10.0), LIFNeuron)
        assert isinstance(create_neuron("Lif", threshold=10.0), LIFNeuron)
        assert isinstance(create_neuron("lif", threshold=10.0), LIFNeuron)
    
    def test_unknown_neuron_type(self, torch_available):
        """Test factory raises error for unknown neuron type."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron
        
        with pytest.raises(ValueError) as excinfo:
            create_neuron("unknown", threshold=10.0)
        
        assert "unknown" in str(excinfo.value).lower()
    
    def test_default_leak_factor(self, torch_available):
        """Test default leak_factor is 0.0."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron
        
        neuron = create_neuron("lif", threshold=10.0)
        
        assert neuron.leak_factor == 0.0
    
    def test_default_leak_mode(self, torch_available):
        """Test default leak_mode is subtractive."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import create_neuron
        
        neuron = create_neuron("lif", threshold=10.0, leak_factor=0.1)
        
        assert neuron.leak_mode == "subtractive"


# =============================================================================
# TEMPORAL BEHAVIOR TESTS
# =============================================================================


class TestNeuronTemporalBehavior:
    """Test neuron behavior over multiple timesteps."""
    
    def test_if_integration_sequence(self, torch_available, device):
        """Test IF neuron integrates over time until spike."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=5.0).to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([1.0], device=device)
        
        spike_times = []
        for t in range(10):
            spikes, membrane = neuron(input_current, membrane)
            if spikes.item() == 1.0:
                spike_times.append(t)
        
        # Should spike at t=4 (membrane: 1,2,3,4,5>=5 spike!)
        assert 4 in spike_times
    
    def test_lif_decay_prevents_spike(self, torch_available, device):
        """Test high leak prevents spike with weak input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # High leak (9.0), weak input (1.0)
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.9, leak_mode="subtractive").to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([1.0], device=device)
        
        # Run for many steps
        total_spikes = 0
        for _ in range(100):
            spikes, membrane = neuron(input_current, membrane)
            total_spikes += spikes.item()
        
        # With input (1.0) < leak (9.0), should never spike
        assert total_spikes == 0
    
    def test_lif_low_leak_accumulates(self, torch_available, device):
        """Test low leak allows accumulation and spike."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        # Low leak (1.0), input (2.0) > leak
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([2.0], device=device)
        
        # Net input per step: 2.0 - 1.0 = 1.0
        # Should spike after ~10 steps
        total_spikes = 0
        for _ in range(20):
            spikes, membrane = neuron(input_current, membrane)
            total_spikes += spikes.item()
        
        assert total_spikes > 0
    
    def test_firing_rate_varies_with_leak(self, torch_available, device):
        """Test firing rate decreases with higher leak."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        input_current = torch.tensor([5.0], device=device)
        n_steps = 100
        
        firing_rates = []
        for leak_factor in [0.0, 0.1, 0.2, 0.3]:
            neuron = LIFNeuron(threshold=10.0, leak_factor=leak_factor, leak_mode="subtractive").to(device)
            membrane = torch.zeros(1, device=device)
            
            total_spikes = 0
            for _ in range(n_steps):
                spikes, membrane = neuron(input_current, membrane)
                total_spikes += spikes.item()
            
            firing_rates.append(total_spikes / n_steps)
        
        # Higher leak should generally mean lower firing rate
        # (though exact relationship depends on input/threshold)
        assert firing_rates[0] >= firing_rates[-1]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestNeuronEdgeCases:
    """Test neuron edge cases and numerical stability."""
    
    def test_large_input(self, torch_available, device):
        """Test handling of large input values."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([1000.0], device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 1.0
        assert membrane.item() == 0.0  # Reset after spike
    
    def test_zero_input(self, torch_available, device):
        """Test handling of zero input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([5.0], device=device)
        input_current = torch.zeros(1, device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 0.0
        # V = 5 + 0 - 1 = 4
        assert membrane.item() == pytest.approx(4.0)
    
    def test_negative_input(self, torch_available, device):
        """Test handling of negative input."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.0, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([5.0], device=device)
        input_current = torch.tensor([-3.0], device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        # V = 5 + (-3) - 0 = 2
        assert membrane.item() == pytest.approx(2.0)
    
    def test_very_small_threshold(self, torch_available, device):
        """Test with very small threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=0.001).to(device)
        
        membrane = torch.zeros(1, device=device)
        input_current = torch.tensor([0.01], device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 1.0
    
    def test_very_large_threshold(self, torch_available, device):
        """Test with very large threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=1e6).to(device)
        
        membrane = torch.tensor([1000.0], device=device)
        input_current = torch.tensor([1000.0], device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        assert spikes.item() == 0.0  # Still below threshold
        assert membrane.item() == pytest.approx(2000.0)
    
    def test_nan_handling(self, torch_available, device):
        """Test behavior with NaN inputs."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(device)
        
        membrane = torch.tensor([float('nan')], device=device)
        input_current = torch.tensor([1.0], device=device)
        
        spikes, membrane = neuron(input_current, membrane)
        
        # NaN propagates (expected behavior)
        assert torch.isnan(membrane).any() or torch.isnan(spikes).any()


# =============================================================================
# CUDA TESTS
# =============================================================================


@pytest.mark.cuda
class TestNeuronsCUDA:
    """Test neurons on CUDA."""
    
    def test_if_cuda(self, cuda_device):
        """Test IF neuron on CUDA."""
        from spikeseg.core.neurons import IFNeuron
        
        neuron = IFNeuron(threshold=10.0).to(cuda_device)
        
        membrane = torch.zeros(2, 4, 16, 16, device=cuda_device)
        input_current = torch.rand(2, 4, 16, 16, device=cuda_device) * 20
        
        spikes, new_membrane = neuron(input_current, membrane)
        
        assert spikes.device == cuda_device
        assert new_membrane.device == cuda_device
    
    def test_lif_cuda(self, cuda_device):
        """Test LIF neuron on CUDA."""
        from spikeseg.core.neurons import LIFNeuron
        
        neuron = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(cuda_device)
        
        membrane = torch.zeros(2, 4, 16, 16, device=cuda_device)
        input_current = torch.rand(2, 4, 16, 16, device=cuda_device) * 20
        
        spikes, new_membrane = neuron(input_current, membrane)
        
        assert spikes.device == cuda_device
        assert new_membrane.device == cuda_device
    
    def test_cuda_cpu_consistency(self, cuda_device, cpu_device):
        """Test CUDA and CPU produce same results."""
        from spikeseg.core.neurons import LIFNeuron
        
        torch.manual_seed(42)
        
        neuron_cpu = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(cpu_device)
        neuron_cuda = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="subtractive").to(cuda_device)
        
        # Same input
        input_cpu = torch.rand(2, 4, 8, 8, device=cpu_device)
        input_cuda = input_cpu.to(cuda_device)
        
        membrane_cpu = torch.zeros_like(input_cpu)
        membrane_cuda = torch.zeros_like(input_cuda)
        
        spikes_cpu, mem_cpu = neuron_cpu(input_cpu, membrane_cpu)
        spikes_cuda, mem_cuda = neuron_cuda(input_cuda, membrane_cuda)
        
        # Compare results
        assert torch.allclose(spikes_cpu, spikes_cuda.cpu())
        assert torch.allclose(mem_cpu, mem_cuda.cpu())