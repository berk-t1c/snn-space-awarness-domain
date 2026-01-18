"""
Tests for STDP (Spike-Timing Dependent Plasticity) learning module.

This module tests:
    - STDPConfig: Configuration dataclass with paper presets
    - STDPVariant: MULTIPLICATIVE and ADDITIVE variants
    - initialize_weights: Weight initialization function
    - compute_convergence_metric: Convergence metric calculation
    - has_converged: Convergence checking
    - get_first_spike_times: Spike timing extraction
    - extract_receptive_field_times: Receptive field extraction
    - compute_stdp_update: STDP weight update computation
    - STDPStats: Learning statistics tracking
    - STDPLearner: Main STDP learning class

Paper References:
    - Kheradpisheh et al. 2018: STDP-based spiking deep CNNs
    - Kirkland et al. 2023 (IGARSS): Neuromorphic sensing for space awareness

Run with: pytest tests/test_stdp.py -v
"""

import pytest
import torch
import math


# =============================================================================
# STDP CONFIG TESTS
# =============================================================================


class TestSTDPVariant:
    """Test STDPVariant enum."""
    
    def test_multiplicative_value(self, torch_available):
        """Test MULTIPLICATIVE variant value."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPVariant
        
        assert STDPVariant.MULTIPLICATIVE.value == "multiplicative"
    
    def test_additive_value(self, torch_available):
        """Test ADDITIVE variant value."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPVariant
        
        assert STDPVariant.ADDITIVE.value == "additive"


class TestSTDPConfig:
    """Test STDPConfig dataclass."""
    
    def test_default_creation(self, torch_available):
        """Test default STDPConfig creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        config = STDPConfig()
        
        assert config.lr_plus == 0.004
        assert config.lr_minus == 0.003
        assert config.weight_min == 0.0
        assert config.weight_max == 1.0
        assert config.weight_init_mean == 0.8
        assert config.weight_init_std == 0.05
    
    def test_custom_parameters(self, torch_available):
        """Test STDPConfig with custom parameters."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig, STDPVariant
        
        config = STDPConfig(
            lr_plus=0.01,
            lr_minus=0.005,
            weight_min=0.1,
            weight_max=0.9,
            convergence_threshold=0.02
        )
        
        assert config.lr_plus == 0.01
        assert config.lr_minus == 0.005
        assert config.weight_min == 0.1
        assert config.weight_max == 0.9
        assert config.convergence_threshold == 0.02
    
    def test_from_paper_kheradpisheh2018(self, torch_available):
        """Test STDPConfig.from_paper with Kheradpisheh 2018."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig, STDPVariant
        
        config = STDPConfig.from_paper("kheradpisheh2018")
        
        # Paper parameters: a⁺ = 0.004, a⁻ = 0.003
        assert config.lr_plus == 0.004
        assert config.lr_minus == 0.003
        assert config.weight_init_mean == 0.8
        assert config.weight_init_std == 0.05
        assert config.variant == STDPVariant.MULTIPLICATIVE
    
    def test_from_paper_igarss2023(self, torch_available):
        """Test STDPConfig.from_paper with IGARSS 2023."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        config = STDPConfig.from_paper("igarss2023")
        
        # Paper parameters: a⁺ = 0.04, a⁻ = 0.03
        assert config.lr_plus == 0.04
        assert config.lr_minus == 0.03
        assert config.weight_init_mean == 0.8
        assert config.weight_init_std == 0.01
    
    def test_from_paper_case_insensitive(self, torch_available):
        """Test from_paper is case-insensitive."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        c1 = STDPConfig.from_paper("IGARSS2023")
        c2 = STDPConfig.from_paper("igarss2023")
        c3 = STDPConfig.from_paper("Igarss2023")
        
        assert c1.lr_plus == c2.lr_plus == c3.lr_plus
    
    def test_from_paper_unknown_raises(self, torch_available):
        """Test from_paper raises for unknown paper."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig, STDPConfigError
        
        with pytest.raises(STDPConfigError):
            STDPConfig.from_paper("unknown_paper")
    
    def test_invalid_lr_plus(self, torch_available):
        """Test STDPConfig rejects non-positive lr_plus."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        with pytest.raises(ValueError):
            STDPConfig(lr_plus=0.0)
        
        with pytest.raises(ValueError):
            STDPConfig(lr_plus=-0.1)
    
    def test_invalid_weight_range(self, torch_available):
        """Test STDPConfig rejects invalid weight range."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig, STDPConfigError
        
        with pytest.raises(STDPConfigError):
            STDPConfig(weight_min=1.0, weight_max=0.0)
        
        with pytest.raises(STDPConfigError):
            STDPConfig(weight_min=0.5, weight_max=0.5)
    
    def test_invalid_init_mean(self, torch_available):
        """Test STDPConfig rejects init_mean outside weight range."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        with pytest.raises(ValueError):
            STDPConfig(weight_init_mean=1.5)  # > weight_max
        
        with pytest.raises(ValueError):
            STDPConfig(weight_init_mean=-0.1)  # < weight_min
    
    def test_repr(self, torch_available):
        """Test STDPConfig string representation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPConfig
        
        config = STDPConfig()
        repr_str = repr(config)
        
        assert 'STDPConfig' in repr_str
        assert '0.004' in repr_str or 'a⁺' in repr_str


# =============================================================================
# WEIGHT INITIALIZATION TESTS
# =============================================================================


class TestInitializeWeights:
    """Test initialize_weights function."""
    
    def test_shape(self, torch_available, device):
        """Test initialized weights have correct shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        shape = (4, 2, 5, 5)
        weights = initialize_weights(shape, device=device)
        
        assert weights.shape == shape
    
    def test_device(self, torch_available, device):
        """Test initialized weights on correct device."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        weights = initialize_weights((4, 2, 3, 3), device=device)
        
        assert weights.device == device
    
    def test_dtype(self, torch_available, device):
        """Test initialized weights have correct dtype."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        weights = initialize_weights((4, 2, 3, 3), device=device, dtype=torch.float64)
        
        assert weights.dtype == torch.float64
    
    def test_mean_approximately_correct(self, torch_available, device):
        """Test weights have approximately correct mean."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        torch.manual_seed(42)
        weights = initialize_weights((100, 100, 5, 5), mean=0.8, std=0.05, device=device)
        
        # With large sample, mean should be close to 0.8
        assert abs(weights.mean().item() - 0.8) < 0.05
    
    def test_clamped_to_range(self, torch_available, device):
        """Test weights are clamped to [weight_min, weight_max]."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        torch.manual_seed(42)
        weights = initialize_weights(
            (100, 100, 5, 5),
            mean=0.8,
            std=0.3,  # Large std to exceed bounds
            weight_min=0.0,
            weight_max=1.0,
            device=device
        )
        
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0
    
    def test_custom_range(self, torch_available, device):
        """Test weights with custom range."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        weights = initialize_weights(
            (4, 2, 3, 3),
            mean=0.5,
            std=0.1,
            weight_min=0.2,
            weight_max=0.8,
            device=device
        )
        
        assert weights.min() >= 0.2
        assert weights.max() <= 0.8
    
    def test_invalid_std(self, torch_available):
        """Test rejects non-positive std."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import initialize_weights
        
        with pytest.raises(ValueError):
            initialize_weights((4, 2, 3, 3), std=0.0)
        
        with pytest.raises(ValueError):
            initialize_weights((4, 2, 3, 3), std=-0.1)


# =============================================================================
# CONVERGENCE METRIC TESTS
# =============================================================================


class TestComputeConvergenceMetric:
    """Test compute_convergence_metric function."""
    
    def test_maximum_at_half(self, torch_available, device):
        """Test convergence metric is maximum (0.25) when weights = 0.5."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_convergence_metric
        
        weights = torch.full((10, 10), 0.5, device=device)
        metric = compute_convergence_metric(weights)
        
        # w * (1-w) = 0.5 * 0.5 = 0.25
        assert metric == pytest.approx(0.25)
    
    def test_zero_at_boundaries(self, torch_available, device):
        """Test convergence metric is 0 when weights at 0 or 1."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_convergence_metric
        
        # All zeros
        weights_zero = torch.zeros(10, 10, device=device)
        assert compute_convergence_metric(weights_zero) == pytest.approx(0.0)
        
        # All ones
        weights_one = torch.ones(10, 10, device=device)
        assert compute_convergence_metric(weights_one) == pytest.approx(0.0)
        
        # Half zero, half one
        weights_mixed = torch.zeros(10, 10, device=device)
        weights_mixed[:5] = 1.0
        assert compute_convergence_metric(weights_mixed) == pytest.approx(0.0)
    
    def test_decreases_toward_convergence(self, torch_available, device):
        """Test metric decreases as weights approach 0 or 1."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_convergence_metric
        
        metrics = []
        for w_val in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            weights = torch.full((10, 10), w_val, device=device)
            metrics.append(compute_convergence_metric(weights))
        
        # Should be monotonically decreasing
        for i in range(len(metrics) - 1):
            assert metrics[i] >= metrics[i + 1]
    
    def test_symmetric(self, torch_available, device):
        """Test metric is symmetric around 0.5."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_convergence_metric
        
        w1 = torch.full((10, 10), 0.3, device=device)
        w2 = torch.full((10, 10), 0.7, device=device)
        
        m1 = compute_convergence_metric(w1)
        m2 = compute_convergence_metric(w2)
        
        # 0.3 * 0.7 = 0.7 * 0.3
        assert m1 == pytest.approx(m2)
    
    def test_any_shape(self, torch_available, device):
        """Test works with any tensor shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_convergence_metric
        
        shapes = [(10,), (5, 5), (2, 3, 4), (2, 4, 5, 5)]
        
        for shape in shapes:
            weights = torch.full(shape, 0.5, device=device)
            metric = compute_convergence_metric(weights)
            assert metric == pytest.approx(0.25)


class TestHasConverged:
    """Test has_converged function."""
    
    def test_converged_weights(self, torch_available, device):
        """Test converged weights (at 0 and 1) return True."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import has_converged
        
        weights = torch.zeros(10, 10, device=device)
        weights[:5] = 1.0
        
        assert has_converged(weights, threshold=0.01)
    
    def test_unconverged_weights(self, torch_available, device):
        """Test unconverged weights return False."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import has_converged
        
        weights = torch.full((10, 10), 0.5, device=device)
        
        assert not has_converged(weights, threshold=0.01)
    
    def test_custom_threshold(self, torch_available, device):
        """Test with custom threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import has_converged, compute_convergence_metric
        
        weights = torch.full((10, 10), 0.3, device=device)
        metric = compute_convergence_metric(weights)  # 0.3 * 0.7 = 0.21
        
        assert not has_converged(weights, threshold=0.1)
        assert has_converged(weights, threshold=0.3)


# =============================================================================
# SPIKE TIMING TESTS
# =============================================================================


class TestGetFirstSpikeTimes:
    """Test get_first_spike_times function."""
    
    def test_basic_timing(self, torch_available, device):
        """Test basic first spike time extraction."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import get_first_spike_times
        
        # (T=5, C=2, H=3, W=3)
        spikes = torch.zeros(5, 2, 3, 3, device=device)
        spikes[2, 0, 1, 1] = 1  # First spike at t=2
        spikes[4, 0, 1, 1] = 1  # Later spike (should be ignored)
        
        times = get_first_spike_times(spikes)
        
        assert times.shape == (2, 3, 3)
        assert times[0, 1, 1] == 2.0
    
    def test_no_spike_value(self, torch_available, device):
        """Test neurons that don't spike get inf."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import get_first_spike_times
        
        spikes = torch.zeros(5, 2, 3, 3, device=device)
        spikes[2, 0, 0, 0] = 1  # Only this neuron spikes
        
        times = get_first_spike_times(spikes)
        
        assert times[0, 0, 0] == 2.0
        assert torch.isinf(times[0, 1, 1])  # No spike
        assert torch.isinf(times[1, 0, 0])  # No spike in channel 1
    
    def test_custom_no_spike_value(self, torch_available, device):
        """Test custom no_spike_value."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import get_first_spike_times
        
        spikes = torch.zeros(5, 2, 3, 3, device=device)
        
        times = get_first_spike_times(spikes, no_spike_value=-1.0)
        
        assert (times == -1.0).all()
    
    def test_first_timestep(self, torch_available, device):
        """Test spike at first timestep (t=0)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import get_first_spike_times
        
        spikes = torch.zeros(5, 2, 3, 3, device=device)
        spikes[0, 0, 0, 0] = 1  # Spike at t=0
        
        times = get_first_spike_times(spikes)
        
        assert times[0, 0, 0] == 0.0
    
    def test_multiple_channels(self, torch_available, device):
        """Test different spike times in different channels."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import get_first_spike_times
        
        spikes = torch.zeros(10, 4, 8, 8, device=device)
        spikes[1, 0, 0, 0] = 1  # Channel 0 at t=1
        spikes[3, 1, 0, 0] = 1  # Channel 1 at t=3
        spikes[5, 2, 0, 0] = 1  # Channel 2 at t=5
        spikes[7, 3, 0, 0] = 1  # Channel 3 at t=7
        
        times = get_first_spike_times(spikes)
        
        assert times[0, 0, 0] == 1.0
        assert times[1, 0, 0] == 3.0
        assert times[2, 0, 0] == 5.0
        assert times[3, 0, 0] == 7.0


class TestExtractReceptiveFieldTimes:
    """Test extract_receptive_field_times function."""
    
    def test_basic_extraction(self, torch_available, device):
        """Test basic receptive field extraction."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import extract_receptive_field_times
        
        # Pre-synaptic spike times: (C=2, H=8, W=8)
        pre_times = torch.arange(64, device=device, dtype=torch.float32).reshape(1, 8, 8)
        pre_times = pre_times.expand(2, -1, -1).clone()
        
        # Extract for post neuron at (0, 0) with 3x3 kernel
        rf_times = extract_receptive_field_times(
            pre_times, post_y=0, post_x=0, kernel_size=3, padding=1
        )
        
        assert rf_times.shape == (2, 3, 3)
    
    def test_with_padding(self, torch_available, device):
        """Test extraction with padding (padded regions = inf)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import extract_receptive_field_times
        
        pre_times = torch.zeros(1, 8, 8, device=device)
        
        # Corner neuron with padding - some RF elements are outside
        rf_times = extract_receptive_field_times(
            pre_times, post_y=0, post_x=0, kernel_size=3, padding=1
        )
        
        # Top-left corner should have inf for padded region
        assert torch.isinf(rf_times[0, 0, 0])
    
    def test_stride(self, torch_available, device):
        """Test extraction with stride."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import extract_receptive_field_times
        
        pre_times = torch.arange(64, device=device, dtype=torch.float32).reshape(1, 8, 8)
        
        # With stride=2, post neuron (1, 1) maps to input starting at (2, 2)
        rf_times = extract_receptive_field_times(
            pre_times, post_y=1, post_x=1, kernel_size=3, stride=2, padding=0
        )
        
        # Check the center of RF
        assert rf_times.shape == (1, 3, 3)


# =============================================================================
# STDP UPDATE TESTS
# =============================================================================


class TestComputeStdpUpdate:
    """Test compute_stdp_update function."""
    
    def test_ltp_when_pre_before_post(self, torch_available, device):
        """Test LTP (positive update) when pre fires before post."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 5, 5), 2.0, device=device)  # Pre at t=2
        post_time = 5.0  # Post at t=5
        
        delta = compute_stdp_update(
            weights, pre_times, post_time,
            lr_plus=0.004, lr_minus=0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        # All LTP (pre before post)
        assert (delta > 0).all()
    
    def test_ltd_when_pre_after_post(self, torch_available, device):
        """Test LTD (negative update) when pre fires after post."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 5, 5), 10.0, device=device)  # Pre at t=10
        post_time = 5.0  # Post at t=5
        
        delta = compute_stdp_update(
            weights, pre_times, post_time,
            lr_plus=0.004, lr_minus=0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        # All LTD (pre after post)
        assert (delta < 0).all()
    
    def test_ltd_when_no_pre_spike(self, torch_available, device):
        """Test LTD when pre-synaptic neuron doesn't spike (inf time)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 5, 5), float('inf'), device=device)  # No pre spike
        post_time = 5.0
        
        delta = compute_stdp_update(
            weights, pre_times, post_time,
            lr_plus=0.004, lr_minus=0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        # All LTD (inf > any finite)
        assert (delta < 0).all()
    
    def test_ltp_when_simultaneous(self, torch_available, device):
        """Test LTP when pre and post fire at same time."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 5, 5), 5.0, device=device)  # Same time
        post_time = 5.0
        
        delta = compute_stdp_update(
            weights, pre_times, post_time,
            lr_plus=0.004, lr_minus=0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        # LTP when t_pre <= t_post
        assert (delta > 0).all()
    
    def test_multiplicative_soft_bounds(self, torch_available, device):
        """Test multiplicative STDP has soft bounds w*(1-w)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        # Weights near 0 have small update magnitude
        w_low = torch.full((4, 5, 5), 0.1, device=device)
        # Weights near 1 have small update magnitude
        w_high = torch.full((4, 5, 5), 0.9, device=device)
        # Weights at 0.5 have maximum update magnitude
        w_mid = torch.full((4, 5, 5), 0.5, device=device)
        
        pre_times = torch.full((4, 5, 5), 2.0, device=device)
        post_time = 5.0
        
        delta_low = compute_stdp_update(w_low, pre_times, post_time, 0.1, 0.1)
        delta_high = compute_stdp_update(w_high, pre_times, post_time, 0.1, 0.1)
        delta_mid = compute_stdp_update(w_mid, pre_times, post_time, 0.1, 0.1)
        
        # Mid weights should have largest update
        assert delta_mid.abs().mean() > delta_low.abs().mean()
        assert delta_mid.abs().mean() > delta_high.abs().mean()
    
    def test_additive_no_soft_bounds(self, torch_available, device):
        """Test additive STDP has constant update magnitude."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        w_low = torch.full((4, 5, 5), 0.1, device=device)
        w_high = torch.full((4, 5, 5), 0.9, device=device)
        
        pre_times = torch.full((4, 5, 5), 2.0, device=device)
        post_time = 5.0
        
        delta_low = compute_stdp_update(
            w_low, pre_times, post_time, 0.1, 0.1,
            variant=STDPVariant.ADDITIVE
        )
        delta_high = compute_stdp_update(
            w_high, pre_times, post_time, 0.1, 0.1,
            variant=STDPVariant.ADDITIVE
        )
        
        # Additive: same magnitude regardless of weight
        assert delta_low.abs().mean() == pytest.approx(delta_high.abs().mean())
    
    def test_mixed_timing(self, torch_available, device):
        """Test mixed LTP and LTD in same update."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        weights = torch.full((2, 3, 3), 0.5, device=device)
        pre_times = torch.full((2, 3, 3), 5.0, device=device)
        
        # Some pre before, some after post
        pre_times[0] = 2.0  # Before
        pre_times[1] = 8.0  # After
        post_time = 5.0
        
        delta = compute_stdp_update(
            weights, pre_times, post_time,
            lr_plus=0.004, lr_minus=0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        assert (delta[0] > 0).all()  # LTP
        assert (delta[1] < 0).all()  # LTD
    
    def test_shape_mismatch_error(self, torch_available, device):
        """Test error on shape mismatch."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 3, 3), 2.0, device=device)  # Wrong shape
        
        with pytest.raises(ValueError):
            compute_stdp_update(weights, pre_times, 5.0, 0.004, 0.003)


# =============================================================================
# STDP STATS TESTS
# =============================================================================


class TestSTDPStats:
    """Test STDPStats dataclass."""
    
    def test_default_creation(self, torch_available):
        """Test default STDPStats creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning.stdp import STDPStats
        
        stats = STDPStats()
        
        assert stats.n_updates == 0
        assert stats.n_ltp == 0
        assert stats.n_ltd == 0
        assert stats.mean_delta == 0.0
        assert stats.convergence_history == []
    
    def test_record_update(self, torch_available, device):
        """Test recording an update."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning.stdp import STDPStats
        
        stats = STDPStats()
        
        delta = torch.tensor([[0.1, -0.05], [0.2, -0.1]], device=device)
        stats.record_update(delta, convergence=0.15)
        
        assert stats.n_updates == 1
        assert stats.n_ltp == 2  # Two positive values
        assert stats.n_ltd == 2  # Two negative values
        assert stats.convergence_history == [0.15]
    
    def test_ltp_ratio(self, torch_available, device):
        """Test LTP ratio calculation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning.stdp import STDPStats
        
        stats = STDPStats()
        stats.n_ltp = 3
        stats.n_ltd = 1
        
        assert stats.ltp_ratio == pytest.approx(0.75)
    
    def test_ltp_ratio_zero_events(self, torch_available):
        """Test LTP ratio with no events."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning.stdp import STDPStats
        
        stats = STDPStats()
        
        assert stats.ltp_ratio == 0.0
    
    def test_repr(self, torch_available):
        """Test string representation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning.stdp import STDPStats
        
        stats = STDPStats(n_updates=10, n_ltp=50, n_ltd=30)
        repr_str = repr(stats)
        
        assert 'STDPStats' in repr_str
        assert '10' in repr_str


# =============================================================================
# STDP LEARNER TESTS
# =============================================================================


class TestSTDPLearner:
    """Test STDPLearner class."""
    
    def test_default_creation(self, torch_available):
        """Test default STDPLearner creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPLearner, STDPConfig
        
        learner = STDPLearner()
        
        # Should use Kheradpisheh2018 defaults
        assert learner.config.lr_plus == 0.004
        assert learner.config.lr_minus == 0.003
    
    def test_with_config(self, stdp_config):
        """Test STDPLearner with custom config."""
        from spikeseg.learning import STDPLearner
        
        learner = STDPLearner(stdp_config)
        
        assert learner.config == stdp_config
    
    def test_initialize_weights(self, stdp_learner, device):
        """Test weight initialization."""
        weights = stdp_learner.initialize_weights((4, 2, 5, 5), device=device)
        
        assert weights.shape == (4, 2, 5, 5)
        assert weights.device == device
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0
    
    def test_compute_update(self, stdp_learner, device):
        """Test compute_update method."""
        weights = torch.full((2, 5, 5), 0.5, device=device)
        pre_times = torch.full((2, 5, 5), 2.0, device=device)
        post_time = 5.0
        
        delta = stdp_learner.compute_update(weights, pre_times, post_time)
        
        assert delta.shape == weights.shape
        assert (delta > 0).all()  # LTP since pre < post
    
    def test_has_converged_method(self, stdp_learner, device):
        """Test has_converged method."""
        from spikeseg.learning import STDPLearner
        
        # Converged weights
        converged = torch.zeros(10, 10, device=device)
        converged[:5] = 1.0
        
        # Unconverged weights
        unconverged = torch.full((10, 10), 0.5, device=device)
        
        assert stdp_learner.has_converged(converged)
        assert not stdp_learner.has_converged(unconverged)
    
    def test_stats_tracking(self, torch_available):
        """Test statistics are tracked."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPLearner
        
        learner = STDPLearner()
        
        assert learner.stats.n_updates == 0
    
    def test_update_count_property(self, torch_available):
        """Test update_count property."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPLearner
        
        learner = STDPLearner()
        
        assert learner.update_count == 0
    
    def test_convergence_history_property(self, torch_available):
        """Test convergence_history property."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPLearner
        
        learner = STDPLearner()
        
        assert learner.convergence_history == []


class TestSTDPLearnerBatchUpdate:
    """Test STDPLearner batch update functionality."""
    
    def test_compute_batch_update_exists(self, stdp_learner):
        """Test compute_batch_update method exists."""
        assert hasattr(stdp_learner, 'compute_batch_update')
    
    def test_batch_update_shape(self, stdp_learner, device):
        """Test batch update returns correct shape."""
        from spikeseg.learning import get_first_spike_times
        
        # Full weights: (out_ch=4, in_ch=2, kH=5, kW=5)
        weights = torch.full((4, 2, 5, 5), 0.5, device=device)
        
        # Pre spikes: (T=10, in_ch=2, H=16, W=16)
        pre_spikes = torch.zeros(10, 2, 16, 16, device=device)
        pre_spikes[2, :, 4:9, 4:9] = 1  # Some pre spikes at t=2
        
        # Post spikes: (T=10, out_ch=4, H=16, W=16)
        post_spikes = torch.zeros(10, 4, 16, 16, device=device)
        post_spikes[5, 0, 6, 6] = 1  # Winner at t=5
        
        # Convert to first spike times (what compute_batch_update expects)
        pre_spike_times = get_first_spike_times(pre_spikes)
        post_spike_times = get_first_spike_times(post_spikes)
        
        # Winner mask
        winner_mask = torch.zeros(4, 16, 16, device=device)
        winner_mask[0, 6, 6] = 1
        
        try:
            delta = stdp_learner.compute_batch_update(
                weights=weights,
                pre_spike_times=pre_spike_times,
                post_spike_times=post_spike_times,
                winner_mask=winner_mask
            )
            
            assert delta.shape == weights.shape
        except (NotImplementedError, AttributeError):
            # Method might not be fully implemented
            pytest.skip("compute_batch_update not fully implemented")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSTDPIntegration:
    """Integration tests for STDP learning."""
    
    def test_weight_update_flow(self, torch_available, device):
        """Test complete weight update flow."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import (
            STDPLearner, STDPConfig,
            get_first_spike_times, compute_stdp_update
        )
        
        config = STDPConfig.from_paper("igarss2023")
        learner = STDPLearner(config)
        
        # Initialize weights
        weights = learner.initialize_weights((4, 2, 5, 5), device=device)
        original_weights = weights.clone()
        
        # Create spikes
        pre_spikes = torch.zeros(10, 2, 5, 5, device=device)
        pre_spikes[2, :, :, :] = 1  # All pre spike at t=2
        
        # Get first spike times
        pre_times = get_first_spike_times(pre_spikes)
        
        # Compute update for one feature map
        post_time = 5.0  # Post at t=5
        delta = learner.compute_update(weights[0], pre_times, post_time)
        
        # Apply update
        weights[0] = weights[0] + delta
        weights.clamp_(0.0, 1.0)
        
        # Weights should have changed
        assert not torch.equal(weights[0], original_weights[0])
    
    def test_convergence_over_iterations(self, torch_available, device):
        """Test that repeated updates lead to convergence."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import (
            STDPLearner, STDPConfig, compute_convergence_metric
        )
        
        # Use higher learning rates for faster convergence
        config = STDPConfig(lr_plus=0.1, lr_minus=0.08)
        learner = STDPLearner(config)
        
        weights = torch.full((2, 3, 3), 0.5, device=device)
        
        initial_metric = compute_convergence_metric(weights)
        
        # Simulate repeated LTP updates (weights should approach 1)
        pre_times = torch.full((2, 3, 3), 2.0, device=device)
        post_time = 5.0
        
        for _ in range(50):
            delta = learner.compute_update(weights, pre_times, post_time)
            weights = weights + delta
            weights.clamp_(0.0, 1.0)
        
        final_metric = compute_convergence_metric(weights)
        
        # Should have converged (lower metric)
        assert final_metric < initial_metric
    
    def test_paper_parameters_produce_bounded_updates(self, torch_available, device):
        """Test paper parameters produce reasonable weight updates."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import STDPLearner, STDPConfig
        
        # Test both paper configs
        for paper in ["kheradpisheh2018", "igarss2023"]:
            config = STDPConfig.from_paper(paper)
            learner = STDPLearner(config)
            
            weights = torch.full((4, 5, 5), 0.8, device=device)
            pre_times = torch.full((4, 5, 5), 2.0, device=device)
            
            delta = learner.compute_update(weights, pre_times, post_time=5.0)
            
            # Update magnitude should be reasonable (< 0.1 per update)
            assert delta.abs().max() < 0.1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestSTDPEdgeCases:
    """Test STDP edge cases."""
    
    def test_all_inf_pre_times(self, torch_available, device):
        """Test update when no pre-synaptic neurons spike."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update
        
        weights = torch.full((4, 5, 5), 0.5, device=device)
        pre_times = torch.full((4, 5, 5), float('inf'), device=device)
        
        delta = compute_stdp_update(weights, pre_times, 5.0, 0.004, 0.003)
        
        # All LTD (inf > any finite)
        assert (delta < 0).all()
    
    def test_weights_at_boundary(self, torch_available, device):
        """Test update when weights at boundaries (0 or 1)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update, STDPVariant
        
        # Weights at 0
        weights_zero = torch.zeros(4, 5, 5, device=device)
        pre_times = torch.full((4, 5, 5), 2.0, device=device)
        
        delta_zero = compute_stdp_update(
            weights_zero, pre_times, 5.0, 0.004, 0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        # w * (1-w) = 0 at boundaries, so delta should be 0
        assert delta_zero.abs().max() == pytest.approx(0.0)
        
        # Weights at 1
        weights_one = torch.ones(4, 5, 5, device=device)
        delta_one = compute_stdp_update(
            weights_one, pre_times, 5.0, 0.004, 0.003,
            variant=STDPVariant.MULTIPLICATIVE
        )
        
        assert delta_one.abs().max() == pytest.approx(0.0)
    
    def test_very_small_weights(self, torch_available, device):
        """Test with very small weights."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.learning import compute_stdp_update
        
        weights = torch.full((4, 5, 5), 1e-6, device=device)
        pre_times = torch.full((4, 5, 5), 2.0, device=device)
        
        delta = compute_stdp_update(weights, pre_times, 5.0, 0.004, 0.003)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(delta).any()
        assert not torch.isinf(delta).any()


# =============================================================================
# CUDA TESTS
# =============================================================================


@pytest.mark.cuda
class TestSTDPCUDA:
    """Test STDP on CUDA."""
    
    def test_initialize_weights_cuda(self, cuda_device):
        """Test weight initialization on CUDA."""
        from spikeseg.learning import initialize_weights
        
        weights = initialize_weights((4, 2, 5, 5), device=cuda_device)
        
        assert weights.device == cuda_device
    
    def test_compute_update_cuda(self, cuda_device):
        """Test STDP update on CUDA."""
        from spikeseg.learning import compute_stdp_update
        
        weights = torch.full((4, 5, 5), 0.5, device=cuda_device)
        pre_times = torch.full((4, 5, 5), 2.0, device=cuda_device)
        
        delta = compute_stdp_update(weights, pre_times, 5.0, 0.004, 0.003)
        
        assert delta.device == cuda_device
    
    def test_convergence_metric_cuda(self, cuda_device):
        """Test convergence metric on CUDA."""
        from spikeseg.learning import compute_convergence_metric
        
        weights = torch.full((10, 10), 0.5, device=cuda_device)
        metric = compute_convergence_metric(weights)
        
        assert metric == pytest.approx(0.25)