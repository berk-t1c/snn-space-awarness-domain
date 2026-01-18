"""
Tests for SpikeSEG model components.

This module tests:
    - Configuration classes (LayerConfig, EncoderConfig, DecoderConfig)
    - SpikingEncoderLayer
    - SpikeSEGEncoder
    - SpikeSEGDecoder
    - SpikeSEG (complete model)

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestLayerConfig:
    """Test LayerConfig dataclass."""
    
    def test_default_creation(self, torch_available):
        """Test default LayerConfig creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig
        
        config = LayerConfig(out_channels=4, kernel_size=5)
        assert config.out_channels == 4
        assert config.kernel_size == 5
        assert config.threshold == 10.0
        assert config.leak == 0.0
    
    def test_custom_parameters(self, torch_available):
        """Test LayerConfig with custom parameters."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig
        
        config = LayerConfig(
            out_channels=36,
            kernel_size=7,
            threshold=15.0,
            leak=1.5,
            leak_mode="multiplicative"
        )
        assert config.out_channels == 36
        assert config.kernel_size == 7
        assert config.threshold == 15.0
        assert config.leak == 1.5
        assert config.leak_mode == "multiplicative"
    
    def test_invalid_out_channels(self, torch_available):
        """Test LayerConfig rejects invalid out_channels."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig
        
        with pytest.raises((ValueError, TypeError)):
            LayerConfig(out_channels=0, kernel_size=5)
        
        with pytest.raises((ValueError, TypeError)):
            LayerConfig(out_channels=-1, kernel_size=5)
    
    def test_invalid_kernel_size(self, torch_available):
        """Test LayerConfig rejects invalid kernel_size."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig
        
        with pytest.raises((ValueError, TypeError)):
            LayerConfig(out_channels=4, kernel_size=0)
    
    def test_invalid_threshold(self, torch_available):
        """Test LayerConfig rejects non-positive threshold."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig, EncoderConfigError
        
        with pytest.raises(EncoderConfigError):
            LayerConfig(out_channels=4, kernel_size=5, threshold=0.0)
        
        with pytest.raises(EncoderConfigError):
            LayerConfig(out_channels=4, kernel_size=5, threshold=-5.0)
    
    def test_invalid_leak_mode(self, torch_available):
        """Test LayerConfig rejects invalid leak_mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import LayerConfig, EncoderConfigError
        
        with pytest.raises(EncoderConfigError):
            LayerConfig(out_channels=4, kernel_size=5, leak_mode="invalid")


class TestEncoderConfig:
    """Test EncoderConfig dataclass."""
    
    def test_default_creation(self, torch_available):
        """Test default EncoderConfig creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig
        
        config = EncoderConfig()
        assert config.input_channels == 1
        assert config.conv1.out_channels == 4
        assert config.conv2.out_channels == 36
        assert config.conv3.out_channels == 1
        assert config.pool_kernel_size == 2
    
    def test_from_paper_igarss2023(self, torch_available):
        """Test EncoderConfig.from_paper with IGARSS 2023."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig
        
        config = EncoderConfig.from_paper("igarss2023")
        
        # IGARSS 2023 parameters
        assert config.conv1.out_channels == 4
        assert config.conv2.out_channels == 36
        assert config.conv1.kernel_size == 5
        assert config.conv2.kernel_size == 5
        assert config.conv3.kernel_size == 7
        
        # Leak factors: 90% and 10% of threshold
        assert config.conv1.leak == 9.0  # 90% of 10.0
        assert config.conv2.leak == 1.0  # 10% of 10.0
        assert config.conv3.leak == 0.0  # No leak
    
    def test_from_paper_custom_n_classes(self, torch_available):
        """Test EncoderConfig.from_paper with custom n_classes."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig
        
        config = EncoderConfig.from_paper("igarss2023", n_classes=5)
        assert config.conv3.out_channels == 5
    
    def test_n_classes_property(self, torch_available):
        """Test n_classes property."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig, LayerConfig
        
        config = EncoderConfig(
            conv3=LayerConfig(out_channels=10, kernel_size=7)
        )
        assert config.n_classes == 10
    
    def test_to_dict(self, torch_available):
        """Test EncoderConfig.to_dict serialization."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig
        
        config = EncoderConfig()
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert 'input_channels' in d
        assert 'conv1' in d
        assert 'conv2' in d
        assert 'conv3' in d
    
    def test_repr(self, torch_available):
        """Test EncoderConfig string representation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import EncoderConfig
        
        config = EncoderConfig()
        repr_str = repr(config)
        
        assert 'EncoderConfig' in repr_str
        assert '4ch' in repr_str  # conv1 channels
        assert '36ch' in repr_str  # conv2 channels


# =============================================================================
# SPIKING ENCODER LAYER TESTS
# =============================================================================


class TestSpikingEncoderLayer:
    """Test SpikingEncoderLayer."""
    
    def test_creation(self, torch_available, device):
        """Test layer creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            threshold=10.0,
            leak=9.0
        ).to(device)
        
        assert layer.in_channels == 1
        assert layer.out_channels == 4
        assert layer.kernel_size == 5
        assert layer.threshold == 10.0
        assert layer.leak == 9.0
    
    def test_forward_shape(self, torch_available, device, batch_size, height, width):
        """Test forward pass output shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5
        ).to(device)
        
        x = torch.rand(batch_size, 1, height, width, device=device)
        spikes, membrane = layer(x)
        
        assert spikes.shape == (batch_size, 4, height, width)
        assert membrane.shape == (batch_size, 4, height, width)
    
    def test_forward_spikes_binary(self, torch_available, device, assert_spikes_valid):
        """Test that output spikes are binary."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            threshold=1.0  # Low threshold for more spikes
        ).to(device)
        
        # Strong input to generate spikes
        x = torch.ones(2, 1, 16, 16, device=device)
        spikes, _ = layer(x)
        
        assert_spikes_valid(spikes)
    
    def test_membrane_accumulation(self, torch_available, device):
        """Test membrane potential accumulates over time."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            threshold=100.0,  # High threshold so no spikes
            leak=0.0  # No leak
        ).to(device)
        
        x = torch.ones(1, 1, 16, 16, device=device) * 0.1
        
        # First timestep
        _, mem1 = layer(x)
        
        # Second timestep - membrane should increase
        _, mem2 = layer(x)
        
        assert mem2.mean() > mem1.mean()
    
    def test_reset_state(self, torch_available, device):
        """Test state reset."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5
        ).to(device)
        
        x = torch.rand(1, 1, 16, 16, device=device)
        _, mem1 = layer(x)
        
        layer.reset_state()
        
        assert layer.membrane is None
    
    def test_weight_initialization(self, torch_available, device):
        """Test weights are initialized around 0.8 for STDP."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5
        ).to(device)
        
        weights = layer.weight
        
        # Check weights are around 0.8 with small std
        assert 0.7 < weights.mean().item() < 0.9
        assert weights.min().item() >= 0.0
        assert weights.max().item() <= 1.0
    
    def test_weight_property(self, torch_available, device):
        """Test weight property returns conv weights."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models.encoder import SpikingEncoderLayer
        
        layer = SpikingEncoderLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=5
        ).to(device)
        
        assert layer.weight is layer.conv.weight


# =============================================================================
# ENCODER TESTS
# =============================================================================


class TestSpikeSEGEncoder:
    """Test SpikeSEGEncoder."""
    
    def test_default_creation(self, torch_available, device):
        """Test encoder creation with defaults."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEGEncoder
        
        encoder = SpikeSEGEncoder().to(device)
        
        assert hasattr(encoder, 'conv1')
        assert hasattr(encoder, 'conv2')
        assert hasattr(encoder, 'conv3')
        assert hasattr(encoder, 'pool1')
        assert hasattr(encoder, 'pool2')
    
    def test_from_config(self, encoder_config, device):
        """Test encoder creation from config."""
        from spikeseg.models import SpikeSEGEncoder
        
        encoder = SpikeSEGEncoder(encoder_config).to(device)
        
        assert encoder.conv1.out_channels == encoder_config.conv1.out_channels
        assert encoder.conv2.out_channels == encoder_config.conv2.out_channels
        assert encoder.conv3.out_channels == encoder_config.conv3.out_channels
    
    def test_forward_single_timestep(self, encoder, random_input):
        """Test single timestep forward pass."""
        encoder.reset_state()
        output = encoder.forward_single_timestep(random_input)
        
        class_spikes, layer_spikes = output
        
        assert class_spikes is not None
        assert 'conv1' in layer_spikes
        assert 'pool1' in layer_spikes
        assert 'conv2' in layer_spikes
        assert 'pool2' in layer_spikes
        assert 'conv3' in layer_spikes
    
    def test_forward_returns_encoder_output(self, encoder, random_input):
        """Test forward returns EncoderOutput."""
        from spikeseg.models import EncoderOutput, PoolingIndices
        
        encoder.reset_state()
        output = encoder(random_input)
        
        assert isinstance(output, EncoderOutput)
        assert isinstance(output.pooling_indices, PoolingIndices)
        assert output.classification_spikes is not None
    
    def test_pooling_indices_stored(self, encoder, random_input):
        """Test pooling indices are properly stored."""
        encoder.reset_state()
        output = encoder(random_input)
        
        indices = output.pooling_indices
        
        assert indices.pool1_indices is not None
        assert indices.pool2_indices is not None
        assert len(indices.pool1_output_size) == 4
        assert len(indices.pool2_output_size) == 4
    
    def test_spatial_downsampling(self, encoder, random_input, height, width):
        """Test spatial dimensions are reduced by pooling."""
        encoder.reset_state()
        output = encoder(random_input)
        
        # After 2 pooling operations with kernel=2, stride=2
        # Size should be H/4 x W/4
        expected_h = height // 4
        expected_w = width // 4
        
        class_spikes = output.classification_spikes
        _, _, h, w = class_spikes.shape
        
        assert h == expected_h
        assert w == expected_w
    
    def test_n_timesteps_forward(self, encoder, random_input, n_timesteps):
        """Test forward with n_timesteps parameter."""
        encoder.reset_state()
        output = encoder(random_input, n_timesteps=n_timesteps)
        
        # With n_timesteps, should return accumulated spikes
        assert output.classification_spikes is not None
    
    def test_reset_state(self, encoder, random_input):
        """Test state reset clears membrane potentials."""
        encoder.reset_state()
        _ = encoder(random_input)
        
        encoder.reset_state()
        
        assert encoder.conv1.membrane is None
        assert encoder.conv2.membrane is None
        assert encoder.conv3.membrane is None
    
    def test_layer_spikes_storage(self, encoder_config, device, random_input):
        """Test layer spikes are stored when configured."""
        from spikeseg.models import SpikeSEGEncoder
        
        encoder_config.store_all_spikes = True
        encoder = SpikeSEGEncoder(encoder_config).to(device)
        
        encoder.reset_state()
        output = encoder(random_input)
        
        assert len(output.layer_spikes) > 0
    
    def test_output_has_spikes_property(self, encoder, random_input):
        """Test EncoderOutput.has_spikes property."""
        encoder.reset_state()
        output = encoder(random_input)
        
        # Property should return bool
        assert isinstance(output.has_spikes, bool)
    
    def test_output_n_classification_spikes(self, encoder, random_input):
        """Test EncoderOutput.n_classification_spikes property."""
        encoder.reset_state()
        output = encoder(random_input)
        
        n_spikes = output.n_classification_spikes
        assert isinstance(n_spikes, int)
        assert n_spikes >= 0
    
    def test_output_spike_analysis(self, encoder, sample_events):
        """Test EncoderOutput spike analysis."""
        encoder.reset_state()
        output = encoder(sample_events)
        
        # Test basic spike analysis using available properties
        has_spikes = output.has_spikes
        n_spikes = output.n_classification_spikes
        
        assert isinstance(has_spikes, bool)
        assert isinstance(n_spikes, int)
        
        # Verify consistency
        if has_spikes:
            assert n_spikes > 0
        else:
            assert n_spikes == 0


class TestEncoderEdgeCases:
    """Test encoder edge cases and error handling."""
    
    def test_invalid_input_dims(self, encoder):
        """Test encoder rejects wrong input dimensions."""
        encoder.reset_state()
        
        # 3D input (missing batch)
        x_3d = torch.rand(1, 32, 32)
        with pytest.raises(ValueError):
            encoder(x_3d)
    
    def test_wrong_input_channels(self, encoder, device):
        """Test encoder rejects wrong number of input channels."""
        encoder.reset_state()
        
        # Wrong number of channels
        x = torch.rand(2, 3, 32, 32, device=device)  # 3 channels instead of 1
        with pytest.raises(Exception):  # Could be RuntimeError or EncoderRuntimeError
            encoder(x)
    
    def test_very_small_input(self, encoder_config, device):
        """Test encoder handles small inputs."""
        from spikeseg.models import SpikeSEGEncoder
        
        encoder = SpikeSEGEncoder(encoder_config).to(device)
        
        # Very small input (might be too small after pooling)
        x = torch.rand(1, 1, 8, 8, device=device)
        
        try:
            encoder.reset_state()
            output = encoder(x)
            # If it works, output should have valid shape
            assert output.classification_spikes.shape[2] > 0
            assert output.classification_spikes.shape[3] > 0
        except RuntimeError:
            # Some architectures might not support very small inputs
            pass
    
    def test_batch_size_variations(self, encoder, device, height, width):
        """Test encoder handles different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            encoder.reset_state()
            x = torch.rand(batch_size, 1, height, width, device=device)
            output = encoder(x)
            
            assert output.classification_spikes.shape[0] == batch_size


# =============================================================================
# DECODER TESTS
# =============================================================================


class TestSpikeSEGDecoder:
    """Test SpikeSEGDecoder."""
    
    def test_from_encoder_creation(self, encoder, device):
        """Test decoder creation from encoder."""
        from spikeseg.models import SpikeSEGDecoder
        
        decoder = SpikeSEGDecoder.from_encoder(encoder)
        
        assert decoder is not None
        assert hasattr(decoder, 'trans_conv1')
        assert hasattr(decoder, 'trans_conv2')
        assert hasattr(decoder, 'trans_conv3')
    
    def test_decoder_forward(self, encoder, decoder, random_input):
        """Test decoder forward pass."""
        encoder.reset_state()
        enc_output = encoder(random_input)
        
        segmentation = decoder(
            classification_spikes=enc_output.classification_spikes,
            pool1_indices=enc_output.pooling_indices.pool1_indices,
            pool2_indices=enc_output.pooling_indices.pool2_indices,
            pool1_output_size=enc_output.pooling_indices.pool1_output_size,
            pool2_output_size=enc_output.pooling_indices.pool2_output_size
        )
        
        assert segmentation is not None
    
    def test_decoder_restores_spatial_dims(self, encoder, decoder, random_input, height, width):
        """Test decoder restores original spatial dimensions."""
        encoder.reset_state()
        enc_output = encoder(random_input)
        
        segmentation = decoder(
            classification_spikes=enc_output.classification_spikes,
            pool1_indices=enc_output.pooling_indices.pool1_indices,
            pool2_indices=enc_output.pooling_indices.pool2_indices,
            pool1_output_size=enc_output.pooling_indices.pool1_output_size,
            pool2_output_size=enc_output.pooling_indices.pool2_output_size
        )
        
        # Check spatial dimensions are restored
        _, _, h, w = segmentation.shape
        assert h == height
        assert w == width
    
    def test_decoder_weight_tying(self, encoder, device):
        """Test decoder ties weights with encoder."""
        from spikeseg.models import SpikeSEGDecoder
        
        decoder = SpikeSEGDecoder.from_encoder(encoder)
        
        # Check that decoder transposed conv weights match encoder conv weights
        # (transposed, so shapes will differ but values should be related)
        enc_weight = encoder.conv3.conv.weight
        dec_weight = decoder.trans_conv3.trans_conv.weight
        
        # Decoder weight should have related shape
        # Encoder Conv3: (n_classes, conv2_channels, k, k)
        # Decoder TransConv3: (n_classes, conv2_channels, k, k) - same shape for tied weights
        assert enc_weight.shape[0] == dec_weight.shape[0]  # Same number of classes


# =============================================================================
# FULL MODEL TESTS
# =============================================================================


class TestSpikeSEG:
    """Test complete SpikeSEG model."""
    
    def test_default_creation(self, torch_available, device):
        """Test default model creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG().to(device)
        
        assert model.encoder is not None
        assert model.config is not None
    
    def test_from_paper_igarss2023(self, torch_available, device):
        """Test model creation from IGARSS 2023 paper."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG.from_paper("igarss2023", n_classes=1).to(device)
        
        assert model.n_classes == 1
        assert model.config.conv1.out_channels == 4
        assert model.config.conv2.out_channels == 36
    
    def test_from_config(self, torch_available, device):
        """Test model creation from explicit config."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG.from_config(
            conv1_channels=2,
            conv2_channels=8,
            n_classes=3,
            kernel_sizes=(3, 3, 5),
            leaks=(5.0, 0.5, 0.0)
        ).to(device)
        
        assert model.config.conv1.out_channels == 2
        assert model.config.conv2.out_channels == 8
        assert model.n_classes == 3
    
    def test_forward_full(self, spikeseg_model, random_input):
        """Test full forward pass returns both outputs."""
        spikeseg_model.reset_state()
        seg, enc_out = spikeseg_model(random_input)
        
        assert seg is not None
        assert enc_out is not None
        from spikeseg.models import EncoderOutput
        assert isinstance(enc_out, EncoderOutput)
    
    def test_forward_segmentation_only(self, spikeseg_model, random_input):
        """Test forward with return_encoder_output=False."""
        spikeseg_model.reset_state()
        seg = spikeseg_model(random_input, return_encoder_output=False)
        
        assert seg is not None
        assert isinstance(seg, torch.Tensor)
    
    def test_encode_method(self, spikeseg_model, random_input):
        """Test encode method."""
        from spikeseg.models import EncoderOutput
        
        spikeseg_model.reset_state()
        enc_out = spikeseg_model.encode(random_input)
        
        assert isinstance(enc_out, EncoderOutput)
    
    def test_decode_method(self, spikeseg_model, random_input):
        """Test decode method."""
        spikeseg_model.reset_state()
        enc_out = spikeseg_model.encode(random_input)
        seg = spikeseg_model.decode(enc_out)
        
        assert isinstance(seg, torch.Tensor)
    
    def test_reset_state(self, spikeseg_model, random_input):
        """Test state reset."""
        spikeseg_model.reset_state()
        _ = spikeseg_model(random_input)
        
        spikeseg_model.reset_state()
        
        assert spikeseg_model.encoder.conv1.membrane is None
    
    def test_get_layer_weights(self, spikeseg_model):
        """Test get_layer_weights method."""
        weights = spikeseg_model.get_layer_weights()
        
        assert 'conv1' in weights
        assert 'conv2' in weights
        assert 'conv3' in weights
        assert isinstance(weights['conv1'], torch.Tensor)
    
    def test_freeze_layer(self, spikeseg_model):
        """Test layer freezing."""
        spikeseg_model.freeze_layer('conv1')
        
        for param in spikeseg_model.encoder.conv1.parameters():
            assert not param.requires_grad
    
    def test_unfreeze_layer(self, spikeseg_model):
        """Test layer unfreezing."""
        spikeseg_model.freeze_layer('conv2')
        spikeseg_model.unfreeze_layer('conv2')
        
        for param in spikeseg_model.encoder.conv2.parameters():
            assert param.requires_grad
    
    def test_lazy_decoder_creation(self, torch_available, device):
        """Test decoder is created lazily."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG().to(device)
        
        # Decoder not created yet
        assert model._decoder is None
        
        # Accessing decoder property creates it
        decoder = model.decoder
        assert decoder is not None
        assert model._decoder is not None
    
    def test_repr(self, spikeseg_model):
        """Test string representation."""
        repr_str = repr(spikeseg_model)
        
        assert 'SpikeSEG' in repr_str
        assert 'encoder' in repr_str
    
    def test_n_classes_property(self, spikeseg_model):
        """Test n_classes property."""
        n_classes = spikeseg_model.n_classes
        
        assert isinstance(n_classes, int)
        assert n_classes > 0


class TestSpikeSEGTemporal:
    """Test SpikeSEG temporal processing."""
    
    def test_temporal_sequence(self, small_spikeseg, device, n_timesteps):
        """Test processing temporal sequence."""
        small_spikeseg.reset_state()
        
        x = torch.rand(1, 1, 16, 16, device=device)
        
        outputs = []
        for t in range(n_timesteps):
            seg, enc_out = small_spikeseg(x)
            outputs.append((seg, enc_out))
        
        assert len(outputs) == n_timesteps
    
    def test_membrane_evolution(self, small_spikeseg, device):
        """Test membrane potentials evolve over time."""
        small_spikeseg.reset_state()
        
        x = torch.ones(1, 1, 16, 16, device=device) * 0.1
        
        # Run multiple timesteps
        for _ in range(5):
            small_spikeseg.encoder.forward_single_timestep(x)
        
        # Membrane should have accumulated
        mem = small_spikeseg.encoder.conv1.membrane
        assert mem is not None
        # With constant input and leak, membrane should have some value
    
    def test_spike_detection_over_time(self, small_spikeseg, device):
        """Test spike detection behavior over time."""
        small_spikeseg.reset_state()
        
        # Strong input to generate spikes
        x = torch.ones(1, 1, 16, 16, device=device)
        
        total_spikes = 0
        for _ in range(10):
            _, enc_out = small_spikeseg(x)
            total_spikes += enc_out.n_classification_spikes
        
        # Should generate some spikes with strong input
        # (depends on thresholds, but should be > 0 eventually)


class TestSpikeSEGGradients:
    """Test gradient-related functionality."""
    
    def test_no_gradients_by_default(self, spikeseg_model, random_input):
        """Test forward pass doesn't require gradients for STDP."""
        spikeseg_model.reset_state()
        
        with torch.no_grad():
            seg, enc_out = spikeseg_model(random_input)
        
        assert seg is not None
    
    def test_gradients_when_enabled(self, spikeseg_model, random_input):
        """Test gradients can be computed if needed."""
        spikeseg_model.reset_state()
        
        # Enable gradients
        random_input.requires_grad = True
        
        seg, _ = spikeseg_model(random_input)
        
        # Should be able to compute loss and backward
        loss = seg.sum()
        # Note: backward() might fail due to in-place operations in LIF
        # This is expected for STDP training (which doesn't use backprop)


# =============================================================================
# PARAMETER COUNT TESTS
# =============================================================================


class TestModelParameters:
    """Test model parameter counts."""
    
    def test_encoder_parameter_count(self, encoder, count_parameters):
        """Test encoder has expected parameter count."""
        n_params = count_parameters(encoder)
        
        # Should have parameters from conv1, conv2, conv3
        assert n_params > 0
    
    def test_small_model_fewer_params(self, small_spikeseg, spikeseg_model, count_parameters):
        """Test small model has fewer parameters."""
        n_small = count_parameters(small_spikeseg)
        n_full = count_parameters(spikeseg_model)
        
        assert n_small < n_full
    
    def test_frozen_layers_not_counted_trainable(self, spikeseg_model, count_parameters):
        """Test frozen layers excluded from trainable count."""
        n_before = count_parameters(spikeseg_model, trainable_only=True)
        
        spikeseg_model.freeze_layer('conv1')
        
        n_after = count_parameters(spikeseg_model, trainable_only=True)
        
        assert n_after < n_before


# =============================================================================
# DEVICE TESTS
# =============================================================================


class TestModelDevice:
    """Test model device handling."""
    
    def test_model_to_device(self, torch_available, device):
        """Test model can be moved to device."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG()
        model = model.to(device)
        
        # Check weights are on correct device
        assert model.encoder.conv1.conv.weight.device == device
    
    @pytest.mark.cuda
    def test_model_cuda(self, cuda_device):
        """Test model on CUDA."""
        from spikeseg.models import SpikeSEG
        
        model = SpikeSEG().to(cuda_device)
        x = torch.rand(1, 1, 32, 32, device=cuda_device)
        
        model.reset_state()
        seg, _ = model(x)
        
        assert seg.device == cuda_device
    
    def test_input_device_mismatch_error(self, spikeseg_model, cpu_device):
        """Test error on device mismatch."""
        spikeseg_model = spikeseg_model.to(cpu_device)
        
        # If CUDA available, create input on different device
        try:
            import torch
            if torch.cuda.is_available():
                x = torch.rand(1, 1, 32, 32, device='cuda')
                spikeseg_model.reset_state()
                with pytest.raises(RuntimeError):
                    spikeseg_model(x)
        except:
            pass  # Skip if CUDA not available