"""
Learning algorithms for Spiking Neural Networks.

This module provides unsupervised learning rules for training SNNs:

- **STDP** (Spike-Timing Dependent Plasticity): Biologically-inspired learning
  where synaptic strength changes based on relative spike timing.
  
- **WTA** (Winner-Take-All): Lateral inhibition mechanism that enforces
  competition between neurons, ensuring feature diversity.

Learning Pipeline:
    1. Forward pass: Input → Spikes
    2. WTA: Select winning neurons
    3. STDP: Update winner's weights based on spike timing
    4. Check convergence: Stop when weights stabilize

Paper References:
    - Kheradpisheh et al. 2018: STDP rule, WTA mechanism, convergence metric
    - Kirkland et al. 2023 (IGARSS): Modified learning rates for space awareness

Example:
    >>> from spikeseg.learning import STDPLearner, STDPConfig, WTAInhibition, WTAConfig
    >>> 
    >>> # Create STDP learner with paper parameters
    >>> stdp_config = STDPConfig.from_paper("kheradpisheh2018")
    >>> learner = STDPLearner(stdp_config)
    >>> 
    >>> # Create WTA for competition
    >>> wta_config = WTAConfig(mode="both", enable_homeostasis=True)
    >>> wta = WTAInhibition(wta_config, n_channels=36, spatial_shape=(16, 16))
    >>> 
    >>> # Training loop
    >>> for spikes, membrane in forward_pass(images):
    ...     filtered_spikes, new_membrane = wta(spikes, membrane)
    ...     winner_mask = wta.get_winner_mask()
    ...     # Apply STDP to winners...
"""

# =============================================================================
# STDP: Spike-Timing Dependent Plasticity
# =============================================================================

from .stdp import (
    # Exceptions
    STDPError,
    STDPConfigError,
    STDPRuntimeError,
    # Configuration
    STDPVariant,
    STDPConfig,
    # Weight initialization
    initialize_weights,
    # Convergence
    compute_convergence_metric,
    has_converged,
    # Spike timing
    get_first_spike_times,
    extract_receptive_field_times,
    # STDP update
    compute_stdp_update,
    # Main learner
    STDPStats,
    STDPLearner,
    # WTA utilities (in STDP module for convenience)
    find_wta_winner,
    apply_lateral_inhibition,
)

# =============================================================================
# WTA: Winner-Take-All Lateral Inhibition
# =============================================================================

from .wta import (
    # Exceptions
    WTAError,
    WTAConfigError,
    WTARuntimeError,
    # Enums
    WTAMode,
    # Configuration
    WTAConfig,
    # Functions
    wta_global_membrane,
    wta_local_membrane,
    wta_by_membrane,
    # Classes
    AdaptiveThreshold,
    WTAInhibition,
    ConvergenceTracker,
)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # =========================================================================
    # STDP Components
    # =========================================================================
    # Exceptions
    "STDPError",
    "STDPConfigError",
    "STDPRuntimeError",
    # Configuration
    "STDPVariant",
    "STDPConfig",
    # Weight initialization
    "initialize_weights",
    # Convergence
    "compute_convergence_metric",
    "has_converged",
    # Spike timing
    "get_first_spike_times",
    "extract_receptive_field_times",
    # STDP update
    "compute_stdp_update",
    # Main learner
    "STDPStats",
    "STDPLearner",
    # WTA utilities
    "find_wta_winner",
    "apply_lateral_inhibition",
    
    # =========================================================================
    # WTA Components
    # =========================================================================
    # Exceptions
    "WTAError",
    "WTAConfigError",
    "WTARuntimeError",
    # Enums
    "WTAMode",
    # Configuration
    "WTAConfig",
    # Functions
    "wta_global_membrane",
    "wta_local_membrane",
    "wta_by_membrane",
    # Classes
    "AdaptiveThreshold",
    "WTAInhibition",
    "ConvergenceTracker",
]

