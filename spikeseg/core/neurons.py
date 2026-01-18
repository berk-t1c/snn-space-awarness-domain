"""
Spiking Neuron Models

This module implements the fundamental spiking neuron models used in SpikeSEG:
- IF (Integrate-and-Fire): Simple accumulator, no leak
- LIF (Leaky Integrate-and-Fire): Accumulator with decay

Paper References:
    - Kheradpisheh et al. 2018: Base neuron dynamics (Equation 1)
    - Kirkland et al. 2023 (IGARSS): Modified leak ratios for space awareness

Mathematical Foundation:
    The membrane potential V evolves as:
    
    ############################################################################
    Note to developer: The membrane potential can be defined in two ways:
    ############################################################################

    Subtractive (IGARSS 2023):
        V(t) = V(t-1) + I(t) - λ
        
    Multiplicative (Kheradpisheh, snnTorch):
        V(t) = β * V(t-1) + I(t)
    
    You can choose the mode of the membrane potential by the parameter "leak_mode" in the constructor of the LIFNeuron class

    Where:
        V(t)   = membrane potential at time t
        I(t)   = input current (weighted sum of incoming spikes)
        λ      = leak term (subtractive mode)
        β      = decay factor (multiplicative mode), typically 0 < β < 1
    
    When V(t) ≥ threshold:
        - Neuron emits a spike (output = 1)
        - Membrane potential resets to 0

Example:
    >>> neuron = LIFNeuron(threshold=10.0, leak_factor=0.9)
    >>> membrane = torch.zeros(1, 4, 32, 32)  # batch, channels, height, width
    >>> input_current = torch.randn(1, 4, 32, 32)
    >>> spikes, new_membrane = neuron(input_current, membrane)
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

# Type alias for leak mode
LeakMode = Literal["subtractive", "multiplicative"]

# =============================================================================
# SPIKE FUNCTION
# =============================================================================
# This is the core operation: convert membrane potential to spikes.
# It's non-differentiable (step function), but we don't need gradients
# because we use STDP, not backpropagation.


def spike_function(membrane: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Generate spikes from membrane potential.
    
    This is a simple threshold comparison:
        spike = 1 if membrane >= threshold else 0
    
    Args:
        membrane: Membrane potential tensor, any shape.
        threshold: Firing threshold (positive float).
    
    Returns:
        Binary spike tensor (same shape as membrane).
        Values are 0.0 or 1.0 (float for compatibility with subsequent ops).
    
    Example:
        >>> membrane = torch.tensor([5.0, 10.0, 15.0])
        >>> spikes = spike_function(membrane, threshold=10.0)
        >>> spikes
        tensor([0., 1., 1.])
    """
    # ge = "greater than or equal"
    # .float() converts True/False to 1.0/0.0
    return (membrane >= threshold).float()


# =============================================================================
# BASE NEURON CLASS
# =============================================================================
# This defines the interface that all neuron types must follow.
# Using a base class ensures consistency and allows code reuse.

class BaseNeuron(nn.Module):
    """
    Abstract base class for spiking neurons.
    
    All neuron types inherit from this class and must implement:
        - forward(): Process one timestep of input
        - reset_state(): Clear membrane potential
    
    Attributes:
        threshold: Membrane potential threshold for firing.
    
    Why inherit from nn.Module?
        - Integrates with PyTorch ecosystem (model.to('cuda'), model.parameters())
        - Enables proper serialization (torch.save/load)
        - Provides hooks for debugging and visualization
    """
    
    def __init__(self, threshold: float = 1.0) -> None:
        """
        Initialize base neuron.
        
        Args:
            threshold: Firing threshold. When membrane >= threshold, neuron spikes.
                       Higher threshold = harder to fire = more selective.
        
        Raises:
            TypeError: If threshold is not a number.
            ValueError: If threshold is not positive.
        """
        super().__init__()  # Always call parent __init__ for nn.Module
        
        # Validate threshold
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        
        # Store threshold as a buffer, not a parameter
        # Buffer: saved with model, moved with .to(device), but NOT trained
        # Parameter: same as buffer, but IS trained (we don't want that here)
        self.register_buffer("threshold", torch.tensor(threshold))
    
    def _validate_inputs(
        self,
        input_current: torch.Tensor,
        membrane: torch.Tensor
    ) -> None:
        """
        Validate input tensors for forward pass.
        
        Call this at the beginning of forward() in subclasses.
        
        Args:
            input_current: Incoming current tensor.
            membrane: Current membrane potential tensor.
        
        Raises:
            ValueError: If shapes don't match or devices differ.
        """
        # Shape check
        if input_current.shape != membrane.shape:
            raise ValueError(
                f"Shape mismatch: input_current {input_current.shape} != "
                f"membrane {membrane.shape}"
            )
        
        # Device check
        if input_current.device != membrane.device:
            raise ValueError(
                f"Device mismatch: input_current on {input_current.device}, "
                f"membrane on {membrane.device}"
            )
    
    def forward(
        self,
        input_current: torch.Tensor,
        membrane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one timestep.

        Args:
            input_current: Incoming current (weighted sum of input spikes).
                          Shape: (batch, channels, height, width) or (batch, features)
            membrane: Current membrane potential (same shape as input_current).

        Returns:
            Tuple of (spikes, new_membrane, pre_reset_membrane):
                - spikes: Binary tensor indicating which neurons fired
                - new_membrane: Updated membrane potential after reset
                - pre_reset_membrane: Membrane before reset (for WTA tie-breaking)

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def reset_state(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Create a fresh membrane potential tensor (all zeros).
        
        Call this at the start of processing a new sample.
        
        Args:
            shape: Desired shape for membrane potential.
            device: Device to create tensor on (CPU or CUDA).
        
        Returns:
            Zero-initialized membrane potential tensor.
        
        Example:
            >>> neuron = LIFNeuron(threshold=10.0)
            >>> membrane = neuron.reset_state((1, 4, 32, 32), torch.device('cpu'))
            >>> membrane.shape
            torch.Size([1, 4, 32, 32])
        """
        return torch.zeros(shape, device=device)


# =============================================================================
# INTEGRATE-AND-FIRE (IF) NEURON
# =============================================================================
# The simplest spiking neuron: just accumulate input until threshold.
# No leak means information is preserved indefinitely.
# Used in Conv3 layer (IGARSS 2023) where λ = 0.


class IFNeuron(BaseNeuron):
    """
    Integrate-and-Fire neuron (no leak).
    
    Dynamics:
        V(t) = V(t-1) + I(t)
        
        if V(t) >= threshold:
            spike = 1
            V(t) = 0  (reset)
        else:
            spike = 0
    
    Use Case:
        - Final classification layer (Conv3 in IGARSS 2023)
        - When you want full temporal integration
        - λ = 0 means no decay
    
    Paper Reference:
        Kirkland et al. 2023: "using a 7x7 final classification kernel with no leakage"
    
    Example:
        >>> neuron = IFNeuron(threshold=2.0)
        >>> membrane = torch.zeros(1, 1)  # Start at 0
        >>> 
        >>> # First input: membrane goes to 1.5
        >>> spikes, membrane = neuron(torch.tensor([[1.5]]), membrane)
        >>> print(f"Spikes: {spikes}, Membrane: {membrane}")
        Spikes: tensor([[0.]]), Membrane: tensor([[1.5000]])
        >>> 
        >>> # Second input: membrane goes to 2.5 >= 2.0, so it fires!
        >>> spikes, membrane = neuron(torch.tensor([[1.0]]), membrane)
        >>> print(f"Spikes: {spikes}, Membrane: {membrane}")
        Spikes: tensor([[1.]]), Membrane: tensor([[0.]])
    """
    
    def __init__(self, threshold: float = 1.0) -> None:
        """
        Initialize IF neuron.
        
        Args:
            threshold: Firing threshold. Default 1.0.
        """
        super().__init__(threshold=threshold)
    
    def forward(
        self,
        input_current: torch.Tensor,
        membrane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one timestep of IF neuron dynamics.

        Args:
            input_current: Incoming current. Shape: (batch, channels, ...)
            membrane: Current membrane potential. Same shape as input_current.

        Returns:
            Tuple of (spikes, new_membrane, pre_reset_membrane):
                - spikes: Binary tensor indicating which neurons fired
                - new_membrane: Membrane after reset
                - pre_reset_membrane: Membrane before reset (for WTA tie-breaking)

        Raises:
            ValueError: If input shapes or devices don't match.
        """
        # Validate inputs
        self._validate_inputs(input_current, membrane)

        # Step 1: Integrate (accumulate input)
        membrane = membrane + input_current

        # Store pre-reset membrane for WTA tie-breaking (Kheradpisheh 2018)
        pre_reset_membrane = membrane.clone()

        # Step 2: Check threshold and generate spikes
        spikes = spike_function(membrane, self.threshold.item())

        # Step 3: Reset neurons that fired
        # Where spikes=1, set membrane to 0
        # Where spikes=0, keep membrane as is
        # This is done with: membrane * (1 - spikes)
        membrane = membrane * (1.0 - spikes)

        return spikes, membrane, pre_reset_membrane
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"IFNeuron(threshold={self.threshold.item():.2f})"


# =============================================================================
# LEAKY INTEGRATE-AND-FIRE (LIF) NEURON
# =============================================================================
# Adds a leak term that causes membrane to decay toward 0.
# This creates a "memory window" - old inputs fade away.
# Different leak values create different temporal behaviors.
# Supports both subtractive (IGARSS 2023) and multiplicative (Kheradpisheh) modes.


class LIFNeuron(BaseNeuron):
    """
    Leaky Integrate-and-Fire neuron with configurable leak mode.
    
    Supports two leak formulations:
    
    1. Subtractive (IGARSS 2023):
        V(t) = V(t-1) + I(t) - λ
        - Constant decay (loses same amount always)
        - More aggressive at low membrane values
        - Default for this implementation
    
    2. Multiplicative (Kheradpisheh, snnTorch):
        V(t) = β * V(t-1) + I(t)
        - Proportional decay (loses % of current value)
        - More "biological" (exponential decay)
        - β = 1 - leak_factor
    
    Comparison with membrane = 100, leak_factor = 0.1:
        - Subtractive (λ=10): 100 - 10 = 90
        - Multiplicative (β=0.9): 100 * 0.9 = 90
        
    Comparison with membrane = 10, leak_factor = 0.1:
        - Subtractive (λ=10): 10 - 10 = 0 (clamped)
        - Multiplicative (β=0.9): 10 * 0.9 = 9
    
    Paper Reference:
        Kirkland et al. 2023 (IGARSS):
        "setting the leak, λ, to 90% and 10% of the neuron threshold 
         in layers 1 and 2 respectively"
        
        - Layer 1: λ = 0.9 * threshold → Detects fast-moving satellites
        - Layer 2: λ = 0.1 * threshold → Accumulates persistent features
        - Layer 3: λ = 0 (use IFNeuron) → Full integration for classification
    
    Example:
        >>> # Layer 1 config from IGARSS paper (subtractive, default)
        >>> neuron = LIFNeuron(threshold=10.0, leak_factor=0.9)
        >>> 
        >>> # Alternative: multiplicative leak (Kheradpisheh style)
        >>> neuron_mult = LIFNeuron(threshold=10.0, leak_factor=0.1, leak_mode="multiplicative")
    """
    
    def __init__(
        self, 
        threshold: float = 1.0, 
        leak_factor: float = 0.0,
        leak_mode: LeakMode = "subtractive"
    ) -> None:
        """
        Initialize LIF neuron.
        
        Args:
            threshold: Firing threshold.
            leak_factor: Leak as fraction of threshold (0.0 to 1.0).
                        
                        For subtractive mode:
                            Actual leak λ = leak_factor * threshold
                            
                        For multiplicative mode:
                            Decay factor β = 1 - leak_factor
                            
                        Examples from IGARSS 2023 (subtractive):
                            - Layer 1: leak_factor=0.9 (90% of threshold)
                            - Layer 2: leak_factor=0.1 (10% of threshold)
                            - Layer 3: leak_factor=0.0 (no leak, use IFNeuron)
                            
            leak_mode: Either "subtractive" or "multiplicative".
                      - "subtractive": V = V + I - λ (IGARSS 2023)
                      - "multiplicative": V = β*V + I (Kheradpisheh, snnTorch)
        
        Raises:
            TypeError: If threshold or leak_factor is not a number.
            ValueError: If threshold is not positive, leak_factor not in [0,1],
                       or leak_mode is invalid.
        """
        # Call parent init (validates threshold)
        super().__init__(threshold=threshold)
        
        # Validate leak_factor
        if not isinstance(leak_factor, (int, float)):
            raise TypeError(f"leak_factor must be a number, got {type(leak_factor).__name__}")
        if not 0.0 <= leak_factor <= 1.0:
            raise ValueError(f"leak_factor must be in [0, 1], got {leak_factor}")
        
        # Validate leak_mode
        if leak_mode not in ("subtractive", "multiplicative"):
            raise ValueError(
                f"leak_mode must be 'subtractive' or 'multiplicative', "
                f"got '{leak_mode}'"
            )
        
        self.leak_mode = leak_mode
        self.leak_factor = leak_factor
        
        # Calculate and store leak/decay values
        if leak_mode == "subtractive":
            # λ = leak_factor * threshold
            leak_value = leak_factor * threshold
            self.register_buffer("leak", torch.tensor(leak_value))
            self.register_buffer("beta", torch.tensor(1.0))  # Not used, but for consistency
        else:  # multiplicative
            # β = 1 - leak_factor
            beta_value = 1.0 - leak_factor
            self.register_buffer("beta", torch.tensor(beta_value))
            self.register_buffer("leak", torch.tensor(0.0))  # Not used, but for consistency
    
    def forward(
        self,
        input_current: torch.Tensor,
        membrane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one timestep of LIF neuron dynamics.

        Args:
            input_current: Incoming current. Shape: (batch, channels, ...)
            membrane: Current membrane potential. Same shape as input_current.

        Returns:
            Tuple of (spikes, new_membrane, pre_reset_membrane):
                - spikes: Binary tensor indicating which neurons fired
                - new_membrane: Membrane after reset
                - pre_reset_membrane: Membrane before reset (for WTA tie-breaking)

        Raises:
            ValueError: If input shapes or devices don't match.
        """
        # Validate inputs
        self._validate_inputs(input_current, membrane)

        # Step 1: Apply leak and integrate input
        if self.leak_mode == "subtractive":
            # V(t) = V(t-1) + I(t) - λ
            membrane = membrane + input_current - self.leak
        else:  # multiplicative
            # V(t) = β * V(t-1) + I(t)
            membrane = self.beta * membrane + input_current

        # Step 2: Clamp to non-negative
        # Biological neurons can't have negative potential (relative to resting)
        # This also prevents numerical instability
        membrane = torch.clamp(membrane, min=0.0)

        # Store pre-reset membrane for WTA tie-breaking (Kheradpisheh 2018)
        pre_reset_membrane = membrane.clone()

        # Step 3: Check threshold and generate spikes
        spikes = spike_function(membrane, self.threshold.item())

        # Step 4: Reset neurons that fired
        membrane = membrane * (1.0 - spikes)

        return spikes, membrane, pre_reset_membrane
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.leak_mode == "subtractive":
            return (
                f"LIFNeuron("
                f"threshold={self.threshold.item():.2f}, "
                f"leak_factor={self.leak_factor:.2f}, "
                f"leak={self.leak.item():.2f}, "
                f"mode=subtractive)"
            )
        else:
            return (
                f"LIFNeuron("
                f"threshold={self.threshold.item():.2f}, "
                f"leak_factor={self.leak_factor:.2f}, "
                f"beta={self.beta.item():.2f}, "
                f"mode=multiplicative)"
            )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
# A factory function creates objects based on parameters.
# This allows configuration-driven code: just specify neuron type in YAML.


def create_neuron(
    neuron_type: str,
    threshold: float,
    leak_factor: float = 0.0,
    leak_mode: LeakMode = "subtractive"
) -> BaseNeuron:
    """
    Create a neuron instance based on type string.
    
    This factory function allows neuron creation from config files:
    
        config.yaml:
            conv1:
                neuron_type: "lif"
                threshold: 10.0
                leak_factor: 0.9
                leak_mode: "subtractive"
    
    Args:
        neuron_type: Either "if" or "lif" (case-insensitive).
        threshold: Firing threshold.
        leak_factor: Leak factor (only used for LIF).
        leak_mode: Leak mode (only used for LIF). 
                   Either "subtractive" or "multiplicative".
    
    Returns:
        Neuron instance (IFNeuron or LIFNeuron).
    
    Raises:
        ValueError: If neuron_type is not recognized.
    
    Example:
        >>> neuron = create_neuron("lif", threshold=10.0, leak_factor=0.9)
        >>> type(neuron)
        <class 'spikeseg.core.neurons.LIFNeuron'>
        
        >>> neuron_mult = create_neuron(
        ...     "lif", threshold=10.0, leak_factor=0.1, leak_mode="multiplicative"
        ... )
        >>> neuron_mult.leak_mode
        'multiplicative'
    """
    neuron_type = neuron_type.lower().strip()
    
    if neuron_type == "if":
        return IFNeuron(threshold=threshold)
    elif neuron_type == "lif":
        return LIFNeuron(
            threshold=threshold, 
            leak_factor=leak_factor,
            leak_mode=leak_mode
        )
    else:
        raise ValueError(
            f"Unknown neuron type: '{neuron_type}'. "
            f"Expected 'if' or 'lif'."
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================
# __all__ defines what gets imported with "from spikeseg.core.neurons import *"
# It's also documentation: these are the public parts of this module.

__all__ = [
    "spike_function",
    "BaseNeuron",
    "IFNeuron", 
    "LIFNeuron",
    "create_neuron",
    "LeakMode",
]
