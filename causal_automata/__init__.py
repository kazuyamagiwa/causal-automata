"""causal-automata: JAX-accelerated cellular automaton engine for causal discovery."""

from causal_automata.bayesian import step as numpy_step
from causal_automata.bayesian_synth import default_conditioning_k
from causal_automata.jax_core import device_info, step, warmup

__all__ = [
    "step",
    "numpy_step",
    "device_info",
    "warmup",
    "default_conditioning_k",
    "__version__",
]

__version__ = "0.1.1"
