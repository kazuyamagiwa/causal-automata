"""causal-automata: JAX-accelerated cellular automaton engine for causal discovery."""

from causal_automata.bayesian import step as numpy_step
from causal_automata.bayesian_synth import default_conditioning_k
from causal_automata.jax_core import (
    binarize,
    detect_colliders,
    device_info,
    run,
    step,
    warmup,
)

__all__ = [
    "step",
    "run",
    "numpy_step",
    "device_info",
    "warmup",
    "binarize",
    "detect_colliders",
    "default_conditioning_k",
    "__version__",
]

__version__ = "0.2.0"
