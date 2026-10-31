"""causal-automata: JAX-accelerated cellular automaton engine for causal discovery."""

from causal_automata.bayesian import (
    dag_to_cpdag,
    fisher_z_threshold,
    step as numpy_step,
)
from causal_automata.jax_core import (
    CausalAutomaton,
    binarize,
    detect_colliders,
    device_info,
    enable_x64,
    run,
    step,
    warmup,
)


def default_conditioning_k(n: int) -> int:
    """Scale-aware default conditioning-set size (3 if n <= 50 else 5)."""
    return 3 if n <= 50 else 5


__all__ = [
    "CausalAutomaton",
    "step",
    "run",
    "numpy_step",
    "fisher_z_threshold",
    "dag_to_cpdag",
    "device_info",
    "enable_x64",
    "warmup",
    "binarize",
    "detect_colliders",
    "default_conditioning_k",
    "__version__",
]

__version__ = "0.3.3"
