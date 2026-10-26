"""Pytest configuration: opt into JAX float64 before any JIT traces."""

from __future__ import annotations

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - CPU-only environments without JAX
    pass
