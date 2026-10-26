"""NumPy ↔ JAX physics parity and Phase-2 recovery after audit fixes."""

from __future__ import annotations

import numpy as np
import pytest

from causal_automata.bayesian_synth import prepare_synthetic_bundle, run_phase2

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from causal_automata.bayesian import step as numpy_step  # noqa: E402
from causal_automata.jax_core import run, step as jax_step  # noqa: E402


def test_numpy_jax_single_tick_close() -> None:
    """One tick of unified physics should agree across backends."""
    bundle = prepare_synthetic_bundle(n=10, seed=0, permute_nodes=False)
    kwargs = dict(
        learning_rate=0.1,
        threshold_death=0.05,
        threshold_birth=0.10,
        k=3,
    )
    An, En = numpy_step(
        bundle.A_init.copy(), bundle.E_init.copy(), bundle.Cov, **kwargs
    )
    Aj, Ej = jax_step(
        jnp.asarray(bundle.A_init),
        jnp.asarray(bundle.E_init),
        jnp.asarray(bundle.Cov),
        **kwargs,
    )
    Aj, Ej = np.asarray(Aj), np.asarray(Ej)
    assert np.array_equal(An > 0, Aj > 0)
    np.testing.assert_allclose(En, Ej, rtol=1e-6, atol=1e-6)


def test_phase2_numpy_and_jax_recover_high_f1() -> None:
    """Audit H1: both backends recover Directed F1 ≥ 0.90 on N=10 seed=0.

    Uses ``permute_nodes=False`` and legacy ``upper_random`` init to match
    the audit SEM (index-ordered motif + upper-tri A0). Default synth
    bundles still permute nodes and use ``corr_flat`` init (L4).
    """
    r_np = run_phase2(
        n=10,
        seed=0,
        ticks=100,
        backend="numpy",
        permute_nodes=False,
        init_mode="upper_random",
    )
    r_jx = run_phase2(
        n=10,
        seed=0,
        ticks=100,
        backend="jax",
        permute_nodes=False,
        init_mode="upper_random",
    )
    assert r_np.metrics_final.f1 >= 0.90
    assert r_jx.metrics_final.f1 >= 0.90
    assert abs(r_np.metrics_final.f1 - r_jx.metrics_final.f1) <= 1e-9
    assert abs(r_np.skeleton_final.f1 - r_jx.skeleton_final.f1) <= 1e-9
    assert np.array_equal(r_np.A_final > 0, r_jx.A_final > 0)


def test_birth_tiebreak_breaks_ceiling_deadlock() -> None:
    """Audit H2: correlated nodes clipped at E=1 still birth a downhill edge."""
    A = jnp.zeros((3, 3))
    E = jnp.array([1.0, 1.0, 0.0])
    Cov = jnp.array(
        [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    Af, _ = run(
        A, E, Cov, ticks=20, k=1, threshold_birth=0.15, threshold_death=0.05
    )
    Af = np.asarray(Af)
    assert int((Af > 0).sum()) >= 1
    assert Af[1, 0] == pytest.approx(1.0) or Af[0, 1] == pytest.approx(1.0)


def test_local_margin_avoids_ceiling_saturation() -> None:
    """Audit H3: random-init N=10 run should not pin many nodes at E≈1."""
    r = run_phase2(
        n=10,
        seed=0,
        ticks=100,
        backend="jax",
        permute_nodes=False,
        init_mode="upper_random",
    )
    assert float(np.max(r.E_final)) < 0.99
    assert int(np.sum(r.E_final >= 0.999)) == 0
