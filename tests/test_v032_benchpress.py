"""Verification tests for v0.3.2 improvements:
1. Sample-size-adaptive Fisher-z thresholds (`fisher_z_threshold`, `n_samples` / `alpha`)
2. Collider down-weighting & correlation-disconnected filtering in top-k conditioning
3. `dag_to_cpdag` utility & Erdos-Renyi (p=20, n=1000) Gaussian benchmark validation
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from causal_automata import (
    CausalAutomaton,
    __version__,
    dag_to_cpdag,
    fisher_z_threshold,
    numpy_step,
)
from causal_automata.bayesian import _union_neighbor_sets
from causal_automata.bayesian_synth import (
    correlation_skeleton_adjacency,
    edge_metrics,
    empirical_covariance,
    flat_elevations,
    skeleton_metrics,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from causal_automata.jax_core import (  # noqa: E402
    _pairwise_topk_Z,
    run as jax_run,
    step as jax_step,
)


def test_version_is_0_3_2() -> None:
    assert __version__ == "0.3.2"


def test_fisher_z_threshold_formula_and_calibration() -> None:
    """Task 1: fisher_z_threshold matches tanh(z_crit / sqrt(max(n - k - 3, 1)))."""
    thr_001 = fisher_z_threshold(n_samples=1000, k_cond=3, alpha=0.01)
    expected = math.tanh(2.5758293035489004 / math.sqrt(1000 - 3 - 3))
    assert thr_001 == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert 0.08 < thr_001 < 0.083

    # Smaller n_samples or stricter alpha increases threshold; larger n decreases it.
    assert fisher_z_threshold(500, 3, 0.01) > thr_001
    assert fisher_z_threshold(5000, 3, 0.01) < thr_001
    assert fisher_z_threshold(1000, 3, 0.05) < thr_001

    # Edge case: n_samples <= k_cond + 3 clamps denominator to 1.
    thr_tiny = fisher_z_threshold(n_samples=4, k_cond=3, alpha=0.01)
    assert thr_tiny == pytest.approx(math.tanh(2.5758293035489004), rel=1e-9)


def test_causal_automaton_auto_detects_n_samples_and_backward_compat() -> None:
    """Task 1: CausalAutomaton.fit(X) calibrates thresholds; cov-only keeps defaults."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1000, 5))
    ca = CausalAutomaton(ticks=5, k=3)
    ca.fit(X)
    expected_d = fisher_z_threshold(1000, k_cond=3, alpha=0.01)
    assert ca.thr_death_ == pytest.approx(expected_d, rel=1e-9)
    assert ca.thr_birth_ == pytest.approx(1.2 * expected_d, rel=1e-9)

    # Passing only cov without n_samples preserves legacy fixed defaults (0.05 / 0.15).
    ca_cov = CausalAutomaton(ticks=5, k=3)
    ca_cov.fit(cov=np.eye(5))
    assert ca_cov.thr_death_ == pytest.approx(0.05)
    assert ca_cov.thr_birth_ == pytest.approx(0.15)


def test_collider_downweighting_and_numpy_jax_conditioning_parity() -> None:
    """Task 2: sunk colliders (E[m] < min(E[i], E[j]) - 0.02) are deprioritized identically in NumPy & JAX."""
    # 4-node graph: pair (0, 1) has two candidate neighbors:
    #   node 2: upstream confounder (E[2] = 0.80, |R_{0,2}| + |R_{1,2}| = 0.70)
    #   node 3: sunk collider      (E[3] = 0.10, |R_{0,3}| + |R_{1,3}| = 1.10)
    # With k=1, raw |R| ranking would pick node 3 (collider), whereas v0.3.2
    # down-weights node 3 because E[3] < min(E[0], E[1]) - 0.02 and selects node 2.
    U = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    E = np.array([0.50, 0.50, 0.80, 0.10], dtype=np.float64)
    Cov = np.array(
        [
            [1.00, 0.20, 0.35, 0.55],
            [0.20, 1.00, 0.35, 0.55],
            [0.35, 0.35, 1.00, 0.10],
            [0.55, 0.55, 0.10, 1.00],
        ],
        dtype=np.float64,
    )
    I = np.array([0], dtype=np.intp)
    J = np.array([1], dtype=np.intp)

    Z_np, counts_np = _union_neighbor_sets(
        U, I, J, k=1, Cov=Cov, E=E, threshold_death=0.05
    )
    Z_jx, valid_jx = _pairwise_topk_Z(
        jnp.asarray(U),
        jnp.asarray(Cov),
        k=1,
        E=jnp.asarray(E),
        threshold_death=0.05,
    )
    assert counts_np[0] == 1
    assert int(Z_np[0, 0]) == 2
    assert bool(valid_jx[0, 1, 0]) is True
    assert int(Z_jx[0, 1, 0]) == 2


def test_dag_to_cpdag_chickering_meek_rules() -> None:
    """Task 3: dag_to_cpdag preserves v-structures, closes under R1-R3, leaves Markov-equivalent edges undirected."""
    # 1. Pure chain 0 -> 1 -> 2 has no v-structures: CPDAG is completely undirected.
    A_chain = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    C_chain = dag_to_cpdag(A_chain)
    assert np.array_equal(C_chain, C_chain.T)
    assert int(C_chain.sum()) == 4

    # 2. Unshielded collider 0 -> 1 <- 2 plus child 1 -> 3 (oriented by Meek R1).
    A_r1 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    C_r1 = dag_to_cpdag(A_r1)
    assert C_r1[0, 1] == 1.0 and C_r1[1, 0] == 0.0
    assert C_r1[2, 1] == 1.0 and C_r1[1, 2] == 0.0
    assert C_r1[1, 3] == 1.0 and C_r1[3, 1] == 0.0

    # 3. Meek R3 diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 0 -> 3 (1 and 2 non-adjacent).
    A_r3 = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    C_r3 = dag_to_cpdag(A_r3)
    # 1 -> 3 <- 2 is v-structure; 0 -> 3 is compelled by R3; 0 - 1 and 0 - 2 stay undirected.
    assert C_r3[1, 3] == 1.0 and C_r3[3, 1] == 0.0
    assert C_r3[2, 3] == 1.0 and C_r3[3, 2] == 0.0
    assert C_r3[0, 3] == 1.0 and C_r3[3, 0] == 0.0
    assert C_r3[0, 1] == 1.0 and C_r3[1, 0] == 1.0
    assert C_r3[0, 2] == 1.0 and C_r3[2, 0] == 1.0


def test_erdos_renyi_p20_n1000_gaussian_benchmark_and_parity() -> None:
    """Task 3: p=20, n=1000 random Erdos-Renyi Gaussian DAG achieves strong Skeleton & Pattern F1 with NumPy/JAX parity."""
    rng = np.random.default_rng(4)
    p = 20
    n_samples = 1000

    # Random Erdos-Renyi DAG with moderate-to-strong edge weights (including 0.10-0.35).
    A_true = np.triu((rng.random((p, p)) < 0.08).astype(np.float64), k=1)
    W = A_true * rng.uniform(0.10, 0.35, size=(p, p))
    X = np.zeros((n_samples, p), dtype=np.float64)
    for j in range(p):
        X[:, j] = X @ W[:, j] + rng.normal(0.0, 1.0, size=n_samples)

    # Fit via CausalAutomaton (auto-calibrates thr_death / thr_birth from n_samples=1000).
    ca_jax = CausalAutomaton(ticks=100, k=3, backend="jax", seed=4).fit(X)
    ca_np = CausalAutomaton(ticks=100, k=3, backend="numpy", seed=4).fit(X)

    assert ca_jax.adjacency_ is not None and ca_np.adjacency_ is not None
    assert ca_jax.cpdag_ is not None and ca_np.cpdag_ is not None

    # Exact NumPy <-> JAX adjacency parity after 100 ticks on p=20.
    assert np.array_equal(ca_jax.adjacency_ > 0, ca_np.adjacency_ > 0)
    np.testing.assert_allclose(ca_jax.elevation_, ca_np.elevation_, rtol=1e-6, atol=1e-6)

    sk_metrics = skeleton_metrics(ca_jax.adjacency_, A_true)
    dir_metrics = edge_metrics(ca_jax.adjacency_, A_true)
    cpdag_true = dag_to_cpdag(A_true)
    pattern_metrics = edge_metrics(ca_jax.cpdag_, cpdag_true)

    # Also verify Fisher-z calibration beats fixed legacy birth threshold (0.15) on recall.
    ca_fixed = CausalAutomaton(
        ticks=100, k=3, threshold_death=0.05, threshold_birth=0.15, backend="jax", seed=4
    ).fit(X)
    assert ca_fixed.adjacency_ is not None
    sk_fixed = skeleton_metrics(ca_fixed.adjacency_, A_true)
    assert sk_metrics.f1 > sk_fixed.f1
    assert sk_metrics.f1 >= 0.90
    assert dir_metrics.f1 >= 0.50
    assert pattern_metrics.f1 >= 0.75
