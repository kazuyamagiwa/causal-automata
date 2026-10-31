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
    assert __version__ == "0.3.4"


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


def test_berkson_birth_marginal_correlation_guard() -> None:
    """v0.3.3 Fix 1: marginally independent pairs (|R| < thr_death) are never born via collider conditioning."""
    # Unshielded collider 0 -> 2 <- 1 with elevated E[2] = 0.90 (so node 2 is NOT
    # down-weighted by elevation). Marginally R[0, 1] = 0.0 < thr_death, but
    # partial correlation |rho_{0,1 | 2}| = 0.36 / (1 - 0.36) = 0.5625 > thr_birth.
    Cov = np.array(
        [
            [1.00, 0.00, 0.60],
            [0.00, 1.00, 0.60],
            [0.60, 0.60, 1.00],
        ],
        dtype=np.float64,
    )
    A0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    E0 = np.array([0.60, 0.40, 0.90], dtype=np.float64)

    A_np, E_np = numpy_step(
        A0, E0, Cov, threshold_death=0.05, threshold_birth=0.15, k=1
    )
    A_jx, E_jx = jax_step(
        jnp.asarray(A0),
        jnp.asarray(E0),
        jnp.asarray(Cov),
        threshold_death=0.05,
        threshold_birth=0.15,
        k=1,
    )
    A_jx_np = np.asarray(A_jx)
    assert A_np[0, 1] == 0.0 and A_np[1, 0] == 0.0
    assert A_jx_np[0, 1] == 0.0 and A_jx_np[1, 0] == 0.0
    assert np.array_equal(A_np > 0, A_jx_np > 0)
    np.testing.assert_allclose(E_np, np.asarray(E_jx), rtol=1e-6, atol=1e-6)


def test_mutual_separator_and_progressive_death_screening() -> None:
    """v0.3.3 Fix 2: mutual separator filtering & progressive s=1..k death prune chain/fork shortcuts without killing true edges."""
    # 4-node graph:
    #   True chain 0 -> 1 -> 2, plus common child 3 of (0, 2): 0 -> 3 <- 2.
    #   For shortcut edge (0, 2):
    #     - Node 1 is the true mutual mediator (screens off 0 and 2 at s=1: rho_{0,2 | 1} = 0).
    #     - Node 3 is a collider for (0, 2) with slightly smaller mutual |R| than node 1,
    #       so Z_1 = [1] and Z_2 = [1, 3].
    #     - Conditioning on Z_2 = [1, 3] re-opens (0, 2) via collider 3 (|rho_{0,2 | 1,3}| > thr_death),
    #       whereas progressive s=1 checks Z_1 = [1] first (|rho_{0,2 | 1}| = 0 < thr_death) and kills (0, 2)!
    rng = np.random.default_rng(123)
    n_samples = 4000
    X0 = rng.normal(size=n_samples)
    X1 = 0.80 * X0 + rng.normal(scale=0.5, size=n_samples)
    X2 = 0.80 * X1 + rng.normal(scale=0.5, size=n_samples)
    X3 = 0.45 * X0 + 0.45 * X2 + rng.normal(scale=0.5, size=n_samples)
    X = np.column_stack([X0, X1, X2, X3])
    Cov = empirical_covariance(X)

    # Start with all forward edges including the false shortcut 0 -> 2;
    # keep E[3] high (0.55) so collider 3 is still selected as the 2nd mutual neighbor in Z_2.
    A0 = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    E0 = np.array([0.90, 0.70, 0.50, 0.55], dtype=np.float64)

    A_np, E_np = numpy_step(
        A0, E0, Cov, threshold_death=0.05, threshold_birth=0.15, k=2
    )
    A_jx, E_jx = jax_step(
        jnp.asarray(A0),
        jnp.asarray(E0),
        jnp.asarray(Cov),
        threshold_death=0.05,
        threshold_birth=0.15,
        k=2,
    )
    A_jx_np = np.asarray(A_jx)

    # Shortcut 0 -> 2 is pruned via s=1 mutual mediator {1}, while true edges 0 -> 1 and 1 -> 2 survive.
    assert A_np[0, 2] == 0.0 and A_jx_np[0, 2] == 0.0
    assert A_np[0, 1] == 1.0 and A_np[1, 2] == 1.0
    assert np.array_equal(A_np > 0, A_jx_np > 0)
    np.testing.assert_allclose(E_np, np.asarray(E_jx), rtol=1e-6, atol=1e-6)


def _count_vstructures(A: np.ndarray) -> set[tuple[int, int, int]]:
    n = A.shape[0]
    D = (A > 0).astype(bool)
    np.fill_diagonal(D, False)
    U = D | D.T
    v_set: set[tuple[int, int, int]] = set()
    for k in range(n):
        parents = np.flatnonzero(D[:, k])
        for idx_a in range(len(parents)):
            for idx_b in range(idx_a + 1, len(parents)):
                i, j = int(parents[idx_a]), int(parents[idx_b])
                if not U[i, j]:
                    v_set.add((min(i, j), k, max(i, j)))
    return v_set


def test_v034_er_p20_n1000_seed1_vstructures_and_zero_spurious_sinks() -> None:
    """v0.3.4: ER p=20, n=1000, seed=1 recovers 100% Skeleton & Pattern F1 with 0 spurious v-structures."""
    rng = np.random.default_rng(1)
    p, n_samples = 20, 1000
    A_true = np.triu((rng.random((p, p)) < 0.08).astype(np.float64), k=1)
    W = A_true * rng.uniform(0.10, 0.35, size=(p, p))
    X = np.zeros((n_samples, p), dtype=np.float64)
    for j in range(p):
        X[:, j] = X @ W[:, j] + rng.normal(0.0, 1.0, size=n_samples)

    ca_jax = CausalAutomaton(ticks=100, k=3, backend="jax", seed=1).fit(X)
    ca_np = CausalAutomaton(ticks=100, k=3, backend="numpy", seed=1).fit(X)
    assert ca_jax.adjacency_ is not None and ca_np.adjacency_ is not None
    assert ca_jax.cpdag_ is not None and ca_np.cpdag_ is not None
    assert ca_jax.elevation_ is not None and ca_np.elevation_ is not None

    # Exact NumPy <-> JAX parity
    assert np.array_equal(ca_jax.adjacency_ > 0, ca_np.adjacency_ > 0)
    np.testing.assert_allclose(ca_jax.elevation_, ca_np.elevation_, rtol=1e-6, atol=1e-6)

    sk = skeleton_metrics(ca_jax.adjacency_, A_true)
    cp_true = dag_to_cpdag(A_true)
    cp = edge_metrics(ca_jax.cpdag_, cp_true)
    v_true = _count_vstructures(A_true)
    v_pred = _count_vstructures(ca_jax.adjacency_)

    assert sk.fn == 0 and sk.fp == 0 and sk.f1 == pytest.approx(1.0)
    assert cp.f1 == pytest.approx(1.0)
    assert v_pred == v_true
    assert len(v_pred - v_true) == 0


def test_v034_4_to_13_survival_and_7_vstructures_motif() -> None:
    """v0.3.4: 20-node, 17-edge, 7-v-structure motif with 4->13->7, 4->9, 4->15 keeps 4->13 (17/17 TP, 0 FP, 0 FN) and 7/7 v-structures."""
    p, n_samples = 20, 1000
    A_true = np.zeros((p, p), dtype=np.float64)
    W = np.zeros((p, p), dtype=np.float64)
    edges_weights = [
        (4, 13, 0.35),
        (13, 7, 0.65),
        (4, 9, 0.65),
        (4, 15, 0.65),
        (0, 9, 0.50),
        (1, 15, 0.50),
        (2, 7, 0.50),
        (3, 10, 0.50),
        (5, 10, 0.50),
        (6, 12, 0.50),
        (8, 12, 0.50),
        (3, 11, 0.45),
        (6, 14, 0.45),
        (11, 16, 0.50),
        (14, 16, 0.50),
        (17, 19, 0.50),
        (18, 19, 0.50),
    ]
    for u, v, w in edges_weights:
        A_true[u, v] = 1.0
        W[u, v] = w

    rng = np.random.default_rng(5)
    topo = [0, 1, 2, 3, 4, 5, 6, 8, 17, 18, 10, 12, 11, 14, 13, 9, 15, 7, 16, 19]
    X = np.zeros((n_samples, p), dtype=np.float64)
    for j in topo:
        X[:, j] = X @ W[:, j] + rng.normal(0.0, 1.0, size=n_samples)

    ca_jax = CausalAutomaton(ticks=100, k=3, backend="jax", seed=1).fit(X)
    ca_np = CausalAutomaton(ticks=100, k=3, backend="numpy", seed=1).fit(X)
    assert ca_jax.adjacency_ is not None and ca_np.adjacency_ is not None
    assert ca_jax.cpdag_ is not None and ca_jax.elevation_ is not None

    assert np.array_equal(ca_jax.adjacency_ > 0, ca_np.adjacency_ > 0)
    np.testing.assert_allclose(ca_jax.elevation_, ca_np.elevation_, rtol=1e-6, atol=1e-6)

    sk = skeleton_metrics(ca_jax.adjacency_, A_true)
    cp_true = dag_to_cpdag(A_true)
    cp = edge_metrics(ca_jax.cpdag_, cp_true)
    v_true = _count_vstructures(A_true)
    v_pred = _count_vstructures(ca_jax.adjacency_)

    # 4-13 survives, 4-7 and 13-15 are pruned -> 17/17 TP, 0 FP, 0 FN (Skeleton F1 = 1.000)
    assert (ca_jax.adjacency_[4, 13] > 0) or (ca_jax.adjacency_[13, 4] > 0)
    assert ca_jax.adjacency_[4, 7] == 0.0 and ca_jax.adjacency_[7, 4] == 0.0
    assert ca_jax.adjacency_[13, 15] == 0.0 and ca_jax.adjacency_[15, 13] == 0.0
    assert sk.tp == 17 and sk.fp == 0 and sk.fn == 0
    assert sk.f1 == pytest.approx(1.0)
    assert cp.f1 == pytest.approx(1.0)
    assert len(v_true) == 7
    assert v_pred == v_true
    for i, k_col, j in v_true:
        assert ca_jax.elevation_[k_col] < min(
            ca_jax.elevation_[i], ca_jax.elevation_[j]
        )


