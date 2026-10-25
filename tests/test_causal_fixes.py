"""Verification suite for jax_core causal-physics fixes (Tasks 1–5)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from causal_automata.bayesian_synth import (  # noqa: E402
    edge_metrics,
    skeleton_metrics,
    undirected_skeleton,
)
from causal_automata.jax_core import detect_colliders, run, step  # noqa: E402


def _shd(A_pred: np.ndarray, A_true: np.ndarray) -> int:
    """Structural Hamming Distance on CPDAG-style undirected + orientation.

    Counts missing / extra undirected edges plus orientation mismatches on
    shared skeleton edges.
    """
    pred = (A_pred > 0).astype(np.float64)
    true = (A_true > 0).astype(np.float64)
    np.fill_diagonal(pred, 0.0)
    np.fill_diagonal(true, 0.0)
    sk_pred = undirected_skeleton(pred)
    sk_true = undirected_skeleton(true)
    # Upper-tri skeleton disagreements (add / delete).
    sk_diff = np.triu(sk_pred != sk_true, k=1).sum()
    # Shared undirected edges with wrong / missing orientation.
    shared = np.triu((sk_pred > 0) & (sk_true > 0), k=1)
    ii, jj = np.nonzero(shared)
    orient = 0
    for i, j in zip(ii.tolist(), jj.tolist()):
        # Match if both have the same directed pattern on {i,j}.
        if pred[i, j] == true[i, j] and pred[j, i] == true[j, i]:
            continue
        orient += 1
    return int(sk_diff + orient)


def _chain_cov(a: float = 0.5) -> np.ndarray:
    """Unit-noise chain 0 → 1 → 2 with edge weight ``a`` (analytical Cov)."""
    # X0 ~ N(0,1); X1 = a X0 + e1; X2 = a X1 + e2; ei ~ N(0,1).
    v0 = 1.0
    v1 = a * a * v0 + 1.0
    v2 = a * a * v1 + 1.0
    c01 = a * v0
    c12 = a * v1
    c02 = a * c01
    return np.array(
        [
            [v0, c01, c02],
            [c01, v1, c12],
            [c02, c12, v2],
        ],
        dtype=np.float64,
    )


def test_run_static_ticks_and_k_compile() -> None:
    """Task 1: explicit ticks/k must not raise ConcretizationTypeError."""
    n = 4
    A0 = jnp.zeros((n, n), dtype=jnp.float64)
    E0 = jnp.linspace(1.0, 0.0, n, dtype=jnp.float64)
    Cov = jnp.eye(n, dtype=jnp.float64)
    Af, Ef = run(A0, E0, Cov, ticks=5, k=2)
    assert Af.shape == (n, n)
    assert Ef.shape == (n,)
    # Second call with different static args recompiles cleanly.
    Af2, Ef2 = run(A0, E0, Cov, ticks=3, k=1)
    assert Af2.shape == (n, n)
    assert Ef2.shape == (n,)


def test_root_stability_chain() -> None:
    """Task 3/5: true 3-node chain keeps E[0] > E[1] > E[2] and both edges."""
    A0 = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    E0 = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float64)
    Cov = jnp.asarray(_chain_cov(0.5))
    Af, Ef = run(
        A0,
        E0,
        Cov,
        ticks=100,
        k=1,
        threshold_death=0.05,
        threshold_birth=0.15,
    )
    Af_np = np.asarray(Af)
    Ef_np = np.asarray(Ef)
    assert Ef_np[0] > Ef_np[1] > Ef_np[2]
    assert Af_np[0, 1] == pytest.approx(1.0)
    assert Af_np[1, 2] == pytest.approx(1.0)


def test_collider_vs_fork_discrimination() -> None:
    """Task 2/5: explaining-away flags collider middle node, not fork hub."""
    a = b = 0.25
    thr_death = 0.10

    # Collider 0 → 1 ← 2 (unit variances; r_02 = 0).
    A_col = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    Cov_col = jnp.array(
        [
            [1.0, a, 0.0],
            [a, 1.0, b],
            [0.0, b, 1.0],
        ],
        dtype=jnp.float64,
    )

    # Fork 0 ← 1 → 2 (r_02 = a*b = 0.0625 < thr_death; screening-off).
    A_fork = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    Cov_fork = jnp.array(
        [
            [1.0, a, a * b],
            [a, 1.0, b],
            [a * b, b, 1.0],
        ],
        dtype=jnp.float64,
    )

    col_flags = np.asarray(detect_colliders(A_col, Cov_col, thr_death))
    fork_flags = np.asarray(detect_colliders(A_fork, Cov_fork, thr_death))

    assert col_flags[1]
    assert not fork_flags[1]


def test_node_permutation_invariance() -> None:
    """Task 5: random node permutation yields equivalent Skeleton F1 and SHD."""
    rng = np.random.default_rng(42)
    # Small identifiable chain + collider motif (N=4).
    A_true = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # Linear-Gaussian samples → empirical Cov (same seed family for both runs).
    n, T = 4, 4000
    eps = rng.normal(size=(T, n))
    X = np.zeros((T, n), dtype=np.float64)
    X[:, 0] = eps[:, 0]
    X[:, 1] = 0.6 * X[:, 0] + eps[:, 1]
    X[:, 2] = 0.6 * X[:, 1] + eps[:, 2]
    X[:, 3] = 0.6 * X[:, 1] + eps[:, 3]
    Cov = np.cov(X, rowvar=False)

    E0 = np.linspace(1.0, 0.0, n, dtype=np.float64)
    # Dense upper-tri init (no index-order privilege beyond elevation).
    A0 = np.triu(np.ones((n, n), dtype=np.float64), k=1)

    Af, _ = run(
        jnp.asarray(A0),
        jnp.asarray(E0),
        jnp.asarray(Cov),
        ticks=80,
        k=2,
        threshold_death=0.05,
        threshold_birth=0.12,
    )
    Af_np = np.asarray(Af)
    sk_f1 = skeleton_metrics(Af_np, A_true).f1
    shd0 = _shd(Af_np, A_true)

    perm = rng.permutation(n)
    P = np.eye(n, dtype=np.float64)[perm]
    A_true_p = P @ A_true @ P.T
    A0_p = P @ A0 @ P.T
    Cov_p = P @ Cov @ P.T
    E0_p = E0[perm]

    Af_p, _ = run(
        jnp.asarray(A0_p),
        jnp.asarray(E0_p),
        jnp.asarray(Cov_p),
        ticks=80,
        k=2,
        threshold_death=0.05,
        threshold_birth=0.12,
    )
    Af_p_np = np.asarray(Af_p)
    sk_f1_p = skeleton_metrics(Af_p_np, A_true_p).f1
    shd_p = _shd(Af_p_np, A_true_p)

    assert sk_f1_p == pytest.approx(sk_f1, abs=1e-9)
    assert shd_p == shd0


def test_step_tick_order_smoke() -> None:
    """Single step remains JIT-compatible after gravity→v_well→death→birth."""
    A = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    E = jnp.array([0.9, 0.5, 0.1], dtype=jnp.float64)
    Cov = jnp.asarray(_chain_cov(0.4))
    A1, E1 = step(A, E, Cov, k=1)
    assert A1.shape == (3, 3)
    assert E1.shape == (3,)
    assert float(jnp.diag(A1).sum()) == 0.0
