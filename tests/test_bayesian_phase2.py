"""Synthetic Phase-2 harness tests (N=10 linear-Gaussian DAG)."""

from __future__ import annotations

import numpy as np
import pytest

from causal_automata.bayesian_synth import (
    N,
    TRUTH_EDGES,
    prepare_synthetic_bundle,
    run_phase2,
    truth_adjacency,
    undirected_skeleton,
)


def test_phase2_truth_motif_is_upper_triangular_dag() -> None:
    A = truth_adjacency(N, edges=TRUTH_EDGES)
    assert A.shape == (N, N)
    assert np.all(np.diag(A) == 0)
    assert np.all(np.tril(A) == 0)
    assert int(A.sum()) == len(TRUTH_EDGES)
    for i, j in TRUTH_EDGES:
        assert A[i, j] == 1.0


def test_prepare_synthetic_bundle_shapes_and_reproducibility() -> None:
    a = prepare_synthetic_bundle(n=N, seed=0)
    b = prepare_synthetic_bundle(n=N, seed=0)
    assert a.X.shape[1] == N
    assert a.Cov.shape == (N, N)
    assert a.A_true.shape == (N, N)
    assert a.A_init.shape == (N, N)
    assert a.E_init.shape == (N,)
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.Cov, b.Cov)
    np.testing.assert_array_equal(a.A_true, b.A_true)
    np.testing.assert_array_equal(a.A_init, b.A_init)
    np.testing.assert_allclose(a.E_init, b.E_init)


def test_corr_flat_init_is_symmetric_and_near_flat() -> None:
    """L4: default init has no upper-tri / index-order bias."""
    bundle = prepare_synthetic_bundle(n=N, seed=0, permute_nodes=True)
    assert np.allclose(bundle.A_init, bundle.A_init.T)
    assert np.all(np.diag(bundle.A_init) == 0.0)
    assert float(np.max(np.abs(bundle.E_init - 0.5))) < 1e-3


def test_upper_random_init_is_strictly_upper_triangular() -> None:
    bundle = prepare_synthetic_bundle(
        n=N, seed=0, permute_nodes=True, init_mode="upper_random"
    )
    assert int(np.tril(bundle.A_init, k=0).sum()) == 0
    assert int(np.triu(bundle.A_init, k=1).sum()) > 0


def test_run_phase2_recovers_seed0_dag() -> None:
    # Unpermuted motif + legacy upper-tri init: historical Phase-2 contract.
    result = run_phase2(
        n=N,
        seed=0,
        ticks=100,
        backend="numpy",
        permute_nodes=False,
        init_mode="upper_random",
    )
    assert result.backend == "numpy"
    assert result.metrics_final.true_edges == len(TRUTH_EDGES)
    # Seed 0 recovers all true edges with high directed / skeleton F1.
    assert result.metrics_final.tp == len(TRUTH_EDGES)
    assert result.metrics_final.fn == 0
    assert result.metrics_final.f1 >= 0.9
    assert result.skeleton_final.f1 >= 0.9
    assert result.metrics_final.f1 >= result.metrics_init.f1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_phase2_improves_or_holds_directed_f1(seed: int) -> None:
    result = run_phase2(
        n=N,
        seed=seed,
        ticks=100,
        backend="numpy",
        permute_nodes=False,
        init_mode="upper_random",
    )
    assert result.metrics_final.f1 >= result.metrics_init.f1 - 1e-9
    assert result.skeleton_final.f1 >= 0.5
    assert result.metrics_final.tp >= 4


def test_l4_corr_flat_closes_permutation_orientation_gap() -> None:
    """L4: corr_flat keeps Directed F1 near Skeleton F1 under node permute.

    Legacy upper_random A0 locks gravity into ``i < j`` on tick 1, so
    Directed F1 collapses on a random labeling while Skeleton F1 stays high.
    """
    legacy = run_phase2(
        n=N,
        seed=0,
        ticks=100,
        backend="numpy",
        permute_nodes=True,
        init_mode="upper_random",
    )
    fixed = run_phase2(
        n=N,
        seed=0,
        ticks=100,
        backend="numpy",
        permute_nodes=True,
        init_mode="corr_flat",
    )
    # Gap is orientation-dominated under legacy init.
    assert legacy.skeleton_final.f1 - legacy.metrics_final.f1 >= 0.20
    assert legacy.metrics_final.f1 < 0.60
    # corr_flat recovers directed edges close to the undirected skeleton.
    assert fixed.metrics_final.f1 >= 0.70
    assert fixed.metrics_final.f1 >= fixed.skeleton_final.f1 - 1e-9
    assert undirected_skeleton(fixed.A_init).sum() == fixed.A_init.sum()
