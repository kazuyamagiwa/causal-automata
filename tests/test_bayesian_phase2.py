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


def test_run_phase2_recovers_seed0_dag() -> None:
    result = run_phase2(n=N, seed=0, ticks=100, backend="numpy")
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
    result = run_phase2(n=N, seed=seed, ticks=100, backend="numpy")
    assert result.metrics_final.f1 >= result.metrics_init.f1 - 1e-9
    assert result.skeleton_final.f1 >= 0.5
    assert result.metrics_final.tp >= 4
