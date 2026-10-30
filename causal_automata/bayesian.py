"""Bayesian CA update: DAG adjacency evolves via partial-correlation CMI.

NumPy reference implementation of the same physics as
:mod:`causal_automata.jax_core`. Each tick applies

    gravity → v_well → death → birth

with local-margin elevation drift (no hard 1.0/0.0 anchors), Cov-equivariant
bound-aware elevation tie-breaks on birth / final projection, ``|R|``-ranked
conditioning sets (no raw-elevation mask), and a multiplicative V-well on
shared neighbors of marginally independent pairs.
"""

from __future__ import annotations

import math

import numpy as np

_RIDGE = 1e-7
_STD_FLOOR = 1e-7
_SCALE_FLOOR = 1e-7
_EXPLAIN_AWAY_MARGIN = 0.02
_GRAVITY_MARGIN = 0.05
_COLLIDER_ELEV_MARGIN = 0.02
_COLLIDER_SCORE_WEIGHT = 0.05
# |R| floor for the static correlation skeleton used by V-well (shielding-robust
# collider cue alongside the evolving adjacency).
_VWELL_SKELETON_THRESHOLD = 0.10

# Exact normal quantile constants for common two-sided alphas; falls back to
# Acklam's rational approximation when scipy is not needed.
_Z_CRIT_TABLE: dict[float, float] = {
    0.001: 3.2905267314919255,
    0.005: 2.807033768343811,
    0.01: 2.5758293035489004,
    0.02: 2.3263478740408408,
    0.05: 1.959963984540054,
    0.10: 1.6448536269514722,
}


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF ``Phi^{-1}(p)`` for ``p in (0, 1)``."""
    try:
        from scipy.stats import norm  # type: ignore[import-untyped]

        return float(norm.ppf(p))
    except ImportError:  # pragma: no cover - fallback when scipy is absent
        # Acklam rational approximation (max abs error < 1.15e-9).
        a = (
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        )
        b = (
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        )
        c = (
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        )
        d = (
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        )
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            )
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


def fisher_z_threshold(
    n_samples: int,
    k_cond: int = 3,
    alpha: float = 0.01,
) -> float:
    """Sample-size-adaptive partial-correlation threshold via Fisher's z-transform.

    Computes ``tanh(z_crit / sqrt(max(n_samples - k_cond - 3, 1)))`` where
    ``z_crit = Phi^{-1}(1 - alpha / 2)`` is the two-sided standard normal
    critical value at significance level ``alpha``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if k_cond < 0:
        raise ValueError(f"k_cond must be >= 0, got {k_cond}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    z_crit = _Z_CRIT_TABLE.get(float(alpha))
    if z_crit is None:
        z_crit = _norm_ppf(1.0 - float(alpha) / 2.0)
    dof = max(int(n_samples) - int(k_cond) - 3, 1)
    return float(math.tanh(z_crit / math.sqrt(dof)))


def dag_to_cpdag(A: np.ndarray) -> np.ndarray:
    """Convert a DAG adjacency matrix to its Completed Partially Directed Acyclic Graph (CPDAG).

    Implements the Chickering / Meek DAG-to-CPDAG conversion:

    1. Start from the undirected skeleton of ``A`` (all edges bidirectional).
    2. Orient unshielded colliders (v-structures) ``i -> k <- j`` where ``i``
       and ``j`` are non-adjacent in the skeleton.
    3. Repeatedly close under Meek's orientation rules R1--R3 until convergence,
       leaving all remaining Markov-equivalent edges undirected
       (``C[i, j] = C[j, i] = 1``).
    """
    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square adjacency matrix")
    n = A_arr.shape[0]
    D = (A_arr > 0).astype(bool)
    np.fill_diagonal(D, False)
    U = D | D.T
    np.fill_diagonal(U, False)

    C = U.astype(np.float64)
    eye = np.eye(n, dtype=bool)
    # 1. Preserve unshielded v-structures i -> k <- j with i, j non-adjacent.
    v_pairs = D[:, None, :] & D[None, :, :] & (~U[:, :, None]) & (~eye[:, :, None])
    has_v_into = np.any(v_pairs, axis=1)
    C = np.where(has_v_into.T & (~has_v_into), 0.0, C)

    # 2. Close under Meek rules R1-R3.
    while True:
        directed = (C > 0) & (C.T == 0)
        undirected = (C > 0) & (C.T > 0)
        adj = (C > 0) | (C.T > 0)
        non_adj = (~adj) & (~eye)

        # R1: orient i -> j if exists k with k -> i and k not adjacent to j.
        r1 = np.any(directed.T[:, :, None] & non_adj[None, :, :], axis=1)
        # R2: orient i -> j if exists k with i -> k and k -> j.
        r2 = np.any(directed[:, :, None] & directed[None, :, :], axis=1)
        # R3: orient i -> j if exists non-adjacent k, l with i - k, i - l, k -> j, l -> j.
        cand = undirected[:, :, None] & directed[None, :, :]
        r3 = np.any(
            cand[:, :, None, :] & cand[:, None, :, :] & non_adj[None, :, :, None],
            axis=(1, 2),
        )

        orient_ij = undirected & (r1 | r2 | r3)
        orient_ij = orient_ij & (~orient_ij.T)
        if not np.any(orient_ij):
            break
        C = np.where(orient_ij.T, 0.0, C)

    return C


def _cov_node_keys(Cov: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Permutation-equivariant per-node keys from ``Cov`` (diag + off-diag).

    Primary signal is ``|diag|``; secondary is a sorted-``|Cov|`` off-diagonal
    fingerprint so equal-variance nodes still separate when their neighbor
    profiles differ. A microscopic index term breaks remaining automorphic
    ties (needed so birth can orient when two nodes are exchangeable).
    """
    C = np.asarray(Cov, dtype=dtype)
    n = C.shape[0]
    diag = np.abs(np.diag(C))
    scale = np.maximum(float(np.max(diag)), 1e-12)
    off = np.abs(C).copy()
    np.fill_diagonal(off, 0.0)
    # Descending off-diagonal profile — label-free given the Cov entries.
    ranked = -np.sort(-off, axis=1)
    weights = (0.5 ** np.arange(n, dtype=dtype))
    fingerprint = ranked @ weights
    fp_scale = np.maximum(float(np.max(np.abs(fingerprint))), 1e-12)
    idx = np.arange(n, dtype=dtype) / max(n, 1)
    return (diag / scale) + 1e-3 * (fingerprint / fp_scale) + 1e-9 * idx


def _tiebreak_elevation(
    E: np.ndarray, Cov: np.ndarray | None = None
) -> np.ndarray:
    """Deterministic elevation jitter so clipped nodes at 0/1 never tie.

    When ``Cov`` is provided, jitter follows Cov-derived node keys
    (permutation-equivariant up to automorphic ties). Otherwise falls back
    to index order (legacy).
    """
    n = E.shape[0]
    if Cov is not None:
        return E + 1e-6 * _cov_node_keys(Cov, E.dtype)
    return E + 1e-6 * (np.arange(n, dtype=E.dtype) / max(n, 1))


def _downhill_mask(E: np.ndarray, Cov: np.ndarray | None = None) -> np.ndarray:
    """``E_i > E_j`` with tie-break only for exact ties at bounds 0/1."""
    raw = E[:, None] > E[None, :]
    e_tb = _tiebreak_elevation(E, Cov)
    tb = e_tb[:, None] > e_tb[None, :]
    at_bound = (E <= 0.0) | (E >= 1.0)
    tied_at_bound = (E[:, None] == E[None, :]) & at_bound[:, None] & at_bound[None, :]
    return np.where(tied_at_bound, tb, raw)


def _undirected_neighbors(A: np.ndarray) -> np.ndarray:
    """Binary undirected adjacency with zero diagonal."""
    U = ((A + A.T) > 0).astype(np.float64)
    np.fill_diagonal(U, 0.0)
    return U


def _correlation_matrix(Cov: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix from a covariance (diagonal-safe)."""
    std = np.sqrt(np.maximum(np.diag(Cov), _STD_FLOOR))
    return Cov / (std[:, None] * std[None, :])


def _topk_by_score(
    mask: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take up to ``k`` masked columns per row with highest ``scores``."""
    M, N = mask.shape
    if k <= 0:
        return np.full((M, 0), -1, dtype=np.intp), np.zeros(M, dtype=np.intp)

    ranked = np.where(mask, scores, -np.inf)
    # Tiny Cov-free index only breaks exact |R| ties (ordering among equals).
    sort_key = -ranked + np.arange(N, dtype=np.float64) * 1e-12
    order = np.argsort(sort_key, axis=1)
    top = order[:, :k]
    rows = np.arange(M)[:, None]
    valid = mask[rows, top]
    Z = np.where(valid, top, -1).astype(np.intp)
    counts = valid.sum(axis=1).astype(np.intp)
    return Z, counts


def _batched_partial_corr(
    Cov: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    Z: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Partial correlation of ``(I[m], J[m])`` given ``Z[m, :counts[m]]``."""
    M = I.shape[0]
    rho = np.zeros(M, dtype=np.float64)
    if M == 0:
        return rho

    k = Z.shape[1] if Z.ndim == 2 else 0
    for n_z in range(0, k + 1):
        idx = np.flatnonzero(counts == n_z)
        if idx.size == 0:
            continue
        parts = [I[idx], J[idx]]
        for t in range(n_z):
            parts.append(Z[idx, t])
        nodes = np.column_stack(parts)
        sub = Cov[nodes[:, :, None], nodes[:, None, :]].astype(np.float64)
        eye = np.eye(sub.shape[-1], dtype=np.float64)
        scale = np.maximum(
            np.mean(np.diagonal(sub, axis1=-2, axis2=-1), axis=-1), _SCALE_FLOOR
        )
        sub = sub + (_RIDGE + 1e-6 * scale)[:, None, None] * eye
        P = np.linalg.inv(sub)
        denom = np.sqrt(np.maximum(P[:, 0, 0] * P[:, 1, 1], 0.0))
        rho[idx] = np.where(denom > 0.0, -P[:, 0, 1] / denom, 0.0)
    return rho


def _v_structure_masks(
    A: np.ndarray,
    Cov: np.ndarray,
    threshold_death: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unshielded explaining-away triples; axes ``(i, k, j)``."""
    n = A.shape[0]
    eye = np.eye(n, dtype=bool)
    U = ((A + A.T) > 0) & (~eye)
    R = _correlation_matrix(Cov)

    wedge = (
        U[:, :, None]
        & U.T[None, :, :]
        & (~U[:, None, :])
        & (~eye[:, None, :])
    )
    r_ij = R[:, None, :]
    r_ik = R[:, :, None]
    r_jk = R.T[None, :, :]
    denom = np.sqrt(np.maximum((1.0 - r_ik**2) * (1.0 - r_jk**2), _STD_FLOOR))
    rho_ij_given_k = np.abs((r_ij - r_ik * r_jk) / denom)
    explains_away = (np.abs(r_ij) < threshold_death) & (
        rho_ij_given_k > np.abs(r_ij) + _EXPLAIN_AWAY_MARGIN
    )
    is_v_triple = wedge & explains_away
    is_collider_k = np.any(is_v_triple, axis=(0, 2))
    is_v_parent_i = np.any(is_v_triple, axis=(1, 2))
    return is_v_triple, is_collider_k, is_v_parent_i


def detect_colliders(
    A: np.ndarray,
    Cov: np.ndarray,
    threshold_death: float = 0.05,
) -> np.ndarray:
    """Boolean mask of explaining-away V-structure colliders."""
    _, is_collider_k, _ = _v_structure_masks(
        np.asarray(A, dtype=np.float64),
        np.asarray(Cov, dtype=np.float64),
        threshold_death,
    )
    return is_collider_k


def _union_neighbor_sets(
    U: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    k: int,
    Cov: np.ndarray,
    *,
    E: np.ndarray | None = None,
    threshold_death: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` neighbors ranked by ``|R[i,z]| + |R[j,z]|`` with collider down-weighting."""
    M = I.shape[0]
    if M == 0:
        return np.full((0, k), -1, dtype=np.intp), np.zeros(0, dtype=np.intp)

    mask = (U[I] > 0) | (U[J] > 0)
    mask[np.arange(M), I] = False
    mask[np.arange(M), J] = False
    R = _correlation_matrix(Cov)
    abs_R = np.abs(R)
    if threshold_death is not None:
        corr_connected = (abs_R[I] >= float(threshold_death)) | (
            abs_R[J] >= float(threshold_death)
        )
        mask = mask & corr_connected
    scores = abs_R[I] + abs_R[J]
    if E is not None:
        below_both = E[None, :] < (
            np.minimum(E[I, None], E[J, None]) - _COLLIDER_ELEV_MARGIN
        )
        scores = np.where(below_both, scores * _COLLIDER_SCORE_WEIGHT, scores)
    return _topk_by_score(mask, scores, k)


def _correlation_skeleton(Cov: np.ndarray, threshold: float) -> np.ndarray:
    """Symmetric 0/1 adjacency from ``|R_ij| > threshold`` (zero diagonal)."""
    R = _correlation_matrix(Cov)
    U = (np.abs(R) > float(threshold)).astype(np.float64)
    np.fill_diagonal(U, 0.0)
    return ((U + U.T) > 0).astype(np.float64)


def _flat_vwell_colliders(
    U: np.ndarray,
    Cov: np.ndarray,
    threshold_death: float,
) -> np.ndarray:
    """Shared neighbors of marginally independent absent pairs on ``U``."""
    n = U.shape[0]
    undirected_absent = U == 0.0
    np.fill_diagonal(undirected_absent, False)
    shared_counts = U @ U
    np.fill_diagonal(shared_counts, 0.0)
    pair_mask = np.triu(undirected_absent & (shared_counts > 0), k=1)
    Pi, Pj = np.nonzero(pair_mask)
    if Pi.size == 0:
        return np.zeros(n, dtype=bool)

    Z_empty = np.zeros((Pi.size, 0), dtype=np.intp)
    counts0 = np.zeros(Pi.size, dtype=np.intp)
    rho = _batched_partial_corr(Cov, Pi, Pj, Z_empty, counts0)
    independent = np.abs(rho) < threshold_death
    if not np.any(independent):
        return np.zeros(n, dtype=bool)

    shared = (U[Pi[independent]] > 0) & (U[Pj[independent]] > 0)
    return shared.any(axis=0)


def _vstructure_gravity_well(
    A: np.ndarray,
    E: np.ndarray,
    Cov: np.ndarray,
    *,
    learning_rate: float,
    threshold_death: float,
) -> np.ndarray:
    """V-well: multiplicatively sink shared neighbors of independent pairs.

    Uses ``E ← E · (1 − 2·lr)`` on collider candidates instead of an additive
    ``E − 2·lr`` slam into the ``0`` clip wall, which destroyed relative
    ordering near the floor and drove even/odd SHD limit cycles.

    Colliders are the union of cues from the evolving undirected adjacency
    and a static ``|R|`` correlation skeleton (shielding-robust).
    """
    U_a = _undirected_neighbors(A)
    U_c = _correlation_skeleton(Cov, _VWELL_SKELETON_THRESHOLD)
    collider = _flat_vwell_colliders(
        U_a, Cov, threshold_death
    ) | _flat_vwell_colliders(U_c, Cov, threshold_death)
    if not np.any(collider):
        return E

    E = E.copy()
    decay = max(0.0, 1.0 - 2.0 * float(learning_rate))
    E[collider] *= decay
    return np.clip(E, 0.0, 1.0)


def _elevation_gravity_target(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Local-margin elevation target (no hard 1.0 / 0.0 anchors)."""
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    raw_mean_parents = (A.T @ E) / np.maximum(in_deg, 1.0)
    raw_mean_children = (A @ E) / np.maximum(out_deg, 1.0)
    delta = _GRAVITY_MARGIN

    both = (in_deg > 0) & (out_deg > 0)
    root = (in_deg == 0) & (out_deg > 0)
    leaf = (out_deg == 0) & (in_deg > 0)

    target = E.copy()
    target[both] = 0.5 * (raw_mean_parents[both] + raw_mean_children[both])
    target[root] = np.minimum(raw_mean_children[root] + delta, 1.0)
    target[leaf] = np.maximum(raw_mean_parents[leaf] - delta, 0.0)
    return target


def step(
    A: np.ndarray,
    E: np.ndarray,
    Cov: np.ndarray,
    *,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    k: int = 2,
    n_samples: int | None = None,
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One Bayesian CA tick: gravity, V-well, death, then birth.

    Parameters
    ----------
    A:
        Binary adjacency ``(N, N)``; ``A[i, j] = 1`` means directed edge ``i→j``.
    E:
        Node elevations ``(N,)``, intended in ``[0, 1]``.
    Cov:
        Global covariance ``(N, N)`` of the continuous observations.
    learning_rate:
        Step size for elevation drift toward the local-margin gravity target.
    threshold_death:
        Kill existing edges with ``|rho|`` below this value.
    threshold_birth:
        Birth candidate edges with ``|rho|`` above this value.
    k:
        Cap on conditioning-set size ``|Z|`` (default 2).
    n_samples:
        Optional sample count ``n``. When provided and ``threshold_death`` /
        ``threshold_birth`` are left at their defaults, thresholds are calibrated
        via :func:`fisher_z_threshold`.
    alpha:
        Optional two-sided significance level for :func:`fisher_z_threshold`
        (default ``0.01`` when ``n_samples`` is provided).

    Returns
    -------
    A_new, E_new
        Updated binary adjacency and elevations (copies; inputs are not mutated).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    E = np.asarray(E, dtype=np.float64).copy().reshape(-1)
    Cov = np.asarray(Cov, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")
    N = A.shape[0]
    if E.shape != (N,):
        raise ValueError(f"E must have shape ({N},), got {E.shape}")
    if Cov.shape != (N, N):
        raise ValueError(f"Cov must have shape ({N}, {N}), got {Cov.shape}")
    if k < 0:
        raise ValueError("k must be non-negative")

    if n_samples is not None:
        alpha_val = 0.01 if alpha is None else float(alpha)
        fz_death = fisher_z_threshold(int(n_samples), k_cond=k, alpha=alpha_val)
        if threshold_death == 0.05:
            threshold_death = fz_death
        if threshold_birth == 0.15:
            threshold_birth = 1.2 * threshold_death

    # --- 1. Gravity: raw downhill, then local-margin drift -------------------
    A *= E[:, None] > E[None, :]
    np.fill_diagonal(A, 0.0)
    target = _elevation_gravity_target(A, E)
    E = np.clip(E + learning_rate * (target - E), 0.0, 1.0)

    # --- 2. V-structure gravity well ----------------------------------------
    E = _vstructure_gravity_well(
        A,
        E,
        Cov,
        learning_rate=learning_rate,
        threshold_death=threshold_death,
    )

    # --- 3. Death ------------------------------------------------------------
    U = _undirected_neighbors(A)
    I, J = np.nonzero(A > 0)
    if I.size:
        Z, counts = _union_neighbor_sets(
            U, I, J, k, Cov, E=E, threshold_death=threshold_death
        )
        rho = _batched_partial_corr(Cov, I, J, Z, counts)
        kill = np.abs(rho) < threshold_death
        if np.any(kill):
            A[I[kill], J[kill]] = 0.0
            U = _undirected_neighbors(A)

    # --- 4. Birth ------------------------------------------------------------
    undirected_absent = (A + A.T) == 0.0
    np.fill_diagonal(undirected_absent, False)
    Bi, Bj = np.nonzero(np.triu(undirected_absent, k=1))
    if Bi.size:
        Z, counts = _union_neighbor_sets(
            U, Bi, Bj, k, Cov, E=E, threshold_death=threshold_death
        )
        rho = _batched_partial_corr(Cov, Bi, Bj, Z, counts)
        strong = np.abs(rho) > threshold_birth
        e_tb = _tiebreak_elevation(E, Cov)
        forward = strong & (e_tb[Bi] > e_tb[Bj])
        backward = strong & (e_tb[Bj] > e_tb[Bi])
        if np.any(forward):
            A[Bi[forward], Bj[forward]] = 1.0
        if np.any(backward):
            A[Bj[backward], Bi[backward]] = 1.0

    # Final downhill with bound-aware, Cov-equivariant tie-break.
    A = (A > 0).astype(np.float64)
    A *= _downhill_mask(E, Cov)
    np.fill_diagonal(A, 0.0)
    return A, E
