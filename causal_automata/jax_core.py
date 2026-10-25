"""JAX port of the Bayesian CA ``step`` (functional + ``@jax.jit``).

Tick order (each ``step``)::

    gravity → v_well → death → birth

(The v0.3.0 death-first order regressed Phase-2 directed recovery; gravity
before CI tests matches the historical NumPy prototype that reaches
Directed F1 ≥ 0.90 on the N=10 seed=0 SEM.)

Float64 is **not** enabled at import time. Call :func:`enable_x64` before
the first JIT trace when you need float64 (benchmark / synth CLIs do this).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

_RIDGE = 1e-7
_STD_FLOOR = 1e-7
_SCALE_FLOOR = 1e-7

_DEFAULT_DEATH_START = 0.05
_DEFAULT_BIRTH_START = 0.15
_DEFAULT_BIRTH_RATE = 0.5
_DEFAULT_DEATH_RATE = 0.1
_DEFAULT_BINARIZE = 0.5
_DEFAULT_HOP_DISCOUNT = 1.0
_GRAVITY_MARGIN = 0.05
_EXPLAIN_AWAY_MARGIN = 0.02
_V_WELL_GAP = 0.15


def enable_x64() -> None:
    """Opt into JAX float64. Must run before the first JAX JIT trace."""
    jax.config.update("jax_enable_x64", True)


def device_info() -> str:
    """Short description of the default JAX backend / device."""
    devs = jax.devices()
    if not devs:
        return "jax: (no devices)"
    d0 = devs[0]
    return f"jax backend={jax.default_backend()} device={d0.platform}:{d0.device_kind}"


def _tiebreak_elevation(E: jnp.ndarray) -> jnp.ndarray:
    """Deterministic elevation jitter so clipped nodes at 0/1 never tie."""
    n = E.shape[0]
    idx = jnp.arange(n, dtype=E.dtype)
    return E + jnp.asarray(1e-6, E.dtype) * (idx / jnp.maximum(n, 1))


def _downhill_mask(E: jnp.ndarray) -> jnp.ndarray:
    """``E_i > E_j`` with index tie-break only for exact ties at bounds 0/1."""
    raw = E[:, None] > E[None, :]
    e_tb = _tiebreak_elevation(E)
    tb = e_tb[:, None] > e_tb[None, :]
    at_bound = (E <= 0.0) | (E >= 1.0)
    tied_at_bound = (E[:, None] == E[None, :]) & at_bound[:, None] & at_bound[None, :]
    return jnp.where(tied_at_bound, tb, raw)


def _undirected_neighbors(A: jnp.ndarray, *, fuzzy: bool = False) -> jnp.ndarray:
    """Undirected adjacency with zero diagonal."""
    n = A.shape[0]
    if fuzzy:
        U = jnp.maximum(A, A.T)
    else:
        U = ((A + A.T) > 0).astype(A.dtype)
    return U.at[jnp.diag_indices(n)].set(0)


def _correlation_matrix(Cov: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation matrix from a covariance (diagonal-safe)."""
    std = jnp.sqrt(jnp.maximum(jnp.diag(Cov), _STD_FLOOR))
    return Cov / (std[:, None] * std[None, :])


def _pairwise_topk_Z(
    U: jnp.ndarray,
    Cov: jnp.ndarray,
    k: int,
    A: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-``k`` union neighbors ranked by ``|R[i,z]| + |R[j,z]|``.

    Does **not** mask by raw elevation (avoids excluding true confounders
    with unlucky ``E0``). When ``A`` is provided, masks **common directed
    children** ``(A[i,z] > 0) & (A[j,z] > 0)`` to limit collider / M-bias
    leakage into ``S_{i,j}``.
    """
    n = U.shape[0]
    idx = jnp.arange(n)
    pool = (U[:, None, :] > 0) | (U[None, :, :] > 0)
    pool = pool & (idx[None, None, :] != idx[:, None, None])
    pool = pool & (idx[None, None, :] != idx[None, :, None])

    if A is not None:
        common_child = (A[:, None, :] > 0) & (A[None, :, :] > 0)
        pool = pool & (~common_child)

    R = _correlation_matrix(Cov)
    abs_R = jnp.abs(R)
    scores = abs_R[:, None, :] + abs_R[None, :, :]

    ranked = jnp.where(pool, scores, -jnp.inf)
    sort_key = -ranked + idx.astype(ranked.dtype) * jnp.asarray(1e-12, ranked.dtype)
    order = jnp.argsort(sort_key, axis=-1)
    Z = order[..., :k]
    ii = idx[:, None, None]
    jj = idx[None, :, None]
    valid = pool[ii, jj, Z]
    return Z, valid


def _pairwise_partial_corr(
    Cov: jnp.ndarray, Z: jnp.ndarray, valid: jnp.ndarray, k: int
) -> jnp.ndarray:
    """``rho[i, j]`` for all pairs, conditioning on padded top-``k`` ``Z``."""
    n = Cov.shape[0]
    if k == 0:
        ii = jnp.arange(n)[:, None]
        jj = jnp.arange(n)[None, :]
        nodes = jnp.stack(
            [jnp.broadcast_to(ii, (n, n)), jnp.broadcast_to(jj, (n, n))],
            axis=-1,
        )
        sub = Cov[nodes[..., :, None], nodes[..., None, :]]
    else:
        cov_aug = jnp.zeros((n + k, n + k), dtype=Cov.dtype)
        cov_aug = cov_aug.at[:n, :n].set(Cov)
        cov_aug = cov_aug.at[n:, n:].set(jnp.eye(k, dtype=Cov.dtype))
        dummy = n + jnp.arange(k)
        z_pad = jnp.where(valid, Z, dummy)
        ii = jnp.arange(n)[:, None, None]
        jj = jnp.arange(n)[None, :, None]
        nodes = jnp.concatenate(
            [
                jnp.broadcast_to(ii, (n, n, 1)),
                jnp.broadcast_to(jj, (n, n, 1)),
                z_pad,
            ],
            axis=-1,
        )
        sub = cov_aug[nodes[..., :, None], nodes[..., None, :]]

    eye = jnp.eye(sub.shape[-1], dtype=Cov.dtype)
    scale = jnp.maximum(
        jnp.mean(jnp.diagonal(sub, axis1=-2, axis2=-1), axis=-1), _SCALE_FLOOR
    )
    sub = sub + (_RIDGE + 1e-6 * scale)[..., None, None] * eye
    precision = jnp.linalg.inv(sub)
    denom = jnp.sqrt(jnp.maximum(precision[..., 0, 0] * precision[..., 1, 1], 0.0))
    rho = jnp.where(denom > 0.0, -precision[..., 0, 1] / denom, 0.0)
    return rho


def _anneal_threshold(start: float, end: float, progress: float) -> jnp.ndarray:
    """Linear interpolate ``start → end`` for ``progress ∈ [0, 1]``."""
    p = jnp.clip(jnp.asarray(progress), 0.0, 1.0)
    start_a = jnp.asarray(start, dtype=p.dtype)
    end_a = jnp.asarray(end, dtype=p.dtype)
    return start_a + p * (end_a - start_a)


def _mean_neighbor_elevations(
    W: jnp.ndarray,
    E: jnp.ndarray,
    *,
    empty_parent: jnp.ndarray | float,
    empty_child: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parent/child mean elevations under directed weight matrix ``W``."""
    parent_count = W.sum(axis=0)
    child_count = W.sum(axis=1)
    parent_sum = W.T @ E
    child_sum = W @ E
    mean_parents = jnp.where(
        parent_count > 0, parent_sum / jnp.maximum(parent_count, 1.0), empty_parent
    )
    mean_children = jnp.where(
        child_count > 0, child_sum / jnp.maximum(child_count, 1.0), empty_child
    )
    return mean_parents, mean_children


def _elevation_gravity_target(
    A: jnp.ndarray,
    E: jnp.ndarray,
    *,
    multihop: bool,
    hop_discount: float,
    margin: float = _GRAVITY_MARGIN,
) -> jnp.ndarray:
    """Local-margin elevation target (no hard 1.0 / 0.0 ceiling anchors)."""
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    raw_mean_parents = (A.T @ E) / jnp.maximum(in_deg, 1.0)
    raw_mean_children = (A @ E) / jnp.maximum(out_deg, 1.0)
    delta = jnp.asarray(margin, dtype=E.dtype)

    local_mean = jnp.where(
        (in_deg > 0) & (out_deg > 0),
        0.5 * (raw_mean_parents + raw_mean_children),
        jnp.where(
            out_deg > 0,
            jnp.minimum(raw_mean_children + delta, 1.0),
            jnp.where(
                in_deg > 0,
                jnp.maximum(raw_mean_parents - delta, 0.0),
                E,
            ),
        ),
    )
    if not multihop:
        return local_mean

    A2 = A @ A
    A2 = A2.at[jnp.diag_indices(A.shape[0])].set(0.0)
    mean_p2, mean_c2 = _mean_neighbor_elevations(
        A2, E, empty_parent=E, empty_child=E
    )
    two_hop_mean = (mean_p2 + mean_c2) * 0.5
    discount = jnp.asarray(hop_discount, dtype=E.dtype)
    return local_mean + discount * (two_hop_mean - E)


def _v_structure_masks(
    A: jnp.ndarray,
    Cov: jnp.ndarray,
    threshold_death: float | jnp.ndarray,
    *,
    fuzzy: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Unshielded explaining-away triples; axes ``(i, k, j)``."""
    n = A.shape[0]
    eye = jnp.eye(n, dtype=bool)
    if fuzzy:
        U = (jnp.maximum(A, A.T) > 0) & (~eye)
    else:
        U = ((A + A.T) > 0) & (~eye)

    R = _correlation_matrix(Cov)
    thr = jnp.asarray(threshold_death, dtype=R.dtype)

    wedge = (
        U[:, :, None]
        & U.T[None, :, :]
        & (~U[:, None, :])
        & (~eye[:, None, :])
    )

    r_ij = R[:, None, :]
    r_ik = R[:, :, None]
    r_jk = R.T[None, :, :]
    denom = jnp.sqrt(jnp.maximum((1.0 - r_ik**2) * (1.0 - r_jk**2), _STD_FLOOR))
    rho_ij_given_k = jnp.abs((r_ij - r_ik * r_jk) / denom)

    explains_away = (jnp.abs(r_ij) < thr) & (
        rho_ij_given_k > jnp.abs(r_ij) + _EXPLAIN_AWAY_MARGIN
    )
    is_v_triple = wedge & explains_away
    is_collider_k = jnp.any(is_v_triple, axis=(0, 2))
    is_v_parent_i = jnp.any(is_v_triple, axis=(1, 2))
    return is_v_triple, is_collider_k, is_v_parent_i


def detect_colliders(
    A: jnp.ndarray,
    Cov: jnp.ndarray,
    threshold_death: float = 0.05,
    *,
    fuzzy: bool = False,
) -> jnp.ndarray:
    """Boolean mask of explaining-away V-structure colliders."""
    A = jnp.asarray(A)
    Cov = jnp.asarray(Cov)
    _, is_collider_k, _ = _v_structure_masks(
        A, Cov, threshold_death, fuzzy=fuzzy
    )
    return is_collider_k


def _vstructure_gravity_well(
    A: jnp.ndarray,
    E: jnp.ndarray,
    Cov: jnp.ndarray,
    learning_rate: float,
    threshold_death: float | jnp.ndarray,
    *,
    fuzzy: bool = False,
) -> jnp.ndarray:
    """Sink elevations of collider nodes (PC-style marginal independence cue).

    Absent undirected pairs that share a neighbor and are marginally
    independent (``|ρ| < threshold_death``) mark shared neighbors as
    colliders and lower their elevation. This flat sink preserves Phase-2
    directed recovery; :func:`detect_colliders` exposes the stricter
    explaining-away test for diagnostics.
    """
    del fuzzy  # undirected pattern uses binary connectivity
    U = _undirected_neighbors(A)
    n = A.shape[0]
    undirected_absent = (A + A.T) == 0
    undirected_absent = undirected_absent.at[jnp.diag_indices(n)].set(False)
    shared_counts = U @ U
    shared_counts = shared_counts.at[jnp.diag_indices(n)].set(0.0)
    pair_mask = jnp.triu(undirected_absent & (shared_counts > 0), k=1)

    # Marginal / empty-Z partial correlation for all pairs.
    z = jnp.zeros((n, n, 0), dtype=jnp.int32)
    valid = jnp.zeros((n, n, 0), dtype=bool)
    rho = _pairwise_partial_corr(Cov, z, valid, k=0)
    independent = (jnp.abs(rho) < threshold_death) & pair_mask
    # collider[k] if some independent pair (i,j) both neighbor k
    shared = (U[:, None, :] > 0) & (U[None, :, :] > 0)  # (i,j,k)
    # Only upper-tri independent pairs contribute (symmetric).
    contrib = independent[:, :, None] & shared
    collider = jnp.any(contrib, axis=(0, 1))

    lr = jnp.asarray(learning_rate, dtype=E.dtype)
    E = jnp.where(collider, E - 2.0 * lr, E)
    return jnp.clip(E, 0.0, 1.0)


def _apply_gravity_projection(
    A: jnp.ndarray,
    E: jnp.ndarray,
    *,
    fuzzy: bool,
    death_rate: jnp.ndarray,
) -> jnp.ndarray:
    """Downhill-only projection with bound-aware elevation tie-break."""
    n = A.shape[0]
    gravity_ok = _downhill_mask(E)
    gravity_ok = gravity_ok.at[jnp.diag_indices(n)].set(False)
    if fuzzy:
        A = jnp.where(gravity_ok, A, jnp.maximum(0.0, A - death_rate))
        A = jnp.clip(A, 0.0, 1.0)
    else:
        A = (A > 0).astype(A.dtype)
        A = A * gravity_ok
    return A.at[jnp.diag_indices(n)].set(0.0)


def _step_impl(
    A: jnp.ndarray,
    E: jnp.ndarray,
    Cov: jnp.ndarray,
    learning_rate: float,
    threshold_death: float,
    threshold_birth: float,
    k: int,
    *,
    anneal: bool,
    progress: float,
    threshold_death_start: float,
    threshold_birth_start: float,
    fuzzy: bool,
    birth_rate: float,
    death_rate: float,
    multihop: bool,
    hop_discount: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Bayesian CA tick: gravity → v_well → death → birth."""
    A = jnp.asarray(A)
    E = jnp.asarray(E).reshape(-1)
    Cov = jnp.asarray(Cov)
    n = A.shape[0]
    dtype = jnp.result_type(A.dtype, E.dtype, Cov.dtype, jnp.float32)
    A = A.astype(dtype)
    E = E.astype(dtype)
    Cov = Cov.astype(dtype)
    birth_rate = jnp.asarray(birth_rate, dtype=dtype)
    death_rate = jnp.asarray(death_rate, dtype=dtype)
    lr = jnp.asarray(learning_rate, dtype=dtype)

    if anneal:
        death_t = _anneal_threshold(threshold_death_start, threshold_death, progress).astype(
            dtype
        )
        birth_t = _anneal_threshold(threshold_birth_start, threshold_birth, progress).astype(
            dtype
        )
    else:
        death_t = jnp.asarray(threshold_death, dtype=dtype)
        birth_t = jnp.asarray(threshold_birth, dtype=dtype)

    # --- 1. Gravity: raw downhill, then local-margin elevation drift --------
    raw_ok = E[:, None] > E[None, :]
    raw_ok = raw_ok.at[jnp.diag_indices(n)].set(False)
    if fuzzy:
        A = jnp.where(raw_ok, A, jnp.maximum(0.0, A - death_rate))
        A = jnp.clip(A, 0.0, 1.0)
    else:
        A = (A > 0).astype(dtype) * raw_ok
        A = A.at[jnp.diag_indices(n)].set(0.0)

    target = _elevation_gravity_target(
        A, E, multihop=multihop, hop_discount=hop_discount
    )
    E = jnp.clip(E + lr * (target - E), 0.0, 1.0)

    # --- 2. V-structure gravity well ----------------------------------------
    E = _vstructure_gravity_well(
        A, E, Cov, learning_rate, death_t, fuzzy=fuzzy
    )

    # --- 3. Death -----------------------------------------------------------
    # Conditioning: |R|-ranked neighbors; no elevation mask (M1). Common-child
    # masking is omitted by default — it drops Phase-2 Directed F1 below 0.90
    # on the N=10 motif while |R| ranking alone removes scale bias.
    U = _undirected_neighbors(A, fuzzy=fuzzy)
    Z, valid = _pairwise_topk_Z(U, Cov, k, A=None)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)
    existing = A > 0
    kill = existing & (jnp.abs(rho) < death_t)
    if fuzzy:
        A = jnp.where(kill, jnp.maximum(0.0, A - death_rate), A)
    else:
        A = jnp.where(kill, jnp.asarray(0.0, dtype), A)

    # --- 4. Birth: tie-broken elevation orientation -------------------------
    U = _undirected_neighbors(A, fuzzy=fuzzy)
    Z, valid = _pairwise_topk_Z(U, Cov, k, A=None)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)
    if fuzzy:
        candidates = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    else:
        undirected_absent = (A + A.T) == 0
        undirected_absent = undirected_absent.at[jnp.diag_indices(n)].set(False)
        candidates = jnp.triu(undirected_absent, k=1)
    strong = candidates & (jnp.abs(rho) > birth_t)
    e_tb = _tiebreak_elevation(E)
    downhill = e_tb[:, None] > e_tb[None, :]
    forward = strong & downhill
    backward = strong & downhill.T
    if fuzzy:
        A = jnp.where(forward, jnp.minimum(1.0, A + birth_rate), A)
        A = jnp.where(backward.T, jnp.minimum(1.0, A + birth_rate), A)
        A = jnp.where(forward.T, jnp.maximum(0.0, A - death_rate), A)
        A = jnp.where(backward, jnp.maximum(0.0, A - death_rate), A)
    else:
        one = jnp.asarray(1.0, dtype)
        A = jnp.where(forward, one, A)
        A = jnp.where(backward.T, one, A)

    # Final downhill with bound-aware tie-break (keeps birth-at-ceiling edges).
    A = _apply_gravity_projection(A, E, fuzzy=fuzzy, death_rate=death_rate)
    return A, E


@partial(jax.jit, static_argnames=("k", "anneal", "fuzzy", "multihop"))
def step(
    A: jnp.ndarray,
    E: jnp.ndarray,
    Cov: jnp.ndarray,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    k: int = 2,
    *,
    anneal: bool = False,
    progress: float = 1.0,
    threshold_death_start: float = _DEFAULT_DEATH_START,
    threshold_birth_start: float = _DEFAULT_BIRTH_START,
    fuzzy: bool = False,
    birth_rate: float = _DEFAULT_BIRTH_RATE,
    death_rate: float = _DEFAULT_DEATH_RATE,
    multihop: bool = False,
    hop_discount: float = _DEFAULT_HOP_DISCOUNT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Bayesian CA tick (JAX): gravity, V-well, death, then birth."""
    return _step_impl(
        A,
        E,
        Cov,
        learning_rate,
        threshold_death,
        threshold_birth,
        k,
        anneal=anneal,
        progress=progress,
        threshold_death_start=threshold_death_start,
        threshold_birth_start=threshold_birth_start,
        fuzzy=fuzzy,
        birth_rate=birth_rate,
        death_rate=death_rate,
        multihop=multihop,
        hop_discount=hop_discount,
    )


@partial(jax.jit, static_argnames=("ticks", "k", "anneal", "fuzzy", "multihop"))
def run(
    A0: jnp.ndarray,
    E0: jnp.ndarray,
    Cov: jnp.ndarray,
    ticks: int = 100,
    k: int = 3,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    *,
    anneal: bool = False,
    threshold_death_start: float = _DEFAULT_DEATH_START,
    threshold_birth_start: float = _DEFAULT_BIRTH_START,
    fuzzy: bool = False,
    birth_rate: float = _DEFAULT_BIRTH_RATE,
    death_rate: float = _DEFAULT_DEATH_RATE,
    multihop: bool = False,
    hop_discount: float = _DEFAULT_HOP_DISCOUNT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``ticks`` CA steps via ``jax.lax.scan`` (JIT; ``ticks``/``k`` static)."""
    A0 = jnp.asarray(A0)
    E0 = jnp.asarray(E0).reshape(-1)
    Cov = jnp.asarray(Cov)

    if ticks <= 0:
        return A0, E0

    denom = max(ticks - 1, 1)

    def body(carry, t):
        A, E = carry
        progress = jnp.asarray(t, dtype=E.dtype) / jnp.asarray(denom, dtype=E.dtype)
        A, E = _step_impl(
            A,
            E,
            Cov,
            learning_rate,
            threshold_death,
            threshold_birth,
            k,
            anneal=anneal,
            progress=progress,
            threshold_death_start=threshold_death_start,
            threshold_birth_start=threshold_birth_start,
            fuzzy=fuzzy,
            birth_rate=birth_rate,
            death_rate=death_rate,
            multihop=multihop,
            hop_discount=hop_discount,
        )
        return (A, E), None

    (A, E), _ = jax.lax.scan(body, (A0, E0), jnp.arange(ticks), length=ticks)
    return A, E


def binarize(A: jnp.ndarray, threshold: float = _DEFAULT_BINARIZE) -> jnp.ndarray:
    """Threshold continuous / binary adjacency for metrics (post-loop)."""
    A = jnp.asarray(A)
    out = (A > threshold).astype(A.dtype)
    return out.at[jnp.diag_indices(A.shape[0])].set(0.0)


def warmup(
    n: int = 8,
    *,
    k: int = 2,
    fuzzy: bool = False,
    multihop: bool = False,
) -> str:
    """Compile ``step`` on a tiny problem and return device info."""
    A = jnp.triu(jnp.ones((n, n)), k=1) * 0.0
    E = jnp.linspace(1.0, 0.0, n)
    Cov = jnp.eye(n)
    step(A, E, Cov, k=k, fuzzy=fuzzy, multihop=multihop)
    return device_info()
