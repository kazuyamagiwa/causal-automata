"""JAX port of the Bayesian CA ``step`` (functional + ``@jax.jit``).

Cross-platform module using only ``jax`` / ``jax.numpy``. Accelerator
selection is left to the installed JAX wheel (CPU, CUDA, or Metal/MPS) —
no hardcoded backend branches.

Install (pick one accelerator wheel)::

    uv pip install "causal-automata[jax]"   # CPU jax/jaxlib
    # Apple Silicon Metal/MPS: install a Metal-enabled JAX wheel (e.g. jax-mps)
    # NVIDIA / Colab: pip install "causal-automata[cuda]"  (see JAX docs)

Smoke::

    from causal_automata.jax_core import step, run, device_info, warmup
    print(warmup())
    A1, E1 = step(A, E, Cov)  # jit-compiled on first call
    Af, Ef = run(A, E, Cov, ticks=100, k=3)  # scan loop; ticks/k are static

Upgrade flags (all opt-in; defaults preserve the binary 1-hop CA):

* **Upgrade 1** — ``anneal=True`` with ``progress`` in ``[0, 1]``.
* **Upgrade 3** — ``fuzzy=True`` keeps continuous ``A ∈ [0, 1]``; binarize
  after the final tick (e.g. :func:`binarize`).
* **Upgrade 4** — ``multihop=True`` adds 2-hop elevation pull via ``A @ A``
  (Maximum Accuracy Mode; ``hop_discount=1.0`` by default).

Tick order (each ``step``)::

    death → birth → v_well → gravity
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Match the NumPy prototype's float64 path (partial-corr precision).
jax.config.update("jax_enable_x64", True)

_RIDGE = 1e-8

# Default linear annealing endpoints (progress 0 → 1). When start == end,
# thresholds are static regardless of ``progress`` (backward compatible).
_DEFAULT_DEATH_START = 0.05
_DEFAULT_BIRTH_START = 0.15

# Upgrade-3 soft birth / death step sizes (used only when ``fuzzy=True``).
# Tuned on N=100 seed=0 (k=5): slower death preserves weak true edges longer.
_DEFAULT_BIRTH_RATE = 0.5
_DEFAULT_DEATH_RATE = 0.1
_DEFAULT_BINARIZE = 0.5

# Upgrade-4 2-hop gravity discount (used only when ``multihop=True``).
# N=100 seed=0 k=5: discount=1.0 lifts Directed F1 0.619 → 0.696.
_DEFAULT_HOP_DISCOUNT = 1.0

# Explaining-away margin: |ρ_{ij|k}| must exceed |ρ_ij| by at least this.
_EXPLAIN_AWAY_MARGIN = 0.02
# Collider elevation sits this far below min(parent elevations).
_V_WELL_GAP = 0.15


def device_info() -> str:
    """Short description of the default JAX backend / device."""
    devs = jax.devices()
    if not devs:
        return "jax: (no devices)"
    d0 = devs[0]
    return f"jax backend={jax.default_backend()} device={d0.platform}:{d0.device_kind}"


def _undirected_neighbors(A: jnp.ndarray, *, fuzzy: bool = False) -> jnp.ndarray:
    """Undirected adjacency with zero diagonal.

    Binary mode: ``(A + A.T) > 0``. Fuzzy mode: continuous
    ``max(A, A.T)`` so gravity / V-wells feel fractional edge mass.
    """
    n = A.shape[0]
    if fuzzy:
        U = jnp.maximum(A, A.T)
    else:
        U = ((A + A.T) > 0).astype(A.dtype)
    return U.at[jnp.diag_indices(n)].set(0)


def _pairwise_topk_Z(
    U: jnp.ndarray,
    Cov: jnp.ndarray,
    k: int,
    E: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-``k`` union neighbors for every pair ``(i, j)``, ranked by |Cov|.

    When ``E`` is provided, candidates ``z`` with
    ``E[z] < min(E[i], E[j])`` are masked out (score ``-inf``) so common
    descendants / colliders cannot enter the conditioning set ``S_{i,j}``.

    Returns
    -------
    Z, valid
        ``Z`` shape ``(N, N, k)``; ``valid`` same shape (bool).
    """
    n = U.shape[0]
    idx = jnp.arange(n)
    # pool[i, j, m] = neighbor of i or j, excluding endpoints
    pool = (U[:, None, :] > 0) | (U[None, :, :] > 0)
    pool = pool & (idx[None, None, :] != idx[:, None, None])
    pool = pool & (idx[None, None, :] != idx[None, :, None])

    # scores[i, j, m] = max(|Cov[m, i]|, |Cov[m, j]|)
    score_i = jnp.abs(Cov.T)[:, None, :]  # (i, 1, m)
    score_j = jnp.abs(Cov.T)[None, :, :]  # (1, j, m)
    scores = jnp.maximum(score_i, score_j)

    if E is not None:
        # Mask common descendants: E[z] < min(E[i], E[j]).
        min_ij = jnp.minimum(E[:, None], E[None, :])
        below_both = E[None, None, :] < min_ij[:, :, None]
        pool = pool & (~below_both)

    ranked = jnp.where(pool, scores, -jnp.inf)
    sort_key = -ranked + idx.astype(ranked.dtype) * jnp.asarray(1e-12, ranked.dtype)
    order = jnp.argsort(sort_key, axis=-1)
    Z = order[..., :k]
    # valid[i,j,t] = pool[i,j,Z[i,j,t]]
    ii = idx[:, None, None]
    jj = idx[None, :, None]
    valid = pool[ii, jj, Z]
    return Z, valid


def _pairwise_partial_corr(
    Cov: jnp.ndarray, Z: jnp.ndarray, valid: jnp.ndarray, k: int
) -> jnp.ndarray:
    """``rho[i, j]`` for all pairs, conditioning on padded top-``k`` ``Z``.

    Builds a batched block of shape ``(N, N, 2+k, 2+k)`` (endpoints plus the
    conditioning set) and inverts with ``jnp.linalg.inv``. ``k`` is a JIT
    static arg, so ``k=3`` → ``5×5``, ``k=4`` → ``6×6``, ``k=5`` → ``7×7``
    each get a dedicated XLA specialization — no Python resize per tick.

    Invalid ``Z`` slots map to independent dummy variables appended to ``Cov``
    so the invert size stays fixed under JIT (conditioning on independents is a
    no-op relative to a smaller ``Z``).
    """
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
        # Augment Cov with k independent unit dummies (pad short Z sets).
        cov_aug = jnp.zeros((n + k, n + k), dtype=Cov.dtype)
        cov_aug = cov_aug.at[:n, :n].set(Cov)
        cov_aug = cov_aug.at[n:, n:].set(jnp.eye(k, dtype=Cov.dtype))
        dummy = n + jnp.arange(k)
        z_pad = jnp.where(valid, Z, dummy)
        ii = jnp.arange(n)[:, None, None]
        jj = jnp.arange(n)[None, :, None]
        # nodes[..., :] = (i, j, Z_0, …, Z_{k-1}) → (2+k)×(2+k) blocks.
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
    scale = jnp.maximum(jnp.mean(jnp.diagonal(sub, axis1=-2, axis2=-1), axis=-1), 1e-8)
    sub = sub + (_RIDGE + 1e-6 * scale)[..., None, None] * eye
    precision = jnp.linalg.inv(sub)
    denom = jnp.sqrt(jnp.maximum(precision[..., 0, 0] * precision[..., 1, 1], 0.0))
    rho = jnp.where(denom > 0.0, -precision[..., 0, 1] / denom, 0.0)
    return rho


def _empty_z_partial_corr(Cov: jnp.ndarray) -> jnp.ndarray:
    """Marginal / empty-``Z`` partial correlation for all pairs (2×2)."""
    n = Cov.shape[0]
    z = jnp.zeros((n, n, 0), dtype=jnp.int32)
    valid = jnp.zeros((n, n, 0), dtype=bool)
    return _pairwise_partial_corr(Cov, z, valid, k=0)


def _anneal_threshold(start: float, end: float, progress: float) -> jnp.ndarray:
    """Linear interpolate ``start → end`` for ``progress ∈ [0, 1]``."""
    p = jnp.clip(jnp.asarray(progress, dtype=jnp.float64), 0.0, 1.0)
    start_a = jnp.asarray(start, dtype=jnp.float64)
    end_a = jnp.asarray(end, dtype=jnp.float64)
    return start_a + p * (end_a - start_a)


def _correlation_matrix(Cov: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation matrix from a covariance (diagonal-safe)."""
    std = jnp.sqrt(jnp.maximum(jnp.diag(Cov), 1e-12))
    return Cov / (std[:, None] * std[None, :])


def _mean_neighbor_elevations(
    W: jnp.ndarray,
    E: jnp.ndarray,
    *,
    empty_parent: jnp.ndarray | float = 1.0,
    empty_child: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parent/child mean elevations under directed weight matrix ``W``.

    Missing parents default to ``empty_parent`` (1.0 anchors roots at the top);
    missing children default to ``empty_child`` (0.0 anchors leaves at the bottom).
    Pass ``E`` for both empties when empty neighborhoods must contribute zero pull
    (e.g. 2-hop gravity).
    """
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
) -> jnp.ndarray:
    """Absolute elevation target for ``E ← E + lr * (target - E)``.

    Roots (``in_deg == 0``) anchor the missing parent reference at 1.0; leaves
    (``out_deg == 0``) anchor the missing child reference at 0.0. Isolated nodes
    keep their current elevation (no drift).

    With ``multihop=True``, influences combine as::

        total_pull = local_pull + hop_discount * two_hop_pull

    where ``two_hop`` uses ``A @ A`` (diagonal cleared). Empty 2-hop
    neighborhoods contribute zero pull.
    """
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    mean_parents, mean_children = _mean_neighbor_elevations(
        A, E, empty_parent=1.0, empty_child=0.0
    )
    has_edges = (in_deg + out_deg) > 0
    local_mean = jnp.where(has_edges, 0.5 * (mean_parents + mean_children), E)
    if not multihop:
        return local_mean

    A2 = A @ A
    A2 = A2.at[jnp.diag_indices(A.shape[0])].set(0.0)
    # Empty 2-hop neighborhoods → fallback E ⇒ zero pull.
    mean_p2, mean_c2 = _mean_neighbor_elevations(
        A2, E, empty_parent=E, empty_child=E
    )
    two_hop_mean = (mean_p2 + mean_c2) * 0.5
    discount = jnp.asarray(hop_discount, dtype=jnp.float64)
    return local_mean + discount * (two_hop_mean - E)


def _v_structure_masks(
    A: jnp.ndarray,
    Cov: jnp.ndarray,
    threshold_death: float | jnp.ndarray,
    *,
    fuzzy: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Unshielded explaining-away triples and collider / parent masks.

    Axis convention for the triple tensor: ``(i, k, j)`` with middle node ``k``.

    Returns
    -------
    is_v_triple, is_collider_k, is_v_parent_i
        ``is_v_triple`` shape ``(N, N, N)``; the others shape ``(N,)``.
    """
    n = A.shape[0]
    eye = jnp.eye(n, dtype=bool)
    if fuzzy:
        U = (jnp.maximum(A, A.T) > 0) & (~eye)
    else:
        U = ((A + A.T) > 0) & (~eye)

    R = _correlation_matrix(Cov)
    thr = jnp.asarray(threshold_death, dtype=R.dtype)

    # Unshielded wedge i — k — j (no direct i—j).
    wedge = (
        U[:, :, None]
        & U.T[None, :, :]
        & (~U[:, None, :])
        & (~eye[:, None, :])
    )

    r_ij = R[:, None, :]
    r_ik = R[:, :, None]
    r_jk = R.T[None, :, :]
    denom = jnp.sqrt(jnp.maximum((1.0 - r_ik**2) * (1.0 - r_jk**2), 1e-12))
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
    """Boolean mask of nodes detected as V-structure colliders (explaining-away)."""
    A = jnp.asarray(A, dtype=jnp.float64)
    Cov = jnp.asarray(Cov, dtype=jnp.float64)
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
    """Relative V-well: sink colliders below parents; gently buoy parents."""
    lr = jnp.asarray(learning_rate, dtype=E.dtype)
    is_v_triple, is_collider_k, is_v_parent_i = _v_structure_masks(
        A, Cov, threshold_death, fuzzy=fuzzy
    )

    # parent_mask[i, k] = True if some j makes (i, k, j) a V-triple.
    parent_mask = jnp.any(is_v_triple, axis=2)
    masked_parent_E = jnp.where(parent_mask, E[:, None], 1.0)
    min_parent_E = jnp.min(masked_parent_E, axis=0)

    v_target_k = jnp.maximum(min_parent_E - _V_WELL_GAP, 0.0)
    E = jnp.where(is_collider_k, E + 2.0 * lr * (v_target_k - E), E)
    E = jnp.where(
        is_v_parent_i & (~is_collider_k),
        jnp.minimum(E + 0.5 * lr, 1.0),
        E,
    )
    return jnp.clip(E, 0.0, 1.0)


def _apply_gravity_projection(
    A: jnp.ndarray,
    E: jnp.ndarray,
    *,
    fuzzy: bool,
    death_rate: jnp.ndarray,
) -> jnp.ndarray:
    """Downhill-only projection with deterministic elevation tie-break."""
    n = A.shape[0]
    # Tiny index jitter so clipped nodes at 0.0 / 1.0 never tie.
    e_tie = E + 1e-6 * (jnp.arange(n, dtype=E.dtype) / jnp.maximum(n, 1))
    gravity_ok = e_tie[:, None] > e_tie[None, :]
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
    """One Bayesian CA tick (unjitted body): death → birth → v_well → gravity."""
    A = jnp.asarray(A, dtype=jnp.float64)
    E = jnp.asarray(E, dtype=jnp.float64).reshape(-1)
    Cov = jnp.asarray(Cov, dtype=jnp.float64)
    n = A.shape[0]
    birth_rate = jnp.asarray(birth_rate, dtype=jnp.float64)
    death_rate = jnp.asarray(death_rate, dtype=jnp.float64)
    lr = jnp.asarray(learning_rate, dtype=jnp.float64)

    if anneal:
        death_t = _anneal_threshold(threshold_death_start, threshold_death, progress)
        birth_t = _anneal_threshold(threshold_birth_start, threshold_birth, progress)
    else:
        death_t = jnp.asarray(threshold_death, dtype=jnp.float64)
        birth_t = jnp.asarray(threshold_birth, dtype=jnp.float64)

    U = _undirected_neighbors(A, fuzzy=fuzzy)
    Z, valid = _pairwise_topk_Z(U, Cov, k, E)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)

    # --- 1. Death ------------------------------------------------------------
    existing = A > 0
    kill = existing & (jnp.abs(rho) < death_t)
    if fuzzy:
        A = jnp.where(kill, jnp.maximum(0.0, A - death_rate), A)
    else:
        A = jnp.where(kill, 0.0, A)

    U = _undirected_neighbors(A, fuzzy=fuzzy)
    Z, valid = _pairwise_topk_Z(U, Cov, k, E)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)

    # --- 2. Birth: elevation-oriented; fuzzy may reinforce existing mass ----
    if fuzzy:
        candidates = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    else:
        undirected_absent = (A + A.T) == 0
        undirected_absent = undirected_absent.at[jnp.diag_indices(n)].set(False)
        candidates = jnp.triu(undirected_absent, k=1)
    strong = candidates & (jnp.abs(rho) > birth_t)
    forward = strong & (E[:, None] > E[None, :])
    backward = strong & (E[None, :] > E[:, None])
    if fuzzy:
        A = jnp.where(forward, jnp.minimum(1.0, A + birth_rate), A)
        A = jnp.where(backward.T, jnp.minimum(1.0, A + birth_rate), A)
        A = jnp.where(forward.T, jnp.maximum(0.0, A - death_rate), A)
        A = jnp.where(backward, jnp.maximum(0.0, A - death_rate), A)
    else:
        A = jnp.where(forward, 1.0, A)
        A = jnp.where(backward.T, 1.0, A)

    # --- 3. V-structure gravity well (collider orientation) ------------------
    E = _vstructure_gravity_well(
        A, E, Cov, learning_rate, death_t, fuzzy=fuzzy
    )

    # --- 4. Gravity: elevation drift, then downhill projection --------------
    target = _elevation_gravity_target(
        A, E, multihop=multihop, hop_discount=hop_discount
    )
    E = jnp.clip(E + lr * (target - E), 0.0, 1.0)
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
    """One Bayesian CA tick (JAX): death, birth, V-well, then gravity.

    Pure functional; wrapped in ``@jax.jit``. ``k``, ``anneal``, ``fuzzy``,
    and ``multihop`` are static. Defaults keep the historical binary,
    1-hop CA with static thresholds.

    Parameters
    ----------
    A:
        Adjacency ``(N, N)``. Binary by default; continuous in ``[0, 1]``
        when ``fuzzy=True``.
    E:
        Node elevations ``(N,)`` in ``[0, 1]``.
    Cov:
        Empirical covariance ``(N, N)`` of the observations.
    learning_rate:
        Elevation drift step size toward the gravity target.
    threshold_death:
        Kill / decay edges with ``|rho|`` below this value (end of anneal).
    threshold_birth:
        Birth / reinforce edges with ``|rho|`` above this value (end of anneal).
    k:
        Conditioning-set size for death/birth partial correlations (static).
        Batched inversions use ``(2+k)×(2+k)`` blocks; raise via the
        benchmark / synth ``--k`` CLI (Upgrade 2).
    anneal:
        If ``False`` (default), ignore ``progress`` / ``*_start`` and use
        the end thresholds as fixed gates. If ``True``, linearly
        interpolate from ``*_start`` to the end thresholds.
    progress:
        Annealing clock in ``[0, 1]`` (tick 0 → 0.0, final tick → 1.0).
        Only used when ``anneal=True``.
    threshold_death_start, threshold_birth_start:
        Early-tick endpoints when ``anneal=True``.
    fuzzy:
        If ``False`` (default), adjacency stays binary (hard birth/death).
        If ``True``, ``A`` is continuous in ``[0, 1]``: birth adds
        ``birth_rate``, death subtracts ``death_rate``, and gravity /
        V-wells use fractional undirected weights. Callers should
        binarize only after the final tick.
    birth_rate, death_rate:
        Soft update sizes when ``fuzzy=True`` (ignored in binary mode).
    multihop:
        If ``True``, elevation drift also pulls toward 2-hop parent/child
        means from ``A @ A``, scaled by ``hop_discount`` (Upgrade 4 /
        Maximum Accuracy Mode).
    hop_discount:
        Weight on the 2-hop elevation target when ``multihop=True``.

    Returns
    -------
    A_new, E_new
        Updated adjacency and elevations (pure; inputs are not mutated).
    """
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
    """Run ``ticks`` CA steps via ``jax.lax.scan`` (JIT; ``ticks``/``k`` static).

    ``ticks`` must be a Python ``int`` (compile-time constant) because it is
    the scan length. ``k`` is static because pairwise partial-correlation
    blocks have shape ``(2+k, 2+k)``.
    """
    A0 = jnp.asarray(A0, dtype=jnp.float64)
    E0 = jnp.asarray(E0, dtype=jnp.float64).reshape(-1)
    Cov = jnp.asarray(Cov, dtype=jnp.float64)

    if ticks <= 0:
        return A0, E0

    denom = max(ticks - 1, 1)

    def body(carry, t):
        A, E = carry
        progress = jnp.asarray(t, dtype=jnp.float64) / jnp.asarray(
            denom, dtype=jnp.float64
        )
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
    A = jnp.asarray(A, dtype=jnp.float64)
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
    A = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1) * 0.0
    E = jnp.linspace(1.0, 0.0, n, dtype=jnp.float64)
    Cov = jnp.eye(n, dtype=jnp.float64)
    # First call triggers JIT compile.
    step(A, E, Cov, k=k, fuzzy=fuzzy, multihop=multihop)
    return device_info()
