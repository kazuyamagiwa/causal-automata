Physics rules and matrix mechanics
==================================

The automaton state at tick :math:`t` is the adjacency
:math:`A^{(t)} \in \mathbb{R}^{N \times N}` together with elevations
:math:`E^{(t)} \in [0, 1]^N`. The observational environment supplies
static forces via the empirical covariance; conditional dependence is
approximated by **partial correlation** (linear-Gaussian proxy for CMI).

Paper-level forces (conceptual)
-------------------------------

Early manuscripts write a single additive update

.. math::

   A^{(t+1)} = \sigma\bigl(A^{(t)} + \alpha G + \beta V - \gamma D\bigr),

where :math:`G` seeded undirected links from mutual information, :math:`V`
compiled collider / sink forces on **edges**, and :math:`D` pruned weak
links. That formula is a **conceptual** sketch only. The shipped engine
implements four named stages with different targets (see below):
**Birth** (not “Gravity”) adds conditional-correlation edges;
**V-well** moves the elevation field :math:`E` (not :math:`A` directly);
**Gravity** is elevation drift plus downhill projection;
**Death** is a fixed top-``k`` partial-correlation prune — a fast
heuristic CI test, **not** an exhaustive :math:`\min_Z` search as in PC.

Implemented tick (code)
-----------------------

The JAX and NumPy ports
(:func:`causal_automata.jax_core.step`,
:func:`causal_automata.bayesian.step`) share one operational order:

1. **Gravity (topography)** — Project to downhill edges with a raw
   :math:`E_i > E_j` compare, then drift each elevation toward a
   **local-margin** target: midpoint of parent/child means when both
   sides exist; roots sit just above mean children
   (:math:`+\delta`, default :math:`\delta=0.05`); leaves sit just below
   mean parents; isolates hold :math:`E`. There are **no** hard anchors
   at ``1.0`` / ``0.0`` (those saturated random inits and reversed
   chains). Optional ``multihop=True`` also pulls toward 2-hop means
   from :math:`A@A`.
2. **V-well** — Acts on **elevations**, not adjacency. Shared neighbors
   of marginally independent absent pairs (on the evolving undirected
   adjacency **and** a static ``|R|`` correlation skeleton) are treated
   as collider candidates and lowered with a multiplicative decay
   :math:`E \leftarrow E\,(1-2\alpha)`. This replaces the older additive
   ``E - 2\alpha`` slam into the ``E=0`` clip wall, which destroyed
   near-floor orderings and drove even/odd SHD limit cycles. Separately,
   :func:`causal_automata.jax_core.detect_colliders` exposes a stricter
   explaining-away (Berkson) test for diagnostics. After V-well moves
   :math:`E`, later birth / downhill projection prefer arrows into
   sinks (:math:`i \to k \leftarrow j`).
3. **Death** — For existing edges, estimate partial correlation given
   the top-``k`` union neighbors ranked by
   :math:`|R_{iz}| + |R_{jz}|` (Pearson correlation, scale-free). Kill
   when :math:`|\rho| < \tau_{\mathrm{death}}`. This is a **heuristic**
   separator (fixed ``k``, ``|R|`` ranking) — not
   :math:`\min_Z I(X_i;X_j\mid Z)`. Raw elevation is **not** used to
   mask candidates (that excluded true confounders under random ``E0``).
4. **Birth** — For absent undirected pairs, birth when
   :math:`|\rho| > \tau_{\mathrm{birth}}`, orienting with a
   deterministic elevation tie-break so nodes clipped together at
   ``0`` / ``1`` still get a direction. Tie-breaks use Cov-derived
   node keys (diag + off-diagonal fingerprint), not raw index order.
   A final bound-aware downhill projection keeps the DAG law.

**Initialization (L4).** Benchmark / synth defaults use a symmetric
correlation skeleton ``A0`` (``|R_{ij}| > \tau_{\mathrm{birth}}``) and
near-flat elevations ``E0 \approx 0.5`` so tick-1 gravity does not lock
arrows into the generator's ``i < j`` labeling. The legacy upper-
triangular random ``A0`` is retained as ``init_mode="upper_random"`` for
the unpermuted H1 regression gate.

**1D elevation embedding.** Downhill-only edges guarantee acyclicity
without matrix-exponential DAG penalties, but they also force a global
scalar ranking of nodes. Wrong elevations can delete true edges until
gravity / V-well reshape ``E``; multihop gravity helps break some
Markov-equivalence ties but does not remove that coupling.

Default API thresholds are ``threshold_death=0.05``,
``threshold_birth=0.15``, ``k=2``, ``learning_rate=0.1``. The synthetic /
benchmark CLIs typically use ``threshold_birth=0.10`` and a
**scale-aware** ``k`` (``3`` if :math:`N\le 50`, ``5`` if :math:`N>50`).

``ticks`` (scan length in ``run``) and ``k`` (block size ``2+k``) are JIT
**static** arguments — see :doc:`jax_static`.

Complexity
----------

* **Downhill acyclicity projection** is :math:`O(N^2)` and avoids
  :math:`O(N^3)` matrix-exponential / spectral DAG penalties.
* A full tick also builds dense pair×candidate score tensors and
  batched ``(2+k)×(2+k)`` inversions, so wall-clock work scales closer
  to :math:`O(N^3)` (with a ``\log N`` factor from per-pair top-``k``
  sorts) at large ``N``.

GPU / accelerator matrix computation
------------------------------------

For continuous linear-Gaussian data, CMI relates to partial correlation.
The engine precomputes a global covariance and, each tick, forms
batched sub-covariances for many edges at once. Inverting those blocks
yields precision matrices whose off-diagonals give :math:`\rho`.

============= ======================================= =====================================
Component     Structure                               JAX / NumPy operation
============= ======================================= =====================================
State         :math:`N \times N`                      Array update / ``jax.lax``-friendly
Death / birth Ranked :math:`Z`, :math:`\rho`          Batched ``linalg.inv`` ``(2+k)``
V-wells       Shared-neighbor independence sinks      Boolean reductions on ``E``
Gravity       Local-margin means + downhill mask      Element-wise / optional ``A@A``
DAG law       Elevations                              Hard downhill + bound tie-break
============= ======================================= =====================================

``@jax.jit`` removes Python loop overhead so hundreds of ticks over
:math:`N=100` complete in well under a second on a Tesla T4 (see
:doc:`benchmarks`).
