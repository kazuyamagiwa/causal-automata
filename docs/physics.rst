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
2. **V-well** — Acts on **elevations**, not adjacency. An unshielded
   triple :math:`i{-}k{-}j` is a statistical collider when conditioning
   on :math:`k` inflates partial correlation **and** no second node
   :math:`m` screens :math:`(i,j)` given :math:`\{k,m\}` (closed-form
   2-node Sepset / backdoor check). True collider centers sink below
   :math:`\min(E_i,E_j)-\delta`; non-collider unshielded centers are
   raised (anti-sink) so they cannot end as local elevation minima.
   After V-well moves :math:`E`, later birth / downhill projection
   prefer arrows into sinks (:math:`i \to k \leftarrow j`).
3. **Death** — Progressive schedule
   :math:`s_{\max}=\min(t+1,k)`. At ``s=1``, an **exact closed-form**
   screen over **all** current skeleton neighbors
   :math:`(U_{im}\lor U_{jm})` kills the edge if any single
   :math:`|\rho(i,j\mid m)| < \tau_{\mathrm{death}}` (zero matrix
   inversions). For ``s≥2``, PC-consistent **endpoint-local** top-``k``
   pools are built separately from :math:`U(i)\setminus\{j\}` and
   :math:`U(j)\setminus\{i\}` (never mixing one-sided children across
   endpoints), and **all** size-``s`` combinations within each pool are
   tested. Berkson-inflating colliders
   (:math:`|\rho(i,j\mid m)| > |R_{ij}| + 0.02`) are excluded from
   separator pools so moral ("married-parent") FPs cannot ride a shared
   child.
4. **Birth** — For absent undirected pairs, require
   :math:`|R_{ij}| \ge \tau_{\mathrm{death}}` **and**
   :math:`|\rho| > \tau_{\mathrm{birth}}` under the same Sepset gates,
   orienting with a deterministic elevation tie-break so nodes clipped
   together at ``0`` / ``1`` still get a direction. Tie-breaks use
   Cov-derived node keys. A final downhill projection of the surviving
   undirected skeleton :math:`U=\max(A,A^\top)` keeps the DAG law and
   reverses mid-tick elevation flips instead of wiping edges.

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
``threshold_birth=0.15``, ``k=2``, ``learning_rate=0.1``. When fitting
from samples via :class:`~causal_automata.CausalAutomaton`, thresholds
are calibrated with
:func:`~causal_automata.fisher_z_threshold`
(:math:`\tau_{\mathrm{death}}=\tanh(z_{1-\alpha/2}/\sqrt{n-k-3})`,
:math:`\tau_{\mathrm{birth}}=1.2\,\tau_{\mathrm{death}}`). The synthetic /
benchmark CLIs typically use a **scale-aware** ``k``
(``3`` if :math:`N\le 50`, ``5`` if :math:`N>50`).

``ticks`` (scan length in ``run``) and ``k`` (block size ``2+k``) are JIT
**static** arguments — see :doc:`jax_static`.

Closed-form Sepset algebra (v0.3.5–v0.3.6)
------------------------------------------

The 1-node partial correlation used for ``s=1`` screens and explaining-
away tests is

.. math::

   \rho(i,j\mid m)
   = \frac{R_{ij}-R_{im}R_{jm}}
          {\sqrt{(1-R_{im}^2)(1-R_{jm}^2)}}.

The 2-node recursion (Sepset / backdoor check; zero inversions) is

.. math::

   \rho(i,j\mid k,m)
   = \frac{\rho(i,j\mid m)-\rho(i,k\mid m)\,\rho(j,k\mid m)}
          {\sqrt{\bigl(1-\rho(i,k\mid m)^2\bigr)
                 \bigl(1-\rho(j,k\mid m)^2\bigr)}}.

Complexity
----------

* **Downhill acyclicity projection** is :math:`O(N^2)` and avoids
  :math:`O(N^3)` matrix-exponential / spectral DAG penalties.
* Exact ``s=1`` screens are :math:`O(N^3)` tensor reductions with **no**
  matrix inversions. ``s≥2`` still builds dense pair×candidate score
  tensors and batched ``(2+s)×(2+s)`` inversions over endpoint-local
  combinations, so wall-clock work scales closer to :math:`O(N^3)` at
  large ``N``.

GPU / accelerator matrix computation
------------------------------------

For continuous linear-Gaussian data, CMI relates to partial correlation.
The engine precomputes a global covariance / correlation and, each tick,
forms closed-form ``s=1`` tensors plus batched sub-covariances for
``s≥2`` endpoint-local combos. Inverting those blocks yields precision
matrices whose off-diagonals give :math:`\rho`.

============= ======================================= =====================================
Component     Structure                               JAX / NumPy operation
============= ======================================= =====================================
State         :math:`N \times N`                      Array update / ``jax.lax``-friendly
Death ``s=1`` All-neighbor :math:`\rho(\cdot\mid m)`  Closed-form tensor, no ``inv``
Death ``s≥2`` Endpoint-local combos                   Batched ``linalg.inv`` ``(2+s)``
V-wells       Explaining-away + Sepset + anti-sink    Boolean / soft updates on ``E``
Gravity       Local-margin means + :math:`U` downhill Element-wise / optional ``A@A``
DAG law       Elevations                              Hard downhill + bound tie-break
============= ======================================= =====================================

``@jax.jit`` removes Python loop overhead so hundreds of ticks over
:math:`N=100` complete in well under a second on a Tesla T4 (see
:doc:`benchmarks`).
