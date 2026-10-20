Physics rules and matrix mechanics
==================================

The automaton state at tick :math:`t` is the adjacency
:math:`A^{(t)} \in \mathbb{R}^{N \times N}` together with elevations
:math:`E^{(t)} \in [0, 1]^N`. The observational environment supplies
static forces via covariance / correlation, mutual information, and
conditional mutual information (approximated here by partial correlation
from the empirical covariance for linear-Gaussian data).

Paper-level forces
------------------

The conceptual transition (Foundations / Intuition papers) is

.. math::

   A^{(t+1)} = \sigma\bigl(A^{(t)} + \alpha G + \beta V - \gamma D\bigr),

where :math:`G` is gravity (birth of strong links), :math:`V` compiles
V-well / sink forces, :math:`D` is death (pruning), and :math:`\sigma`
clips to a valid range. Weights :math:`\alpha, \beta, \gamma` balance the
forces.

Gravity (birth skeleton)
^^^^^^^^^^^^^^^^^^^^^^^^

High mutual information seeds undirected connections:

.. math::

   G_{ij} =
   \begin{cases}
   1 & \text{if } I(X_i; X_j) > \tau_{\mathrm{gravity}} \\
   0 & \text{otherwise}
   \end{cases}

Analogy: stars forming from dense gas — the strongest correlations form
the skeleton.

V-wells (collider / sink detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For triplets :math:`(i, j, k)`,

.. math::

   V_{ijk} = I(X_i; X_j \mid X_k) - I(X_i; X_j).

If conditioning on :math:`k` *increases* dependence, :math:`k` is a sink.
The rule pushes :math:`A_{ik}, A_{jk}` toward 1 and suppresses reverse
arrows — locking in :math:`i \to k \leftarrow j`.

Death (pruning)
^^^^^^^^^^^^^^^

If any separating set :math:`Z` makes :math:`i` and :math:`j` independent,

.. math::

   D_{ij} = \mathbf{1}\Bigl(\min_Z I(X_i; X_j \mid Z) < \tau_{\mathrm{death}}\Bigr).

False transitive edges (ice cream ↔ sharks) are erased.

Implemented tick (code)
-----------------------

The NumPy prototype (:func:`causal_automata.bayesian.step`) and the JAX
port (:func:`causal_automata.jax_core.step`) share one operational
physics adapted for batched partial correlation:

1. **Gravity (topography)** — Keep only downhill edges
   (:math:`E_i > E_j`). Drift each elevation toward the midpoint of its
   parent and child means (learning rate). Isolated sides use the node's
   own elevation (no pull).
2. **V-structure gravity well** — Absent undirected pairs that share
   neighbors and are marginally independent
   (:math:`|\rho| < \tau_{\mathrm{death}}` with empty :math:`Z`) mark
   shared neighbors as colliders. Those elevations sink by
   ``2 * learning_rate``, then clip to :math:`[0, 1]`. Later birth
   prefers arrows into the collider.
3. **Death** — For existing edges, estimate partial correlation given the
   top-``k`` union neighbors ranked by
   :math:`\max(|\mathrm{Cov}_{m i}|, |\mathrm{Cov}_{m j}|)`. Kill when
   :math:`|\rho| < \tau_{\mathrm{death}}`.
4. **Birth** — For all absent undirected pairs, birth when
   :math:`|\rho| > \tau_{\mathrm{birth}}`, orienting by elevation
   (equal elevation → no-op). Re-apply downhill clear after sinks.

Default API thresholds are ``threshold_death=0.05``,
``threshold_birth=0.15``, ``k=2``, ``learning_rate=0.1``. The synthetic /
benchmark CLIs typically use ``threshold_birth=0.10`` and a
**scale-aware** ``k`` (``3`` if :math:`N\le 50`, ``5`` if :math:`N>50`).
Optional ``multihop=True`` adds discounted 2-hop elevation pull from
:math:`A@A` (Maximum Accuracy Mode).

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
Gravity       Downhill mask + optional :math:`A@A`    Element-wise / ``matmul``
V-wells       Pair + shared-neighbor masks            Boolean reductions; elevation sinks
Death / birth Ranked :math:`Z`, :math:`\rho`          Batched ``linalg.inv`` ``(2+k)``
DAG law       Elevations                              Hard downhill constraint
============= ======================================= =====================================

``@jax.jit`` removes Python loop overhead so hundreds of ticks over
:math:`N=100` complete in well under a second on a Tesla T4 (see
:doc:`benchmarks`).
