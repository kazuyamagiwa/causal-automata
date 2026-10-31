Theory: causal discovery as automata
====================================

This page summarizes the conceptual foundations behind
**causal-automata**, drawn from the project papers *Foundations of Causal
Automata*, *Causal Automata: From Intuition to GPU Physics*, and
*Bayesian Automata and Network Evolution* (Yamagiwa, 2026). See
:doc:`bibliography` for citations.

Correlation vs. causation
-------------------------

Observational data only gives correlations. Ice-cream sales and shark
attacks both rise in July; a naive model might conclude that ice cream
causes sharks. Causal discovery aims to recover the *mechanism* — here a
hidden common cause (summer temperature).

We represent mechanisms as a **Directed Acyclic Graph (DAG)**:

* **Nodes** — measured variables.
* **Directed edges** — causal influence (:math:`A \to B` means :math:`A`
  causes :math:`B`).
* **Acyclic** — no directed cycles; time does not loop.

The software starts from a blank (or noisy) map of variables and draws
arrows by evolving an adjacency state under local rules.

Independence, forks, and colliders
----------------------------------

Dependence is measured with mutual information (MI). High MI means the
variables are related; near-zero MI means independence. **Conditional
mutual information** (CMI) :math:`I(X_i; X_j \mid X_k)` asks whether
knowing :math:`X_i` still informs :math:`X_j` after observing
:math:`X_k`.

Fork (common cause)
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Temperature
      /     \
     v       v
   Ice cream  Shark attacks

:math:`B` and :math:`C` look dependent, but :math:`I(B; C \mid A) \approx 0`
once the common cause is known. Conditioning *blocks* the path.

Collider (V-structure)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Rain          Sprinkler
     \             /
      v           v
       Grass wet

Rain and sprinkler are independent a priori. Observing wet grass
*opens* dependence between them. The signature

.. math::

   V_{ijk} = I(X_i; X_j \mid X_k) - I(X_i; X_j)

is large and positive when :math:`k` is a collider (sink). The automaton
uses this as a downward force — the **V-well**.

From Pearl to grids
-------------------

Judea Pearl’s belief propagation replaced global combinatorial
evaluation with local message passing: parents send causal evidence
(:math:`\pi`) and children send diagnostic evidence (:math:`\lambda`).
Cellular automata share the same engine — gather neighbors, aggregate,
update — but traditionally aggregate by neighbor counts on a lattice.

Structurally, a 2D grid is an adjacency matrix with fixed degree. Belief
propagation, shortest-path search, and CA differ mainly in the
aggregation function (products of probabilities, min-plus distances, or
active-neighbor counts).

Generative topology and Bayesian automata
-----------------------------------------

Classical network analysis fixes the graph and routes information
through it. A Bayesian Automaton inverts that: CA rules act on the
**adjacency matrix itself**. Edges are born, survive, or die according
to predictive power over an external dataset rather than pure geometry.

Updating every edge in parallel would create cycles. The implementation
assigns each node a continuous **elevation** :math:`E \in [0, 1]` and
enforces a hard rule: edges may only flow downhill. Because arrows
cannot point upward, cycles are impossible.

Parallel physics as an alternative to sequential search
-------------------------------------------------------

Constraint-based methods (PC) run sequential independence tests.
Continuous optimizers (GOLEM / NOTEARS) score DAGs with gradient
descent. Both are accurate in theory; their cost grows when many pairs
and conditioning sets are evaluated one at a time.

The Causal Automaton reframes structure learning as a physical
simulation: map variables to a grid, define local forces (gravity,
sinks, birth, death), and let the DAG emerge from parallel state
updates. On GPU, the transition is dense linear algebra — see
:doc:`physics` and :doc:`benchmarks`.

PC-consistent Sepset screening in the automaton
-----------------------------------------------

Classical PC builds a separating set :math:`S` for :math:`(i,j)` from
**one** endpoint's adjacency list at a time
(:math:`\mathrm{adj}(i)\setminus\{j\}` **or**
:math:`\mathrm{adj}(j)\setminus\{i\}`). Mixing one-sided children from
**both** endpoints into a single conditioning set can open Berkson paths
and delete true edges (historically ``4\to 13``).

v0.3.6 mirrors that discipline inside the cellular update:

* **``s=1``** — closed-form partial correlation against **every** current
  skeleton neighbor (not only the top-1 score).
* **``s≥2``** — endpoint-local top-``k`` pools; all size-``s``
  combinations; Berkson-inflating colliders excluded from :math:`Z`.
* **V-well** — a triple is a collider only if explaining-away holds
  **and** no second node screens the pair (2-node Sepset recursion).

Together with sample-size-adaptive Fisher-z thresholds
(:func:`~causal_automata.fisher_z_threshold`) and CPDAG conversion
(:func:`~causal_automata.dag_to_cpdag`), these rules lift Pattern F1 on
the Benchpress ER ``p=20``, ``n=1000``, 5-seed suite to
``0.933 ± 0.075`` (SHD ``3.4 ± 4.4``) — see :doc:`benchmarks`.
