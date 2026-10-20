Command-line tools
==================

The package registers console scripts and supports ``python -m`` entry
points.

Synthetic DAG harness
---------------------

.. code-block:: bash

   python -m causal_automata.bayesian_synth [options]
   # or: causal-automata-synth

Builds a known linear-Gaussian SEM, estimates covariance, initializes a
random upper-triangular adjacency plus random elevations, runs the CA for
a fixed number of ticks, and prints directed / skeleton recovery metrics.

Useful flags
^^^^^^^^^^^^

======================== ====================================================
Flag                     Description
======================== ====================================================
``--n``                  Number of nodes (default ``10``; Phase-2 motif).
``--backend``            ``numpy`` (default) or ``jax``.
``--ticks``              Simulation length (default ``100``).
``--seed`` / ``--seeds`` Single seed, or aggregate over ``0..seeds-1``.
``--k``                  Conditioning-set size. Default is **scale-aware**:
                         ``k=3`` if ``N≤50``, ``k=5`` if ``N>50``.
``--threshold-death``    Kill edges with :math:`|\rho|` below this.
``--threshold-birth``    Birth edges with :math:`|\rho|` above this.
``--learning-rate``      Elevation drift step size.
``--target-edges``       True-edge count for sparse DAGs when ``n != 10``.
======================== ====================================================

Examples::

   python -m causal_automata.bayesian_synth --n 10
   python -m causal_automata.bayesian_synth --n 50 --backend jax
   python -m causal_automata.bayesian_synth --n 100 --backend jax --seeds 5

Benchmark vs PC / GOLEM
-----------------------

.. code-block:: bash

   python -m causal_automata.benchmark [options]
   # or: causal-automata-benchmark

Requires the ``gcastle`` and ``jax`` extras for full tables. Default scale
is ``N=100`` with GOLEM ``num_iter=20000``. Use ``--ca-only`` to time the
JAX CA without PC / GOLEM.

Core flags
^^^^^^^^^^

===================== ========================================================
Flag                  Description
===================== ========================================================
``--n``               Nodes (default ``100``).
``--ticks``           CA ticks (default ``100``).
``--k``               Conditioning-set size. **Default is dynamic**:
                      ``k=3`` when ``N≤50``, ``k=5`` when ``N>50``
                      (Upgrade 2). Larger ``k`` builds ``(2+k)×(2+k)``
                      batched inversions under ``@jax.jit``.
``--k-sweep``         Comma-separated ``k`` values on one SEM (e.g.
                      ``3,4,5``); labels each CA row.
``--ca-only``         Skip PC / GOLEM; score the JAX CA only.
``--golem-iters``     GOLEM Adam budget (default ``20000``).
``--smoke``           Device checks + GOLEM init only (no full learn).
===================== ========================================================

Upgrade flags (all **opt-in**; defaults stay binary 1-hop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================= ======================================================
Flag                    Description
======================= ======================================================
``--anneal``            **Upgrade 1.** Linear threshold annealing over
                        progress :math:`\in [0,1]` (stricter early, looser
                        late). Default remains static
                        ``death=0.05`` / ``birth=0.10``.
``--fuzzy``             **Upgrade 3.** Continuous edge weights
                        :math:`A_{ij}\in[0,1]` with soft birth/death;
                        binarize only after the final tick
                        (``A > 0.5``). Experimental — did not beat binary
                        on the default N=100 SEM.
``--compare-fuzzy``     A/B binary vs ``--fuzzy`` on the same SEM.
``--birth-rate`` /      Soft increments when ``--fuzzy`` is set
``--death-rate``        (defaults ``0.5`` / ``0.1``).
``--multihop``          **Upgrade 4 — Maximum Accuracy Mode.** 2-hop
                        elevation gravity via :math:`A@A`, weighted by
                        ``--hop-discount`` (default ``1.0``). Breaks
                        Markov-equivalence ties; on N=100 seed=0 k=5
                        lifts Directed F1 **0.619 → 0.696** with flat
                        wall time.
``--compare-multihop``  A/B 1-hop vs ``--multihop`` on the same SEM.
``--hop-discount``      Weight on the 2-hop elevation pull (default
                        ``1.0``).
======================= ======================================================

Examples::

   # Default N=100, k=5, binary 1-hop baseline
   python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100

   # Maximum Accuracy Mode
   python -m causal_automata.benchmark --n 100 --ca-only --k 5 --multihop
   python -m causal_automata.benchmark --n 100 --ca-only --k 5 --compare-multihop

   # Conditioning sweep / opt-in experiments
   python -m causal_automata.benchmark --n 100 --ca-only --k-sweep 3,4,5
   python -m causal_automata.benchmark --n 100 --ca-only --compare-fuzzy
   python -m causal_automata.benchmark --n 50 --ca-only --anneal

   python -m causal_automata.benchmark --smoke
