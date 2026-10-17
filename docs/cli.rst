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
``--k``                  Conditioning-set cap (CLI default ``3``).
``--threshold-death``    Kill edges with :math:`|\rho|` below this.
``--threshold-birth``    Birth edges with :math:`|\rho|` above this.
``--learning-rate``      Elevation drift step size.
``--target-edges``       True-edge count for sparse DAGs when ``n != 10``.
======================== ====================================================

Examples::

   python -m causal_automata.bayesian_synth --n 10
   python -m causal_automata.bayesian_synth --n 50 --backend jax
   python -m causal_automata.bayesian_synth --n 100 --seeds 5

Benchmark vs PC / GOLEM
-----------------------

.. code-block:: bash

   python -m causal_automata.benchmark [options]
   # or: causal-automata-benchmark

Requires the ``gcastle`` and ``jax`` extras. Default scale is ``N=100``
with GOLEM ``num_iter=20000``.

Useful flags
^^^^^^^^^^^^

================== ====================================================
Flag               Description
================== ====================================================
``--n``            Nodes (default ``100``).
``--ticks``        CA ticks (default ``100``).
``--k``            Conditioning-set cap (default ``3``).
``--golem-iters``  GOLEM Adam budget (default ``20000``).
``--smoke``        Device checks + GOLEM init only (no full learn).
================== ====================================================

Examples::

   python -m causal_automata.benchmark
   python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100
   python -m causal_automata.benchmark --smoke
