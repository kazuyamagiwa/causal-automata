Benchmarks
==========

**causal-automata** v0.3.4 targets dense synthetic linear DAGs at
``N=50``–``N=100`` with a fully ``@jax.jit``-compiled step. The narrative
speedups from the project milestone are on the order of **~5,000×** vs
NOTEARS-class continuous optimization and **~57×** vs PC at **N=50**
scale. Exact wall times depend on hardware (CPU vs Tesla T4) and flags
(``k``, ``--multihop``).

N=100 — Maximum Accuracy Mode
-----------------------------

CPU JAX, ``seed=0``, static thresholds, ``ticks=100``, ``k=5``, binary
edges (Upgrade 4 A/B):

================================== ============= ============= ==============
Algorithm                          Skeleton F1   Directed F1   Wall time (s)
================================== ============= ============= ==============
Bayesian CA (JAX, 1-hop)           0.937         0.619         ~2.9
Bayesian CA (JAX, --multihop)      0.928         0.696         ~2.9
================================== ============= ============= ==============

``--multihop`` is **Maximum Accuracy Mode**: 2-hop elevation gravity via
:math:`A@A` breaks Markov-equivalence ties with essentially flat wall
time versus 1-hop.

N=100 — vs gCastle (Tesla T4)
-----------------------------

Google Colab **Tesla T4**, GOLEM ``num_iter=20000``, historical CA
``k=3`` (pre–Upgrade-2 default):

=================== ============= ============= ==============
Algorithm           Skeleton F1   Directed F1   Wall time (s)
=================== ============= ============= ==============
PC (gCastle)        0.968         0.813         528.631
GOLEM (gCastle)     0.577         0.403         251.182
Bayesian CA (JAX)   0.812         0.483         0.323
=================== ============= ============= ==============

In this setup the CA finishes in **0.323s** vs **251s** (GOLEM) and
**529s** (PC).

Conditioning-set sweep (Upgrade 2)
----------------------------------

Same SEM as above, binary 1-hop, CPU JAX. Default ``k`` is **3** for
``N≤50`` and **5** for ``N>50``.

======= ============= ============= ==============
``k``   Skeleton F1   Directed F1   Wall time (s)
======= ============= ============= ==============
3       0.812         0.483         2.314
4       0.915         0.546         2.481
5       **0.937**     **0.619**     **2.871**
6       0.937         0.619         2.454
7       0.937         0.619         3.049
8       0.937         0.627         3.647
======= ============= ============= ==============

Directed F1 saturates near ``k=5``; no collider-trap collapse through
``k=8`` on this graph.

Opt-in experiments
------------------

* **``--anneal``** (Upgrade 1) — adaptive death/birth thresholds; kept
  opt-in after the proposed schedule regressed F1 vs static gates.
* **``--fuzzy``** (Upgrade 3) — continuous :math:`A\in[0,1]`; best tuned
  run stayed slightly under binary Directed F1 (**0.615** vs **0.619**).
* **``--multihop``** (Upgrade 4) — see Maximum Accuracy Mode above.

Reproduce
---------

.. code-block:: bash

   pip install -e ".[jax,gcastle]"
   # Colab GPU example:
   #   pip install -U "jax[cuda13]" gcastle torch
   #   pip install -e .

   # Baseline N=100 (dynamic k=5)
   python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100

   # Maximum Accuracy Mode
   python -m causal_automata.benchmark --n 100 --ca-only --k 5 --multihop
   python -m causal_automata.benchmark --n 100 --ca-only --k 5 --compare-multihop

   python -m causal_automata.benchmark --smoke   # device + GOLEM init only

The harness shares SEM helpers with
:mod:`causal_automata.bayesian_synth`
(:func:`~causal_automata.bayesian_synth.prepare_synthetic_bundle`).
PC uses gCastle ``variant="stable"`` with Fisher-Z tests. GOLEM uses
:data:`causal_automata.benchmark.GOLEM_DEFAULT_KWARGS` (no early-stop —
fixed Adam budget). The CA row warms up JIT outside the timed window.

Interpreting metrics
--------------------

* **Directed F1** — precision/recall on oriented edges vs ground truth.
* **Skeleton F1** — same on undirected pairs (either orientation counts).
* **Wall time** — algorithm ``learn`` / CA step loop only (data generation
  excluded; CA compile excluded).

PC remains a strong accuracy baseline when runtime is acceptable. The
automaton is aimed at settings where parallel adjacency updates — and
optional 2-hop gravity — are useful alternatives to sequential CI testing
or long Adam runs.
