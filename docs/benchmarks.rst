Benchmarks
==========

Reported figures (Google Colab **Tesla T4**, GOLEM ``num_iter=20000``,
synthetic linear DAG with ``N=100``):

=================== ============= ============= ==============
Algorithm           Skeleton F1   Directed F1   Wall time (s)
=================== ============= ============= ==============
PC (gCastle)        0.968         0.813         528.631
GOLEM (gCastle)     0.577         0.403         251.182
Bayesian CA (JAX)   0.812         0.483         0.323
=================== ============= ============= ==============

In this setup, Bayesian CA completes in ``0.323s``, compared with
``251.182s`` for GOLEM and ``528.631s`` for PC. Skeleton / directed F1
are ``0.812`` / ``0.483`` for Bayesian CA, ``0.577`` / ``0.403`` for
GOLEM, and ``0.968`` / ``0.813`` for PC.

Reproduce
---------

.. code-block:: bash

   pip install -e ".[jax,gcastle]"
   # Colab GPU example:
   #   pip install -U "jax[cuda13]" gcastle torch
   #   pip install -e .

   python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100
   python -m causal_automata.benchmark --smoke   # device + GOLEM init only

The harness shares SEM helpers with
:mod:`causal_automata.bayesian_synth` (:func:`~causal_automata.bayesian_synth.prepare_synthetic_bundle`).
PC uses gCastle ``variant="stable"`` with Fisher-Z tests. GOLEM uses the
project defaults in :data:`causal_automata.benchmark.GOLEM_DEFAULT_KWARGS`
(no early-stop in gCastle — fixed Adam budget only). The CA row warms up
JIT outside the timed window.

Interpreting metrics
--------------------

* **Directed F1** — precision/recall on oriented edges vs ground truth.
* **Skeleton F1** — same on undirected pairs (either orientation counts).
* **Wall time** — algorithm ``learn`` / CA step loop only (data generation
  excluded; CA compile excluded).

PC remains a strong accuracy baseline when runtime is acceptable. The
automaton is aimed at settings where parallel adjacency updates are a
useful alternative to sequential CI testing or long Adam runs.
