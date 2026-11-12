Benchmarks
==========

**causal-automata** v0.3.6 is evaluated on two tracks: a full
`Benchpress <https://github.com/felixleopoldo/benchpress>`_ suite on
**NVIDIA A10** (Lambda Cloud; Apptainer ``--nv``) over pcalg Erdős–Rényi
and scale-free DAGs, and the in-repo N=100 linear-Gaussian harness vs
gCastle PC / GOLEM.

Benchpress — NVIDIA A10 full run
--------------------------------

Benchmarks run with Benchpress on NVIDIA A10. Package under test:
``causal-automata==0.3.6`` (GPU/JAX).

Design
~~~~~~

* **Graphs:** pcalg random DAGs — Erdős–Rényi (ER) and scale-free /
  power (SF).
* **Sizes:** ``p ∈ {20, 50, 80, 100}``, average degree ``d=4``,
  ``n=1000`` samples, seeds ``1..10`` → **80 cells**.
* **Data:** continuous SEM (``sem_params`` min=0.25 max=1), standardized
  IID observations.
* **Algorithms:**

  * **Causal Automata (CA)** — this package, GPU/JAX in Apptainer
    ``--nv``
  * **NOTEARS** — ``gcastle_notears`` (linear)
  * **PC** — ``pcalg_pc`` (Gaussian)
  * **GOLEM** — ``gcastle_golem`` (EV)
  * **BiDAG order-MCMC** — hardlimit=16; accuracy ceiling where
    completed

* **Excluded:** FGES / Tetrad — metric fields all NA (not reported).

Hardware fingerprint
~~~~~~~~~~~~~~~~~~~~

* GPU: NVIDIA A10, 23028 MiB, driver 580.105.08
* CPU: 30× Xeon Platinum 8358 @ 2.60GHz; ~222 GiB RAM
* Host: Ubuntu 22.04 / Lambda Stack; Apptainer 1.5.5

Overall results (mean ± std over usable cells)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 10 16 16 16 14

   * - Algorithm
     - n
     - F1 skeleton
     - F1 pattern
     - SHD pattern
     - time (s)
   * - **Causal Automata**
     - 80/80
     - 0.822 ± 0.088
     - 0.754 ± 0.121
     - 50.1 ± 40.3
     - **0.76 ± 0.32**
   * - NOTEARS
     - 80/80
     - 0.896 ± 0.056
     - 0.768 ± 0.092
     - 39.0 ± 18.9
     - 31.4 ± 29.5
   * - PC
     - 71/80
     - 0.888 ± 0.061
     - 0.720 ± 0.085
     - 50.4 ± 27.5
     - 80.4 ± 342
   * - GOLEM
     - 80/80
     - 0.679 ± 0.086
     - 0.469 ± 0.084
     - 95.2 ± 55.5
     - 18.5 ± 11.2
   * - BiDAG MCMC
     - 63/80
     - 0.936 ± 0.049
     - 0.872 ± 0.093
     - 33.7 ± 34.9
     - 10.9 ± 16.6

Notes:

* **PC:** 71 cells with usable F1 (some SF failures/timeouts); time mean
  inflated by SF outliers.
* **BiDAG MCMC:** completed-cell mean only. Coverage **ER 40/40**,
  **SF 23/40** (missing seeds at ``p≥50`` due to BiDAG itsearch
  parent-set hardlimit=16). Do not treat SF MCMC means as full coverage.

Takeaways
~~~~~~~~~

* **Speed:** CA ~0.5–1.2s through ``p=100``; NOTEARS tens of seconds; PC
  can explode on SF; GOLEM ~5–35s.
* **Accuracy:** CA ≫ GOLEM; beats PC on pattern more often than not;
  neck-and-neck with NOTEARS on pattern (NOTEARS stronger on skeleton).
* **Ceiling:** Where BiDAG order-MCMC finishes, it sets a higher bar
  (~9× median / ~14× mean slower) — expensive oracle, not a default.
* **Graphs:** On pcalg ER/SF, CA is not simply “ER good / SF bad”; SF
  pattern F1 often matches or exceeds ER, with high seed variance.

Paired wins (same family × ``p`` × seed; CA vs other)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**F1 pattern — CA win rate**

.. list-table::
   :header-rows: 1
   :widths: 18 14 14 14 22

   * - vs
     - n paired
     - CA wins
     - CA loses
     - mean Δ (CA − other)
   * - NOTEARS
     - 80
     - 37 (**46%**)
     - 43
     - −0.014
   * - PC
     - 71
     - 44 (**62%**)
     - 27
     - +0.042
   * - GOLEM
     - 80
     - 79 (**99%**)
     - 1
     - +0.286
   * - BiDAG MCMC
     - 63
     - 14 (**22%**)
     - 49
     - −0.110

**F1 skeleton — CA win rate**

.. list-table::
   :header-rows: 1
   :widths: 18 10 18 16

   * - vs
     - n
     - CA wins
     - mean Δ
   * - NOTEARS
     - 80
     - 10 (12%)
     - −0.074
   * - PC
     - 71
     - 12 (17%)
     - −0.060
   * - GOLEM
     - 80
     - 76 (95%)
     - +0.143
   * - BiDAG MCMC
     - 63
     - 6 (10%)
     - −0.107

CA skeleton errors are FP-leaning (~58% of FP+FN).

MCMC as accuracy ceiling (overlap only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BiDAG order-MCMC completed **63/80** cells (ER 40/40, SF 23/40). On the
overlap:

* **ER** (n=40): CA F1 pattern 0.728 vs MCMC 0.848; times ~0.76s vs
  ~6.7s
* **SF** (n=23): CA F1 pattern 0.822 vs MCMC 0.915; times ~0.65s vs
  ~17.9s
* Runtime ratio MCMC/CA: **~9× median**, **~14× mean**

By family × ``p`` (mean F1 pattern / mean time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 12 14 16 16 14 18

   * - Cell
     - CA
     - NOTEARS
     - PC
     - GOLEM
     - MCMC
   * - ER p=20
     - 0.69 / 0.5s
     - 0.66 / 11s
     - 0.67 / 0.2s
     - 0.45 / 6s
     - 0.92 / 1.4s
   * - ER p=50
     - 0.70 / 0.5s
     - 0.80 / 22s
     - 0.72 / 0.6s
     - 0.41 / 11s
     - 0.89 / 4s
   * - ER p=80
     - 0.76 / 0.8s
     - 0.80 / 45s
     - 0.74 / 1.1s
     - 0.41 / 23s
     - 0.82 / 9s
   * - ER p=100
     - 0.76 / 1.2s
     - 0.85 / 53s
     - 0.76 / 1.5s
     - 0.42 / 34s
     - 0.77 / 13s
   * - SF p=20
     - 0.79 / 0.5s
     - 0.69 / 4s
     - 0.75 / 0.4s
     - 0.52 / 6s
     - 0.96 / 1.5s
   * - SF p=50
     - 0.72 / 0.5s
     - 0.75 / 27s
     - 0.69 / 271s\*
     - 0.50 / 11s
     - 0.88 / 22s (n=6)
   * - SF p=80
     - 0.80 / 0.8s
     - 0.80 / 36s
     - 0.70 / 315s\*
     - 0.52 / 22s
     - 0.89 / 38s (n=3)
   * - SF p=100
     - 0.80 / 1.2s
     - 0.78 / 54s
     - 0.70 / 223s\*
     - 0.53 / 35s
     - 0.86 / 38s (n=4)

\* PC SF times include severe outliers / incomplete seeds. MCMC SF rows
at ``p≥50`` are completed-cell means only (not full 10-seed coverage).

Mechanisms powering these numbers (``v0.3.2``–``v0.3.6``):

* Sample-size-adaptive Fisher-z thresholds via
  :class:`~causal_automata.CausalAutomaton` /
  :func:`~causal_automata.fisher_z_threshold`.
* Closed-form ``s=1`` singleton screens over **all** skeleton neighbors
  and ``s≥2`` 2-node recursive Sepset combos.
* PC-consistent endpoint-local neighbor pools (no cross-endpoint child
  mixing) with Berkson collider exclusion from separator sets.
* V-well / non-collider anti-sink elevation dynamics plus undirected
  final gravity projection (:math:`U=\max(A,A^\top)`).

See :doc:`physics` and :doc:`theory` for the mathematical detail.

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

The in-repo harness shares SEM helpers with
:mod:`causal_automata.bayesian_synth`
(:func:`~causal_automata.bayesian_synth.prepare_synthetic_bundle`).
PC uses gCastle ``variant="stable"`` with Fisher-Z tests. GOLEM uses
:data:`causal_automata.benchmark.GOLEM_DEFAULT_KWARGS` (no early-stop —
fixed Adam budget). The CA row warms up JIT outside the timed window.

Benchpress A10 rows above are from the external Singularity/Apptainer
pipeline on Lambda Cloud (not the gCastle harness).

Interpreting metrics
--------------------

* **Directed F1** — precision/recall on oriented edges vs ground truth.
* **Skeleton F1** — same on undirected pairs (either orientation counts).
* **Pattern / CPDAG F1** — F1 on the Meek / Chickering completed partially
  directed pattern (:func:`~causal_automata.dag_to_cpdag`).
* **Pattern SHD** — structural Hamming distance on the CPDAG pattern.
* **Wall time** — algorithm ``learn`` / CA step loop only (data generation
  excluded). Benchpress ``causal_automata`` times are measured under
  Apptainer ``--nv`` on the A10 host above; the in-repo gCastle harness
  warms JIT outside the timed window.

PC remains a strong accuracy baseline when runtime is acceptable. The
automaton is aimed at settings where parallel adjacency updates — and
optional 2-hop gravity — are useful alternatives to sequential CI testing
or long Adam runs.
