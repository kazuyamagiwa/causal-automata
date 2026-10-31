Quickstart
==========

Minimal discovery loop
----------------------

The public API centers on a single tick function and an optional scan
driver. The package exports the JIT-compiled JAX step as
:func:`causal_automata.step`, the multi-tick :func:`causal_automata.run`,
and the NumPy prototype as :func:`causal_automata.numpy_step`.

.. code-block:: python

   import numpy as np
   from causal_automata import (
       run,
       step,
       warmup,
       default_conditioning_k,
       __version__,
   )
   from causal_automata.bayesian_synth import prepare_synthetic_bundle

   print(__version__, warmup())  # e.g. 0.3.3 + device string

   n = 100
   bundle = prepare_synthetic_bundle(n=n, seed=0)
   A, E, Cov = bundle.A_init, bundle.E_init, bundle.Cov
   k = default_conditioning_k(n)  # 3 if N≤50, 5 if N>50 (Python int)

   # Preferred: jit-scanned horizon (ticks and k are static args)
   A, E = run(A, E, Cov, ticks=100, k=k, threshold_birth=0.10)

   # Equivalent explicit loop:
   # for _ in range(100):
   #     A, E = step(A, E, Cov, k=k, threshold_birth=0.10)

   print("Predicted edges:", int((A > 0).sum()))

``ticks`` and ``k`` must be ordinary Python integers. Passing traced JAX
scalars raises ``ConcretizationTypeError`` — see :doc:`jax_static` for
``functools.partial`` patterns.

Maximum Accuracy Mode (2-hop gravity)
-------------------------------------

Pass ``multihop=True`` (CLI: ``--multihop``) so elevation drift also sees
neighbors two hops away via :math:`A@A`::

   A, E = run(
       A, E, Cov,
       ticks=100,
       k=k,
       threshold_birth=0.10,
       multihop=True,
       hop_discount=1.0,
   )

On the default N=100 SEM this lifts Directed F1 from **0.619** (1-hop)
to **0.696** with flat wall time. See :doc:`benchmarks`.

State variables
---------------

=========== ===============================================================
Symbol      Meaning
=========== ===============================================================
``A``       Adjacency :math:`(N, N)`. Binary by default; with
            ``fuzzy=True`` continuous weights in :math:`[0, 1]`
            (binarize after the loop, e.g. ``A > 0.5``).
``E``       Node elevations :math:`(N,)` in :math:`[0, 1]`. Arrows may
            only point *downhill* (:math:`E_{\mathrm{parent}} >
            E_{\mathrm{child}}`), which enforces acyclicity.
``Cov``     Fixed empirical covariance of continuous observations.
=========== ===============================================================

Each call to ``step`` applies, in order:

1. **Gravity** — raw downhill projection, then local-margin elevation
   drift (roots sit just above children, leaves just below parents;
   isolates hold ``E``). Optional ``multihop=True`` also pulls toward
   2-hop means from :math:`A@A`.
2. **V-well** — lowers elevations of shared neighbors of marginally
   independent pairs (collider cue). This updates ``E`` only; edges are
   re-oriented by later birth / downhill projection.
3. **Death / birth** — prune weak and spawn strong edges using partial
   correlations conditioned on the top-``k`` union neighbors ranked by
   absolute **correlation** :math:`|R|` (not raw ``|Cov|``). Birth
   orients with a deterministic elevation tie-break so nodes clipped at
   the same bound still get a direction.

NumPy vs JAX
------------

.. code-block:: python

   from causal_automata.bayesian import step as numpy_step
   from causal_automata.jax_core import step as jax_step, run

   A1, E1 = numpy_step(A, E, Cov, threshold_birth=0.10, k=3)
   A2, E2 = jax_step(A, E, Cov, threshold_birth=0.10, k=3)
   A3, E3 = run(A, E, Cov, ticks=100, k=3, threshold_birth=0.10)

Both backends share the **same** core physics (parity-tested on the
Phase-2 N=10 SEM). Prefer JAX when ``N`` is large or a GPU / MPS device
is available; prefer NumPy for debugging and environments without JAX.
Upgrade flags (``multihop``, ``fuzzy``, ``anneal``) are JAX-only.

Synthetic end-to-end harness
----------------------------

.. code-block:: bash

   python -m causal_automata.bayesian_synth --n 10
   python -m causal_automata.bayesian_synth --n 50 --backend jax
   python -m causal_automata.bayesian_synth --n 100 --backend jax

   # Benchmark Maximum Accuracy Mode
   python -m causal_automata.benchmark --n 100 --ca-only --k 5 --multihop

See :doc:`cli` for the full flag list and :doc:`benchmarks` for comparison
against gCastle PC and GOLEM.
