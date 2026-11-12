causal-automata
===============

**causal-automata** (v0.3.6) is a JAX-accelerated cellular automaton
engine for *causal discovery*: recovering a Directed Acyclic Graph (DAG)
from observational data by evolving an adjacency matrix under
physics-inspired local rules.

Alongside constraint-based search (PC) and continuous score-based
methods (GOLEM / NOTEARS-class), the engine treats edges as particles on
a grid. Death, birth, V-wells (explaining-away collider sinks), and
gravity update every candidate edge in parallel under ``@jax.jit``.
Benchmarks run with Benchpress on NVIDIA A10 (pcalg ER + scale-free,
``p∈{20,50,80,100}``, ``d=4``, ``n=1000``, 80 cells) put ``v0.3.6`` at
Skeleton F1 ``0.822 ± 0.088``, Pattern F1 ``0.754 ± 0.121``, and
``0.76 ± 0.32s`` mean wall time — neck-and-neck with NOTEARS on pattern,
ahead of PC more often than not, and well ahead of GOLEM, while staying
~0.5–1.2s through ``p=100`` (see :doc:`benchmarks`). Optional
``--multihop`` (Maximum Accuracy Mode) uses :math:`A@A` for 2-hop vision
— **0.696** Directed F1 on the N=100 harness.

See :doc:`installation` and :doc:`quickstart` to get started. If you wrap
``step`` / ``run`` yourself, read :doc:`jax_static` before tracing
``ticks`` or ``k``.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   jax_static
   cli
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory
   physics
   glossary

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   bibliography

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
