causal-automata
===============

**causal-automata** (v0.2.0) is a JAX-accelerated cellular automaton
engine for *causal discovery*: recovering a Directed Acyclic Graph (DAG)
from observational data by evolving an adjacency matrix under
physics-inspired local rules.

Alongside constraint-based search (PC) and continuous score-based
methods (GOLEM / NOTEARS-class), the engine treats edges as particles on
a grid. Gravity, V-wells (collider sinks), birth, and death update every
candidate edge in parallel under ``@jax.jit``. Optional **``--multihop``**
(Maximum Accuracy Mode) uses :math:`A@A` for 2-hop vision — **0.696**
Directed F1 on the N=100 harness — with wall times reported under
:doc:`benchmarks`.

See :doc:`installation` and :doc:`quickstart` to get started.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
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
