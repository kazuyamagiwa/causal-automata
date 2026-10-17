causal-automata
===============

**causal-automata** is a JAX-accelerated cellular automaton engine for
*causal discovery*: recovering a Directed Acyclic Graph (DAG) from
observational data by evolving an adjacency matrix under physics-inspired
local rules.

Instead of sequential conditional-independence searches (PC) or slow
continuous optimization (GOLEM), the engine treats edges as particles on a
grid. Gravity, V-wells (collider sinks), birth, and death update every
candidate edge in parallel — typically recovering an :math:`N=100` graph
in a fraction of a second on GPU.

.. code-block:: bash

   pip install -e ".[jax]"
   python -m causal_automata.bayesian_synth --n 10
   python -c "from causal_automata import warmup; print(warmup())"

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
