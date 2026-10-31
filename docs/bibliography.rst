Bibliography
============

Primary sources for the theory pages (author manuscripts, October 2026):

* Yamagiwa, M. (2026). *Foundations of Causal Automata: A Physics-Inspired
  Framework for Directed Acyclic Graph Discovery.*
* Yamagiwa, M. (2026). *Causal Automata: From Intuition to GPU Physics —
  A Beginner-Friendly Guide to Directed Acyclic Graph Discovery.*
* Yamagiwa, M. (2026). *Bayesian Automata and Network Evolution —
  Synthesizing Cellular Automata, Message Passing, and Generative Topology.*

Related background
------------------

* Pearl, J. — belief propagation / local message passing on Bayesian
  networks (:math:`\pi` / :math:`\lambda` messages).
* Spirtes, Glymour, Scheines — PC algorithm and constraint-based causal
  discovery.
* GOLEM and related continuous DAG optimization (as implemented in
  gCastle for the project benchmarks).

Software
--------

* This package: ``causal-automata`` v0.3.3 — JAX ``@jit`` engine with
  multiplicative V-wells, local-margin gravity, JIT-safe ``run(ticks, k)``,
  dynamic ``k``, ``corr_flat`` init, and ``--multihop`` Maximum Accuracy Mode
  (:mod:`causal_automata.bayesian`, :mod:`causal_automata.jax_core`).
* Benchmarks via `gCastle <https://github.com/huawei-noah/trustworthyAI>`_
  PC and GOLEM.
