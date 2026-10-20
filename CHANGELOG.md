# Changelog

All notable changes to **causal-automata** are documented in this file.

## [0.2.0] — 2026-10-20

Milestone release: JAX-compiled CA physics, N=100 scaling, and Maximum Accuracy Mode.

### Added

- **JAX backend integration** — Fully `@jax.jit`-compiled physics engine in
  `causal_automata.jax_core` for CPU, NVIDIA CUDA, and Apple Metal/MPS
  (accelerator selected by the installed JAX wheel).
- **Dynamic conditioning (`k`-scaling)** — `default_conditioning_k(n)` uses
  **k=3** when **N ≤ 50** and **k=5** when **N > 50** (Upgrade 2). Override
  with `--k` / `--k-sweep`.
- **Multi-hop vision (`--multihop`)** — Matrix-multiplication 2-hop gravity
  via `A @ A` breaks Markov-equivalence ties (Upgrade 4). On the N=100
  seed=0 SEM with k=5 this is **Maximum Accuracy Mode**: Directed F1
  **0.696** (baseline 1-hop **0.619**) with flat wall time.
- **Fuzzy logic (`--fuzzy`)** — Opt-in continuous edge weights
  `A ∈ [0, 1]` with soft birth/death; binarize only after the final tick
  (Upgrade 3; experimental — did not beat binary on the default SEM).
- **Threshold annealing (`--anneal`)** — Opt-in linear decay of death/birth
  thresholds over progress ∈ [0, 1] (Upgrade 1; default remains static).

### Performance

- Massive wall-time advantage vs continuous score-based and constraint-based
  baselines on the project harness (see docs benchmarks): on the order of
  **~5,000×** vs NOTEARS-class continuous optimization and **~57×** vs PC
  at **N=50** scale in the project narrative; on Colab Tesla T4 **N=100**
  the JAX CA at historical k=3 finished in **0.323s** vs GOLEM **251s** /
  PC **529s**.

### Changed

- Benchmark / synth CLIs document Upgrade flags (`--multihop`, `--fuzzy`,
  `--anneal`, `--k`, `--compare-*`).
- README and Sphinx docs updated for the v0.2.0 engine.

## [0.1.2] — 2026-10

- Packaging / docs polish; PyPI badges and classifiers.

## [0.1.1] — 2026-10

- Early public docs and benchmark harness.

## [0.1.0] — 2026-10

- Initial NumPy Bayesian CA prototype and synthetic harness.
