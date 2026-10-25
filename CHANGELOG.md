# Changelog

All notable changes to **causal-automata** are documented in this file.

## [0.3.1] — 2026-10-25

Audit-fix release: unify NumPy/JAX physics, repair elevation saturation /
birth deadlocks, align docs with the implemented estimator.

### Fixed

- **H2 / H3** — Local-margin gravity (no hard ``1.0`` / ``0.0`` anchors) and
  bound-aware elevation tie-breaks on birth / final projection so nodes
  clipped together at a bound still orient and chains no longer reverse from
  ceiling saturation.
- **H1 / M4** — NumPy ``bayesian.step`` and JAX ``jax_core.step`` now share
  the same tick order and forces; Phase-2 N=10 seed=0 Directed F1 returns to
  **≥ 0.90** on both backends (parity-tested).
- **M1** — Conditioning sets are ranked by absolute **correlation** ``|R|``
  (scale-free); raw-elevation masking of candidates is removed.
- **L1** — License metadata unified on **Apache-2.0** (``pyproject.toml``,
  README, NOTICE).
- **L2** — Removed import-time ``jax_enable_x64``; CLIs / tests call
  :func:`enable_x64` explicitly.
- **L4** — Synthetic / benchmark bundles apply a random node permutation by
  default so index-ordered topo labels are not privileged.

### Changed

- Tick order is **gravity → v_well → death → birth** (the death-first v0.3.0
  JAX order regressed directed recovery on the Phase-2 motif).
- Docs (``physics.rst``, README, glossary, quickstart) distinguish Birth vs
  Gravity, document V-well as an elevation force, and state top-``k`` CI /
  \(O(N^2)\) projection vs \(O(N^3)\) tick cost honestly.

## [0.3.0] — 2026-10-24

Physics correctness release: V-well / gravity / conditioning fixes, JIT-safe
`run()`, and documentation for static `ticks` / `k`.

### Added

- **`run(A0, E0, Cov, ticks=..., k=...)`** — `jax.lax.scan` driver with
  `static_argnames=("ticks", "k", ...)` so scan length and `(2+k)×(2+k)`
  partial-correlation blocks stay compile-time constants.
- **`detect_colliders`** — public explaining-away collider mask helper.
- **Docs** — Read the Docs section on JAX concretization rules for `ticks` /
  `k`, with `functools.partial` examples; `.readthedocs.yaml` installs via
  native `uv` (`method: uv`, extras `docs` + `cpu`) to match GitHub Actions CI.
- **Tests** — `tests/test_causal_fixes.py` (root stability, collider vs fork,
  node-permutation Skeleton F1 / SHD).

### Fixed

- **V-well / explaining-away** — collider detection uses unshielded triples with
  Berkson (explaining-away) partial correlations; relative elevation targets
  replace the flat sink that clipped cascaded colliders.
- **Root / isolate gravity** — missing parents anchor at `1.0`, missing children
  at `0.0`; isolates hold `E`; downhill projection uses index tie-breaks.
- **Conditioning M-bias** — top-`k` candidates with `E[z] < min(E[i], E[j])`
  are masked out of `S_{i,j}`. Tick order is now death → birth → v_well → gravity.

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
