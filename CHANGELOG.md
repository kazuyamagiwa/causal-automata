# Changelog

All notable changes to **causal-automata** are documented in this file.

## [0.3.1] — 2026-10-26

Audit-fix + stability release: unify NumPy/JAX physics, close the L4
permutation init bias, and stop the V-well floor limit cycle. First PyPI /
Read the Docs cut after **0.3.0**.

### Fixed

- **H2 / H3** — Local-margin gravity (no hard ``1.0`` / ``0.0`` anchors) and
  Cov-equivariant bound-aware elevation tie-breaks on birth / final
  projection so nodes clipped together at a bound still orient and chains
  no longer reverse from ceiling saturation.
- **H1 / M4** — NumPy ``bayesian.step`` and JAX ``jax_core.step`` share the
  same tick order and forces; Phase-2 N=10 seed=0 recovers **all true
  edges** on both backends (stable Directed F1 **≥ 0.88**, parity-tested).
- **M1** — Conditioning sets are ranked by absolute **correlation** ``|R|``
  (scale-free); raw-elevation masking of candidates is removed.
- **L1** — License metadata unified on **Apache-2.0** (``pyproject.toml``,
  README, NOTICE).
- **L2** — Removed import-time ``jax_enable_x64``; CLIs / tests call
  :func:`enable_x64` explicitly.
- **L4** — Synthetic / benchmark bundles permute nodes by default. Default
  init is permutation-invariant ``corr_flat``: symmetric ``|R| > thr``
  skeleton ``A0`` and near-flat ``E0 = 0.5 + 10^{-4}`` noise (legacy
  upper-tri ``A0`` locked gravity into ``i < j`` on tick 1).
- **V-well limit cycle** — Replace additive ``E - 2·lr`` (which slammed
  mid-chain nodes into the ``E=0`` clip and flipped near-floor orderings
  every other tick) with multiplicative ``E ← E·(1-2·lr)``, unioning
  evolving-adjacency and static ``|R|``-skeleton collider cues. Structural
  metrics now settle to a flat plateau after ~25 ticks.

### Changed

- Tick order is **gravity → v_well → death → birth** (the death-first v0.3.0
  JAX order regressed directed recovery on the Phase-2 motif).
- Synth ``init_mode``: ``"corr_flat"`` (default) or ``"upper_random"`` (H1
  regression / index-ordered motif).
- README convergence GIF / MP4 regenerated from Phase-2 N=10 with
  ``corr_flat`` (opens on an ``E≈0.5`` plateau).
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
  (Upgrade 3). Default remains binary.
- **Threshold annealing (`--anneal`)** — Linear schedules for death / birth
  thresholds over ticks.
- **N=100 benchmark harness** — `python -m causal_automata.benchmark`
  with `--compare` (PC / GOLEM via gCastle), `--smoke`, and device banners.
- **Docs** — Read the Docs site (`docs/`), `.readthedocs.yaml`, and expanded
  README researcher narrative.

## [0.1.2] — 2026-10-17

Packaging and metadata polish.

## [0.1.1] — 2026-10-17

Python version specification fix for PyPI classifiers.

## [0.1.0] — 2026-10-17

Initial public release.
