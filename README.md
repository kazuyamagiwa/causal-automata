# causal-automata

[![PyPI Version](https://img.shields.io/pypi/v/causal-automata.svg?cacheSeconds=3600&bust=1)](https://pypi.org/project/causal-automata/)
[![Python Versions](https://img.shields.io/pypi/pyversions/causal-automata.svg?cache=1)](https://pypi.org/project/causal-automata/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A blazing-fast, JAX-accelerated Cellular Automaton engine for causal discovery.

## Installation

```bash
pip install causal-automata
```

With CPU JAX:

```bash
pip install "causal-automata[jax]"
```

With NVIDIA CUDA JAX (and the Colab-style stack):

```bash
pip install "causal-automata[cuda]"
```

Or using `uv`:

```bash
uv add "causal-automata[jax]"
```

## Benchmark — N=100

Default stack (Upgrade 2): **`k=3` if N≤50**, **`k=5` if N>50**, binary edges, 1-hop gravity. On N=100 (`seed=0`, static thresholds, `ticks=100`) that baseline hits Skeleton **0.937** / Directed **0.619** in about **2.8s**.

**Maximum Accuracy Mode** — pass **`--multihop`** (Upgrade 4): 2-hop elevation gravity via `A @ A` lifts Directed F1 to **0.696** with essentially flat wall time.

| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |
| --- | ---: | ---: | ---: |
| PC (gcastle) | 0.968 | 0.813 | 528.631 |
| GOLEM (gcastle) | 0.577 | 0.403 | 251.182 |
| Bayesian CA (JAX, k=5, 1-hop) | **0.937** | 0.619 | ~2.9 |
| Bayesian CA (JAX, k=5, `--multihop`) | 0.928 | **0.696** | ~2.9 |

PC / GOLEM wall times are Colab **Tesla T4** (`num_iter=20000`). CA times are CPU JAX for the Upgrade-2 / Upgrade-4 A/B; at historical `k=3` the same CA finished in **0.323s** on T4 with Directed F1 **0.483**.

Opt-in flags (defaults stay fast & binary):

- **`--multihop`** — Maximum Accuracy Mode (2-hop gravity, Directed F1 **0.696** on this SEM).
- **`--fuzzy`** — continuous `A ∈ [0, 1]` soft birth/death (experimental; did not beat binary here).

Conditioning-set sweep (CA-only, 1-hop binary, same SEM):

| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |
| --- | ---: | ---: | ---: |
| Bayesian CA (JAX, k=3) | 0.812 | 0.483 | 2.314 |
| Bayesian CA (JAX, k=4) | 0.915 | 0.546 | 2.481 |
| Bayesian CA (JAX, k=5) | **0.937** | **0.619** | **2.871** |
| Bayesian CA (JAX, k=6) | 0.937 | 0.619 | 2.454 |
| Bayesian CA (JAX, k=7) | 0.937 | 0.619 | 3.049 |
| Bayesian CA (JAX, k=8) | 0.937 | 0.627 | 3.647 |

## Documentation

Full docs (theory, physics, API, CLI): https://causal-automata.readthedocs.io/en/latest/

## How it works — CA physics + V-well

State is a binary DAG adjacency `A`, node elevations `E ∈ [0,1]`, and a fixed empirical covariance `Cov` from continuous observations. Each tick is a local cellular update:

1. **Gravity** — keep only downhill edges (`E[parent] > E[child]`), then drift elevations toward the midpoint of parent/child means. With **`--multihop`**, that drift also sees 2-hop neighbors from `A @ A` (discounted pull) to help break Markov-equivalence ties.
2. **V-structure gravity well** — absent undirected pairs that share neighbors and are marginally independent (`|ρ|` below the death threshold with empty conditioning set) mark shared neighbors as **colliders**. Those elevations are sunk (`E -= 2·lr`), so later birth prefers arrows into the collider — the classic V-structure `i → k ← j`.
3. **Death / birth** — kill weak edges and birth strong ones using partial correlations conditioned on the top-`k` union neighbors ranked by `|Cov|`. Birth orients by elevation after the `|ρ|` gate. Optional **`--fuzzy`** keeps continuous weights in `[0, 1]` and binarizes only after the final tick.

The NumPy prototype (`bayesian.step`) and the pure functional JAX port (`jax_core.step`, `@jax.jit`) share this physics.

## Run

```bash
# Synthetic DAG harness (NumPy default; use --backend jax for JIT)
python -m causal_automata.bayesian_synth --n 10
python -m causal_automata.bayesian_synth --n 50 --backend jax
python -m causal_automata.bayesian_synth --n 100

# Benchmark vs gCastle PC / GOLEM (default N=100, k=5, 1-hop)
python -m causal_automata.benchmark
python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100

# Maximum Accuracy Mode (Upgrade 4) — Directed F1 0.696 on the N=100 SEM
python -m causal_automata.benchmark --n 100 --ca-only --k 5 --multihop
python -m causal_automata.benchmark --n 100 --ca-only --k 5 --compare-multihop

python -m causal_automata.benchmark --n 100 --ca-only --k-sweep 5,6,7,8
python -m causal_automata.benchmark --smoke   # device checks + GOLEM init only
```

Smoke the JAX compile:

```bash
python -c "from causal_automata import warmup; print(warmup())"
```

## Package layout

```
causal-automata/
├── causal_automata/
│   ├── __init__.py
│   ├── bayesian.py          # NumPy Bayesian CA step
│   ├── bayesian_synth.py    # Synthetic linear-DAG harness
│   ├── jax_core.py          # JAX @jit port of step
│   └── benchmark.py         # vs gCastle PC / GOLEM
├── docs/                    # Sphinx / Read the Docs
├── .readthedocs.yaml
├── pyproject.toml
├── README.md
└── NOTICE
```

## License / notice

See `NOTICE`. Copyright 2026 Masakazu Yamagiwa.
