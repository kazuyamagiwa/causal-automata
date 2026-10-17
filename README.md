# causal-automata

A blazing-fast, JAX-accelerated Cellular Automaton engine for causal discovery.

## Benchmark — N=100 on Tesla T4

Final figures (Colab **Tesla T4**, GOLEM `num_iter=20000`):

| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |
| --- | ---: | ---: | ---: |
| PC (gcastle) | 0.968 | 0.813 | 528.631 |
| GOLEM (gcastle) | 0.577 | 0.403 | 251.182 |
| Bayesian CA (JAX) | 0.812 | 0.483 | 0.323 |

Bayesian CA is ~**778×** faster than GOLEM and ~**1636×** faster than PC, while beating GOLEM on both skeleton (**0.812 vs 0.577**) and directed (**0.483 vs 0.403**) F1.

## Documentation

Full docs (theory, physics, API, CLI): https://causal-automata.readthedocs.io/en/latest/

Build locally from `docs/`:

```bash
pip install -e ".[docs,jax]"
cd docs && make html
```

## How it works — CA physics + V-well

State is a binary DAG adjacency `A`, node elevations `E ∈ [0,1]`, and a fixed empirical covariance `Cov` from continuous observations. Each tick is a local cellular update:

1. **Gravity** — keep only downhill edges (`E[parent] > E[child]`), then drift elevations toward the midpoint of parent/child means.
2. **V-structure gravity well** — absent undirected pairs that share neighbors and are marginally independent (`|ρ|` below the death threshold with empty conditioning set) mark shared neighbors as **colliders**. Those elevations are sunk (`E -= 2·lr`), so later birth prefers arrows into the collider — the classic V-structure `i → k ← j`.
3. **Death / birth** — kill weak edges and birth strong ones using partial correlations conditioned on the top-`k` union neighbors ranked by `|Cov|`. Birth orients by elevation after the `|ρ|` gate.

The NumPy prototype (`bayesian.step`) and the pure functional JAX port (`jax_core.step`, `@jax.jit`) share this physics.

## Install (Colab T4 / CUDA)

```bash
# Runtime → Change runtime type → T4 GPU
pip install -U "jax[cuda13]" gcastle torch networkx scikit-learn numpy
pip install -e .
```

CPU / portable (no CUDA wheel):

```bash
pip install -e ".[jax,gcastle]"
# or: uv sync --extra jax --extra gcastle
```

Full NVIDIA extra (mirrors the Colab stack in project metadata):

```bash
pip install -e ".[cuda]"
```

## Run

```bash
# Synthetic DAG harness (NumPy default; use --backend jax for JIT)
python -m causal_automata.bayesian_synth --n 10
python -m causal_automata.bayesian_synth --n 50 --backend jax
python -m causal_automata.bayesian_synth --n 100

# Benchmark vs gCastle PC / GOLEM (default N=100)
python -m causal_automata.benchmark
python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100
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
