"""Benchmark JAX Bayesian CA vs gCastle PC / GOLEM.

Default scale is **N=100** (same SEM helpers as :mod:`bayesian_synth`).
GOLEM default budget is ``num_iter=20000`` (no early-stop in gCastle).

Install::

    uv sync --extra gcastle --extra jax
    # or: pip install gcastle

    # Colab T4 / NVIDIA (example):
    #   pip install -U "jax[cuda13]" gcastle torch
    #   # see JAX install docs if the CUDA extra name differs

Run::

    uv run python -m causal_automata.benchmark
    uv run python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100

    # CA-only upgrade A/B (skip PC / GOLEM):
    uv run python -m causal_automata.benchmark --n 50 --ca-only
    uv run python -m causal_automata.benchmark --n 50 --ca-only --anneal

    # Upgrade 2 — conditioning-set sweep (same SEM bundle, several k):
    uv run python -m causal_automata.benchmark --n 100 --ca-only --k-sweep 3,4,5

    # Upgrade 3 — binary vs fuzzy continuous edges (same SEM, default k):
    uv run python -m causal_automata.benchmark --n 100 --ca-only --compare-fuzzy

    # Upgrade 4 — 1-hop vs 2-hop gravity (binary, default k):
    uv run python -m causal_automata.benchmark --n 100 --ca-only --k 5 --compare-multihop

    # CPU / no-GPU smoke (import + GOLEM init only; no invented GPU timings):
    uv run python -m causal_automata.benchmark --smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

from causal_automata.bayesian_synth import (
    SyntheticBundle,
    default_conditioning_k,
    edge_metrics,
    prepare_synthetic_bundle,
    skeleton_metrics,
)


@dataclass(frozen=True)
class BenchRow:
    algorithm: str
    skeleton_f1: float
    directed_f1: float
    wall_time_s: float
    directed_tp: int
    directed_fp: int
    directed_fn: int
    skeleton_tp: int
    skeleton_fp: int
    skeleton_fn: int
    notes: str = ""


def _p(*args, **kwargs) -> None:
    """Print with flush so device notes appear promptly under pipes/tee."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def print_device_checks() -> dict:
    """Print JAX / PyTorch device availability; return a small summary dict."""
    info: dict = {"torch_cuda": False, "jax_backend": "unavailable", "jax_devices": []}

    _p("=== Device checks ===")
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        info["torch_cuda"] = cuda
        _p(f"torch.cuda.is_available(): {cuda}")
        if cuda:
            _p(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            _p(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        else:
            _p("torch: CUDA not available (CPU)")
    except ImportError as exc:
        _p(f"torch: not installed ({exc})")

    try:
        import jax

        from causal_automata.jax_core import device_info

        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
        _p(f"jax.default_backend(): {info['jax_backend']}")
        _p(f"jax.devices(): {info['jax_devices']}")
        _p(f"jax_core.device_info(): {device_info()}")
    except ImportError as exc:
        _p(f"jax: not installed ({exc})")

    _p()
    return info


def _require_gcastle():
    try:
        from castle.algorithms import GOLEM, PC
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "gcastle is required for this benchmark.\n"
            "  uv sync --extra gcastle\n"
            "  # or: pip install gcastle\n"
            f"Import error: {exc}"
        ) from exc
    return PC, GOLEM


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_golem_device_type(*, want_gpu: bool | None = None) -> str:
    """Pick GOLEM ``device_type`` without crashing when CUDA is absent.

    gCastle's GOLEM raises ``ValueError`` if ``device_type='gpu'`` and
    ``torch.cuda.is_available()`` is False — so we only request GPU when CUDA
    is actually present.
    """
    if want_gpu is None:
        want_gpu = _cuda_available()
    if want_gpu and _cuda_available():
        _p("GOLEM: passing device_type='gpu'")
        return "gpu"
    if want_gpu and not _cuda_available():
        _p(
            "GOLEM: CUDA unavailable — falling back to device_type='cpu' "
            "(gCastle GOLEM raises ValueError if device_type='gpu' without CUDA)."
        )
        return "cpu"
    _p("GOLEM: passing device_type='cpu' (CUDA unavailable)")
    return "cpu"


def _as_binary_adj(mat) -> np.ndarray:
    # JAX device arrays often yield read-only NumPy views; force a writeable copy.
    A = np.asarray(mat, dtype=np.float64).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square adjacency, got shape {A.shape}")
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(np.float64)


def _score(A_pred: np.ndarray, A_true: np.ndarray) -> tuple:
    d = edge_metrics(A_pred, A_true)
    s = skeleton_metrics(A_pred, A_true)
    return d, s


def run_pc(X: np.ndarray, A_true: np.ndarray, *, alpha: float = 0.05) -> BenchRow:
    PC, _ = _require_gcastle()
    t0 = time.perf_counter()
    pc = PC(variant="stable", alpha=alpha, ci_test="fisherz")
    pc.learn(X)
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(pc.causal_matrix)
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="PC (gcastle)",
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=f"variant=stable alpha={alpha} fisherz",
    )


# Benchmark-tuned GOLEM-EV defaults (gCastle has no early-stop / tol knobs).
# num_iter=20000: loss typically plateaus by ~15k under the prior 100k budget.
GOLEM_DEFAULT_KWARGS = {
    "lambda_1": 0.02,
    "lambda_2": 5.0,
    "equal_variances": True,
    "learning_rate": 1e-3,
    "num_iter": 20_000,
    "checkpoint_iter": 5_000,
    "seed": 1,
    "graph_thres": 0.3,
}


def run_golem(
    X: np.ndarray,
    A_true: np.ndarray,
    *,
    lambda_1: float = GOLEM_DEFAULT_KWARGS["lambda_1"],
    lambda_2: float = GOLEM_DEFAULT_KWARGS["lambda_2"],
    equal_variances: bool = GOLEM_DEFAULT_KWARGS["equal_variances"],
    learning_rate: float = GOLEM_DEFAULT_KWARGS["learning_rate"],
    num_iter: int = GOLEM_DEFAULT_KWARGS["num_iter"],
    checkpoint_iter: int = GOLEM_DEFAULT_KWARGS["checkpoint_iter"],
    seed: int = GOLEM_DEFAULT_KWARGS["seed"],
    graph_thres: float = GOLEM_DEFAULT_KWARGS["graph_thres"],
    want_gpu: bool | None = None,
) -> BenchRow:
    """Run gCastle GOLEM (PyTorch); prefer GPU when CUDA is available.

    gCastle GOLEM exposes no early-stopping / tolerance parameters — training
    always runs a fixed ``num_iter`` Adam loop (``checkpoint_iter`` only logs).
    """
    _, GOLEM = _require_gcastle()
    device_type = resolve_golem_device_type(want_gpu=want_gpu)
    kwargs = {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "equal_variances": equal_variances,
        "learning_rate": learning_rate,
        "num_iter": num_iter,
        "checkpoint_iter": checkpoint_iter,
        "seed": seed,
        "graph_thres": graph_thres,
        "device_type": device_type,
    }
    # Defensible Details line: full kwargs + explicit no-early-stop note.
    note = (
        " ".join(f"{k}={v}" for k, v in kwargs.items())
        + " early_stop=absent(gcastle)"
    )
    _p(f"GOLEM kwargs: {note}")
    t0 = time.perf_counter()
    model = GOLEM(**kwargs)
    model.learn(X)
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(model.causal_matrix)
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="GOLEM (gcastle)",
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=note,
    )


def smoke_golem_init() -> None:
    """Import/init GOLEM without ``learn`` — for CPU VMs / CI smoke tests."""
    _, GOLEM = _require_gcastle()
    device_type = resolve_golem_device_type()
    model = GOLEM(device_type=device_type, num_iter=10, checkpoint_iter=5)
    _p(f"GOLEM smoke init OK: device={model.device} device_type={model.device_type}")

    # Document gCastle's hard fail if callers pass gpu blindly.
    if not _cuda_available():
        try:
            GOLEM(device_type="gpu", num_iter=1)
            _p("GOLEM(device_type='gpu') unexpectedly succeeded without CUDA")
        except ValueError as exc:
            _p(f"GOLEM(device_type='gpu') without CUDA raises ValueError (expected): {exc}")


# Upgrade-1 annealing endpoints (linear in progress ∈ [0, 1]).
# Early ticks: strict gates (solidify strong backbones).
# Late ticks: permissive gates (subtle edge formation).
ANNEAL_DEATH_START = 0.10
ANNEAL_DEATH_END = 0.05
ANNEAL_BIRTH_START = 0.20
ANNEAL_BIRTH_END = 0.05


FUZZY_BINARIZE = 0.5


def run_jax_ca(
    Cov: np.ndarray,
    A_init: np.ndarray,
    E_init: np.ndarray,
    A_true: np.ndarray,
    *,
    ticks: int = 100,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.10,
    k: int = 3,
    anneal: bool = False,
    threshold_death_start: float = ANNEAL_DEATH_START,
    threshold_birth_start: float = ANNEAL_BIRTH_START,
    threshold_death_end: float | None = None,
    threshold_birth_end: float | None = None,
    label_k: bool = False,
    fuzzy: bool = False,
    birth_rate: float = 0.5,
    death_rate: float = 0.1,
    binarize_threshold: float = FUZZY_BINARIZE,
    multihop: bool = False,
    hop_discount: float = 1.0,
    algorithm: str | None = None,
) -> BenchRow:
    try:
        import jax.numpy as jnp

        from causal_automata.jax_core import device_info, step as jax_step
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "JAX is required for the CA row of this benchmark.\n"
            "  uv sync --extra jax\n"
            f"Import error: {exc}"
        ) from exc

    A = jnp.asarray(A_init, dtype=jnp.float64)
    E = jnp.asarray(E_init)
    Cov_j = jnp.asarray(Cov)

    if anneal:
        death_end = (
            ANNEAL_DEATH_END if threshold_death_end is None else threshold_death_end
        )
        birth_end = (
            ANNEAL_BIRTH_END if threshold_birth_end is None else threshold_birth_end
        )
        death_start = threshold_death_start
        birth_start = threshold_birth_start
        denom = max(ticks - 1, 1)

        def _call(tick: int):
            return jax_step(
                A,
                E,
                Cov_j,
                learning_rate=learning_rate,
                threshold_death=death_end,
                threshold_birth=birth_end,
                k=k,
                anneal=True,
                progress=float(tick) / float(denom),
                threshold_death_start=death_start,
                threshold_birth_start=birth_start,
                fuzzy=fuzzy,
                birth_rate=birth_rate,
                death_rate=death_rate,
                multihop=multihop,
                hop_discount=hop_discount,
            )

        schedule = (
            f"anneal death {death_start:.2f}→{death_end:.2f} "
            f"birth {birth_start:.2f}→{birth_end:.2f}"
        )
    else:
        # Historical static Phase-6 gates; no progress / start knobs.
        death_end = threshold_death
        birth_end = threshold_birth

        def _call(_tick: int):
            return jax_step(
                A,
                E,
                Cov_j,
                learning_rate=learning_rate,
                threshold_death=death_end,
                threshold_birth=birth_end,
                k=k,
                anneal=False,
                fuzzy=fuzzy,
                birth_rate=birth_rate,
                death_rate=death_rate,
                multihop=multihop,
                hop_discount=hop_discount,
            )

        schedule = f"static death={death_end:.2f} birth={birth_end:.2f}"

    edge_mode = (
        f"fuzzy birth_rate={birth_rate:.2f} death_rate={death_rate:.2f} "
        f"binarize>{binarize_threshold:.2f}"
        if fuzzy
        else "binary"
    )
    hop_mode = (
        f"multihop discount={hop_discount:.2f}" if multihop else "1-hop"
    )

    # Warmup compile outside the timed window (discard result; re-init state).
    _ = _call(0)
    A = jnp.asarray(A_init, dtype=jnp.float64)
    E = jnp.asarray(E_init)
    t0 = time.perf_counter()
    for tick in range(ticks):
        A, E = _call(tick)
    elapsed = time.perf_counter() - t0
    A_np = np.asarray(A).copy()
    if fuzzy:
        A_pred = _as_binary_adj((A_np > binarize_threshold).astype(np.float64))
    else:
        A_pred = _as_binary_adj(A_np)
    d, s = _score(A_pred, A_true)
    if algorithm is None:
        tags = []
        if fuzzy:
            tags.append("fuzzy")
        if multihop:
            tags.append("multihop")
        if label_k or tags:
            tag_s = (", ".join(tags) + ", " if tags else "") + f"k={k}"
            algorithm = f"Bayesian CA (JAX, {tag_s})"
        else:
            algorithm = "Bayesian CA (JAX)"
    return BenchRow(
        algorithm=algorithm,
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=(
            f"ticks={ticks} k={k} {edge_mode} {hop_mode} {schedule} "
            f"{device_info()}"
        ),
    )


def markdown_table(rows: list[BenchRow]) -> str:
    lines = [
        "| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.algorithm} | {r.skeleton_f1:.3f} | {r.directed_f1:.3f} | "
            f"{r.wall_time_s:.3f} |"
        )
    return "\n".join(lines)


def _parse_k_sweep(spec: str) -> list[int]:
    """Parse ``'3,4,5'`` into a list of non-negative ints (order preserved)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--k-sweep requires at least one integer")
    vals: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --k-sweep entry {p!r}; expected integers"
            ) from exc
        if v < 0:
            raise argparse.ArgumentTypeError(f"k must be >= 0, got {v}")
        vals.append(v)
    return vals


def run_benchmark(
    *,
    n: int = 100,
    seed: int = 0,
    ticks: int = 100,
    k: int | None = None,
    k_sweep: list[int] | None = None,
    n_samples: int | None = None,
    golem_iters: int = GOLEM_DEFAULT_KWARGS["num_iter"],
    ca_only: bool = False,
    anneal: bool = False,
    fuzzy: bool = False,
    compare_fuzzy: bool = False,
    birth_rate: float = 0.5,
    death_rate: float = 0.1,
    multihop: bool = False,
    compare_multihop: bool = False,
    hop_discount: float = 1.0,
) -> tuple[list[BenchRow], SyntheticBundle]:
    if k is None:
        k = default_conditioning_k(n)
    bundle = prepare_synthetic_bundle(n=n, seed=seed, n_samples=n_samples)
    want_gpu = _cuda_available()

    rows: list[BenchRow] = []
    if not ca_only:
        rows.append(run_pc(bundle.X, bundle.A_true))
        rows.append(
            run_golem(bundle.X, bundle.A_true, want_gpu=want_gpu, num_iter=golem_iters)
        )
    ks = list(k_sweep) if k_sweep else [k]
    label_k = len(ks) > 1 and not compare_fuzzy and not compare_multihop
    for ki in ks:
        if compare_fuzzy:
            rows.append(
                run_jax_ca(
                    bundle.Cov,
                    bundle.A_init,
                    bundle.E_init,
                    bundle.A_true,
                    ticks=ticks,
                    k=ki,
                    anneal=anneal,
                    fuzzy=False,
                    multihop=multihop,
                    hop_discount=hop_discount,
                    algorithm=f"Bayesian CA (JAX, binary, k={ki})",
                )
            )
            rows.append(
                run_jax_ca(
                    bundle.Cov,
                    bundle.A_init,
                    bundle.E_init,
                    bundle.A_true,
                    ticks=ticks,
                    k=ki,
                    anneal=anneal,
                    fuzzy=True,
                    birth_rate=birth_rate,
                    death_rate=death_rate,
                    multihop=multihop,
                    hop_discount=hop_discount,
                    algorithm=f"Bayesian CA (JAX, fuzzy, k={ki})",
                )
            )
        elif compare_multihop:
            rows.append(
                run_jax_ca(
                    bundle.Cov,
                    bundle.A_init,
                    bundle.E_init,
                    bundle.A_true,
                    ticks=ticks,
                    k=ki,
                    anneal=anneal,
                    fuzzy=fuzzy,
                    birth_rate=birth_rate,
                    death_rate=death_rate,
                    multihop=False,
                    algorithm=f"Bayesian CA (JAX, 1-hop, k={ki})",
                )
            )
            rows.append(
                run_jax_ca(
                    bundle.Cov,
                    bundle.A_init,
                    bundle.E_init,
                    bundle.A_true,
                    ticks=ticks,
                    k=ki,
                    anneal=anneal,
                    fuzzy=fuzzy,
                    birth_rate=birth_rate,
                    death_rate=death_rate,
                    multihop=True,
                    hop_discount=hop_discount,
                    algorithm=f"Bayesian CA (JAX, multihop, k={ki})",
                )
            )
        else:
            rows.append(
                run_jax_ca(
                    bundle.Cov,
                    bundle.A_init,
                    bundle.E_init,
                    bundle.A_true,
                    ticks=ticks,
                    k=ki,
                    anneal=anneal,
                    label_k=label_k,
                    fuzzy=fuzzy,
                    birth_rate=birth_rate,
                    death_rate=death_rate,
                    multihop=multihop,
                    hop_discount=hop_discount,
                )
            )
    return rows, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of nodes (default 100; uses bayesian_synth SEM helpers).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Conditioning-set size for CA partial correlations "
            "(default: k=3 if N<=50, k=5 if N>50 — Upgrade 2). "
            "JIT-static; batched inv blocks are (2+k)×(2+k)."
        ),
    )
    parser.add_argument(
        "--k-sweep",
        type=_parse_k_sweep,
        default=None,
        metavar="Ks",
        help=(
            "Comma-separated conditioning sizes to compare on one SEM bundle "
            "(e.g. 3,4,5). Overrides --k for the CA row(s); labels each k."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override sample count (default: synth helper, 200*n for n>10).",
    )
    parser.add_argument(
        "--golem-iters",
        type=int,
        default=GOLEM_DEFAULT_KWARGS["num_iter"],
        help=(
            "GOLEM num_iter (default 20000; gCastle has no early-stop/tol — "
            "fixed Adam budget only)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Device checks + GOLEM import/init only (no full learn / no timings).",
    )
    parser.add_argument(
        "--ca-only",
        action="store_true",
        help="Skip PC / GOLEM baselines; time and score Bayesian CA (JAX) only.",
    )
    parser.add_argument(
        "--anneal",
        action="store_true",
        help=(
            "Enable Upgrade-1 adaptive thresholds "
            f"(death {ANNEAL_DEATH_START:.2f}→{ANNEAL_DEATH_END:.2f}, "
            f"birth {ANNEAL_BIRTH_START:.2f}→{ANNEAL_BIRTH_END:.2f}). "
            "Default remains static death=0.05 / birth=0.10 (Phase-6 baseline)."
        ),
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help=(
            "Upgrade 3 (opt-in): continuous edge weights A∈[0,1] with soft "
            "birth/death instead of hard 0/1 snaps. Binarize only after the "
            "final tick (A>0.5). Experimental — did not beat binary on the "
            "N=100 seed=0 SEM; default remains binary."
        ),
    )
    parser.add_argument(
        "--compare-fuzzy",
        action="store_true",
        help=(
            "A/B: run binary and --fuzzy CA on the same SEM bundle "
            "(Upgrade 3 comparison table)."
        ),
    )
    parser.add_argument(
        "--birth-rate",
        type=float,
        default=0.5,
        help="Fuzzy birth increment (default 0.5; only with --fuzzy).",
    )
    parser.add_argument(
        "--death-rate",
        type=float,
        default=0.1,
        help="Fuzzy death decay (default 0.1; only with --fuzzy).",
    )
    parser.add_argument(
        "--multihop",
        action="store_true",
        help=(
            "Upgrade 4 Maximum Accuracy Mode (opt-in): 2-hop elevation "
            "gravity via A@A, weighted by --hop-discount (default 1.0). "
            "On N=100 seed=0 k=5, lifts Directed F1 from 0.619 → 0.696 "
            "with flat wall time. Default remains fast 1-hop gravity."
        ),
    )
    parser.add_argument(
        "--compare-multihop",
        action="store_true",
        help=(
            "A/B: run 1-hop and --multihop CA on the same SEM bundle "
            "(Upgrade 4 comparison table)."
        ),
    )
    parser.add_argument(
        "--hop-discount",
        type=float,
        default=1.0,
        help=(
            "Weight on the 2-hop elevation pull when --multihop is set "
            "(default 1.0; tuned for N=100 Directed F1 0.696)."
        ),
    )
    args = parser.parse_args(argv)

    logging.getLogger("castle").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    print_device_checks()
    if not args.ca_only:
        _require_gcastle()

    if args.smoke:
        _require_gcastle()
        smoke_golem_init()
        _p()
        _p(
            "Smoke complete (no GOLEM.learn / no F1 table). "
            "For Colab T4 full numbers:"
        )
        _p('  pip install -U "jax[cuda13]" gcastle torch')
        _p("  python -m causal_automata.benchmark --n 100")
        return 0

    if args.k is not None and args.k < 0:
        raise SystemExit(f"--k must be >= 0, got {args.k}")

    rows, bundle = run_benchmark(
        n=args.n,
        seed=args.seed,
        ticks=args.ticks,
        k=args.k,
        k_sweep=args.k_sweep,
        n_samples=args.samples,
        golem_iters=args.golem_iters,
        ca_only=args.ca_only,
        anneal=args.anneal,
        fuzzy=args.fuzzy,
        compare_fuzzy=args.compare_fuzzy,
        birth_rate=args.birth_rate,
        death_rate=args.death_rate,
        multihop=args.multihop,
        compare_multihop=args.compare_multihop,
        hop_discount=args.hop_discount,
    )
    _p(
        f"N={bundle.n}  seed={bundle.seed}  samples={bundle.X.shape[0]}  "
        f"true_edges={len(bundle.truth_edges)}"
    )
    _p()
    _p(markdown_table(rows))
    _p()
    _p("Details:")
    for r in rows:
        _p(
            f"- {r.algorithm}: dir TP/FP/FN={r.directed_tp}/{r.directed_fp}/{r.directed_fn}  "
            f"skel TP/FP/FN={r.skeleton_tp}/{r.skeleton_fp}/{r.skeleton_fn}  "
            f"({r.notes})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
