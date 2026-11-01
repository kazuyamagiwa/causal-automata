#!/usr/bin/env python3
"""Generate README hero animation assets (GIF + MP4).

Storyboard — Phase-2 N=10 diamond motif, edgeless start ``A0 = 0`` under
**v0.3.6** dynamics (progressive ``s_max``, closed-form ``s=1`` screens,
endpoint-local ``s≥2``, V-well / anti-sink):

* Fixed top-to-bottom topological layout of the ground-truth DAG.
* Faint grey "ghost" true-DAG arrows under every frame (including Tick 0).
* Estimated edges colored by correctness:
  - solid green/cyan  — correct direction over a true edge
  - solid amber/gold  — reversed true-skeleton edge (still flipping)
  - dashed coral/red  — transitive / confounded false positive
* Nodes colored by live elevation ``E[i](t)`` with a Root↔Sink colorbar.
* Top scoreboard: ``Tick | s_max | Skeleton F1 | Pattern F1 | SHD``.

Writes:

* ``assets/elevation_convergence.gif`` — primary README embed
* ``assets/elevation_convergence.mp4`` — iPhone / reduced-motion fallback

Usage::

    python scripts/generate_hero_gif.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageSequence

# Repo root on sys.path so `causal_automata` resolves without an editable install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from causal_automata.bayesian import dag_to_cpdag, fisher_z_threshold  # noqa: E402
from causal_automata.bayesian_synth import (  # noqa: E402
    edge_metrics,
    prepare_synthetic_bundle,
    skeleton_metrics,
)
from causal_automata.jax_core import enable_x64, step  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N = 10
# Animate through progressive s=1→s=2→s=3; motif typically converges by tick 5.
TICKS = 8
HOLD_FINAL_SEC = 2.5
SEED = 42
K = 3
N_SAMPLES = 1000
FPS = 4  # slow enough to read each birth/death phase
DPI = 120
FIGSIZE = (10.5, 7.2)
LEARNING_RATE = 0.10
MULTIHOP = False
OUT_GIF = _ROOT / "assets" / "elevation_convergence.gif"
OUT_MP4 = _ROOT / "assets" / "elevation_convergence.mp4"

BG = "#0b1220"
PANEL = "#111827"
FG = "#e2e8f0"
MUTED = "#94a3b8"
GHOST = "#94a3b8"
CORRECT = "#22d3ee"  # cyan — covers grey ghost when correct
REVERSED = "#fbbf24"  # amber — right skeleton, wrong direction
FALSE_POS = "#f87171"  # coral — transitive / confounded chord
GHOST_ALPHA = 0.22

MAX_GIF_BYTES = 5 * 1024 * 1024
MAX_MP4_BYTES = 10 * 1024 * 1024

THRESHOLD_DEATH = fisher_z_threshold(N_SAMPLES, k_cond=K, alpha=0.01)
THRESHOLD_BIRTH = 1.2 * THRESHOLD_DEATH
HOLD_FINAL = max(int(round(HOLD_FINAL_SEC * FPS)), 1)


def build_hero_problem(
    *,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A_true, Cov, A0=0, E0≈0.5)`` for the Phase-2 N=10 motif."""
    bundle = prepare_synthetic_bundle(
        n=N,
        seed=seed,
        permute_nodes=True,
        init_mode="corr_flat",
        skeleton_threshold=float(THRESHOLD_BIRTH),
        elevation_noise=0.0,
    )
    # Edgeless start so Tick 0 shows only the grey ground-truth ghost.
    A0 = np.zeros((N, N), dtype=np.float64)
    # Exact plateau — dynamics on this motif need the flat start for the
    # birth → FP prune → amber flip → perfect storyboard (seed=42, k=3).
    E0 = np.full(N, 0.5, dtype=np.float64)
    return bundle.A_true, bundle.Cov, A0, E0


def run_trajectory(
    A0: np.ndarray,
    E0: np.ndarray,
    Cov: np.ndarray,
    *,
    ticks: int = TICKS,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan CA ticks and return histories including the t=0 initial state."""
    A0_j = jnp.asarray(A0, dtype=jnp.float64)
    E0_j = jnp.asarray(E0, dtype=jnp.float64)
    Cov_j = jnp.asarray(Cov, dtype=jnp.float64)
    ticks_j = jnp.arange(ticks, dtype=jnp.int32)

    def body(carry, tick):
        A, E = carry
        A_next, E_next = step(
            A,
            E,
            Cov_j,
            k=k,
            threshold_birth=THRESHOLD_BIRTH,
            threshold_death=THRESHOLD_DEATH,
            learning_rate=LEARNING_RATE,
            multihop=MULTIHOP,
            tick=tick,
        )
        return (A_next, E_next), (A_next, E_next)

    (_, _), (A_hist, E_hist) = jax.lax.scan(body, (A0_j, E0_j), xs=ticks_j)

    A_full = np.concatenate(
        [np.asarray(A0_j)[None, ...], np.asarray(A_hist)], axis=0
    )
    E_full = np.concatenate(
        [np.asarray(E0_j)[None, ...], np.asarray(E_hist)], axis=0
    )
    return A_full, E_full


def _structural_hamming(A_pred: np.ndarray, A_true: np.ndarray) -> int:
    """Directed SHD: extras + missings + reversals (each reversal once)."""
    pred = (A_pred > 0.5) & ~np.eye(A_pred.shape[0], dtype=bool)
    true = (A_true > 0) & ~np.eye(A_true.shape[0], dtype=bool)
    only_pred = pred & ~true
    only_true = ~pred & true
    rev = only_pred & only_true.T
    n_rev = int(np.sum(np.triu(rev | rev.T, k=1)))
    n_extra = int(np.sum(only_pred) - np.sum(rev))
    n_missing = int(np.sum(only_true) - np.sum(rev))
    return n_extra + n_missing + n_rev


def _n_directed_edges(A: np.ndarray) -> int:
    return int(np.sum((A > 0.5) & ~np.eye(A.shape[0], dtype=bool)))


def compute_metrics(
    A_hist: np.ndarray, A_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-tick SHD, skeleton F1, directed F1, pattern F1, edge count, s_max."""
    T = A_hist.shape[0]
    cpdag_true = dag_to_cpdag(A_true)
    shd = np.zeros(T, dtype=np.float64)
    sk_f1 = np.zeros(T, dtype=np.float64)
    dir_f1 = np.zeros(T, dtype=np.float64)
    pat_f1 = np.zeros(T, dtype=np.float64)
    n_edges = np.zeros(T, dtype=np.int32)
    s_max = np.zeros(T, dtype=np.int32)
    for t in range(T):
        A_t = A_hist[t]
        shd[t] = _structural_hamming(A_t, A_true)
        sk_f1[t] = skeleton_metrics(A_t, A_true).f1
        dir_f1[t] = edge_metrics(A_t, A_true).f1
        pat_f1[t] = edge_metrics(dag_to_cpdag(A_t), cpdag_true).f1
        n_edges[t] = _n_directed_edges(A_t)
        # Frame 0 is the initial state (no death yet); frames 1.. use tick=0..
        tick = max(t - 1, 0)
        s_max[t] = min(tick + 1, K) if K > 0 else 0
    return shd, sk_f1, dir_f1, pat_f1, n_edges, s_max


def topological_layered_layout(A_true: np.ndarray) -> np.ndarray:
    """Non-overlapping top-to-bottom layout from longest-path layers of A_true.

    Returns ``xy`` with shape ``(N, 2)`` in axes coordinates; y decreases from
    roots (high) to sinks (low).
    """
    n = A_true.shape[0]
    true = (A_true > 0.5) & ~np.eye(n, dtype=bool)
    # Longest-path layering (roots = layer 0).
    layer = np.zeros(n, dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for j in range(n):
            parents = np.flatnonzero(true[:, j])
            if parents.size == 0:
                continue
            cand = int(layer[parents].max()) + 1
            if cand > layer[j]:
                layer[j] = cand
                changed = True

    max_layer = int(layer.max()) if n else 0
    xy = np.zeros((n, 2), dtype=np.float64)
    for L in range(max_layer + 1):
        nodes = np.flatnonzero(layer == L)
        # Stable left-to-right by original index for reproducibility.
        nodes = np.sort(nodes)
        m = len(nodes)
        xs = np.linspace(0.12, 0.88, m) if m > 1 else np.array([0.5])
        # Mild stagger so stacked layers don't look like a ladder.
        xs = xs + (0.02 if L % 2 else -0.02) * (m > 1)
        y = 0.92 - (0.80 * L / max(max_layer, 1))
        for k, node in enumerate(nodes):
            xy[node, 0] = float(xs[k])
            xy[node, 1] = float(y)
    return xy


def _force_full_frame_delta(im: Image.Image, frame_idx: int) -> Image.Image:
    """Nudge opposite corners so Pillow cannot crop to a partial dirty-rect."""
    out = im.copy()
    px = out.load()
    assert px is not None
    w, h = out.size
    r, g, b = px[0, 0]
    px[0, 0] = (r, g, (b + 1 + (frame_idx % 3)) % 256)
    r2, g2, b2 = px[w - 1, h - 1]
    px[w - 1, h - 1] = (r2, g2, (b2 + 1 + (frame_idx % 5)) % 256)
    return out


def _load_rgb_frames(src_path: Path, *, hold_final: int) -> list[Image.Image]:
    """Composite PillowWriter output into full RGB frames (+ end hold)."""
    im = Image.open(src_path)
    frames: list[Image.Image] = []
    for frame in ImageSequence.Iterator(im):
        frames.append(frame.convert("RGB"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {src_path}")
    last = frames[-1]
    frames.extend(last.copy() for _ in range(hold_final))
    return frames


def _save_safari_safe_gif(
    frames: list[Image.Image],
    dest_path: Path,
    *,
    fps: int = FPS,
) -> None:
    """Write a full-canvas opaque GIF (disposal=1) for desktop browsers."""
    duration_ms = max(int(round(1000 / fps)), 1)
    stamped = [_force_full_frame_delta(fr, i) for i, fr in enumerate(frames)]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    stamped[0].save(
        dest_path,
        format="GIF",
        save_all=True,
        append_images=stamped[1:],
        duration=[duration_ms] * len(stamped),
        loop=0,
        disposal=1,
        optimize=False,
    )


def _save_mp4(frames: list[Image.Image], dest_path: Path, *, fps: int = FPS) -> None:
    """Encode H.264 MP4 via ffmpeg (widest iPhone / GitHub playback)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH (required for README MP4)")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = dest_path.parent / ".elevation_frames_tmp"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    try:
        for i, fr in enumerate(frames):
            fr.save(frame_dir / f"frame_{i:04d}.png")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dest_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def _classify_edge(
    i: int, j: int, A_true: np.ndarray
) -> str:
    """Return ``correct`` | ``reversed`` | ``false_positive`` for estimated i→j."""
    if A_true[i, j] > 0.5:
        return "correct"
    if A_true[j, i] > 0.5:
        return "reversed"
    return "false_positive"


def _arrow(
    xy: np.ndarray,
    i: int,
    j: int,
    *,
    color: str,
    alpha: float,
    lw: float,
    style: str,
    zorder: int,
    rad: float = 0.0,
    mutation_scale: float = 14,
) -> FancyArrowPatch:
    """Fancy arrow between fixed layout positions, shortened toward nodes."""
    p0 = xy[i].copy()
    p1 = xy[j].copy()
    vec = p1 - p0
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color, alpha=0.0)
    unit = vec / dist
    # Keep arrowheads clear of node disks.
    shrink = 0.045
    p0s = p0 + unit * shrink
    p1s = p1 - unit * shrink
    return FancyArrowPatch(
        p0s,
        p1s,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        lw=lw,
        color=color,
        alpha=alpha,
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
    )


def render_animation(
    A_hist: np.ndarray,
    E_hist: np.ndarray,
    A_true: np.ndarray,
    *,
    out_gif: Path = OUT_GIF,
    out_mp4: Path = OUT_MP4,
    hold_final: int = HOLD_FINAL,
    fps: int = FPS,
) -> tuple[Path, Path]:
    """Single-panel layered DAG with ghost true edges → GIF + MP4."""
    shd, sk_f1, dir_f1, pat_f1, n_edges, s_max = compute_metrics(A_hist, A_true)
    n_ticks = A_hist.shape[0]
    xy = topological_layered_layout(A_true)
    elev_cmap = mpl.colormaps["coolwarm"]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#334155",
            "text.color": FG,
            "axes.labelcolor": FG,
        }
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.06, 0.14, 0.72, 0.76])
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.suptitle(
        "From Empty Grid to True Causal DAG   ·   causal-automata v0.3.6",
        color=FG,
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    subtitle = fig.text(
        0.42,
        0.915,
        "Phase-2 N=10 motif  ·  A₀ = 0  ·  grey ghost = ground-truth target",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=9,
    )

    scoreboard = fig.text(
        0.42,
        0.875,
        "",
        ha="center",
        va="top",
        color=FG,
        fontsize=10,
        family="DejaVu Sans Mono",
        fontweight="bold",
    )

    # --- Ghost true-DAG arrows (static, all frames) -------------------------
    true_ij = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if A_true[i, j] > 0.5
    ]
    ghost_artists = []
    for i, j in true_ij:
        art = _arrow(
            xy,
            i,
            j,
            color=GHOST,
            alpha=GHOST_ALPHA,
            lw=2.4,
            style="-",
            zorder=2,
            rad=0.05,
            mutation_scale=16,
        )
        ax.add_patch(art)
        ghost_artists.append(art)

    # --- Live estimated edges + nodes --------------------------------------
    edge_artists: list = []

    def _clear_edges() -> None:
        nonlocal edge_artists
        for art in edge_artists:
            art.remove()
        edge_artists = []

    def _draw_estimated(A_t: np.ndarray) -> None:
        _clear_edges()
        for i in range(N):
            for j in range(N):
                if A_t[i, j] <= 0.5:
                    continue
                kind = _classify_edge(i, j, A_true)
                if kind == "correct":
                    color, style, alpha, lw, z, rad = (
                        CORRECT,
                        "-",
                        0.95,
                        2.2,
                        4,
                        0.08,
                    )
                elif kind == "reversed":
                    color, style, alpha, lw, z, rad = (
                        REVERSED,
                        "-",
                        0.95,
                        2.0,
                        4,
                        -0.12,
                    )
                else:
                    color, style, alpha, lw, z, rad = (
                        FALSE_POS,
                        "--",
                        0.85,
                        1.6,
                        3,
                        0.18,
                    )
                art = _arrow(
                    xy,
                    i,
                    j,
                    color=color,
                    alpha=alpha,
                    lw=lw,
                    style=style,
                    zorder=z,
                    rad=rad,
                    mutation_scale=13,
                )
                ax.add_patch(art)
                edge_artists.append(art)

    # Stretch color limits to the live elevation range so Root↔Sink
    # stratification is readable (absolute E often stays in a narrow band).
    e_lo = float(E_hist.min())
    e_hi = float(E_hist.max())
    if e_hi - e_lo < 1e-3:
        e_lo, e_hi = 0.0, 1.0
    else:
        pad = 0.05 * (e_hi - e_lo)
        e_lo, e_hi = e_lo - pad, e_hi + pad

    scat = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=E_hist[0],
        cmap=elev_cmap,
        vmin=e_lo,
        vmax=e_hi,
        s=320,
        zorder=6,
        edgecolors="#0f172a",
        linewidths=1.4,
    )
    labels_txt = [
        ax.text(
            xy[i, 0],
            xy[i, 1],
            str(i),
            ha="center",
            va="center",
            fontsize=9,
            color="#0f172a",
            fontweight="bold",
            zorder=7,
        )
        for i in range(N)
    ]

    # Colorbar for elevation.
    cax = fig.add_axes([0.82, 0.28, 0.02, 0.48])
    cbar = fig.colorbar(scat, cax=cax)
    cbar.set_label("Node Elevation E(t)", color=FG, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED, labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    cax.text(
        0.5,
        1.06,
        "Root\n(High)",
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    cax.text(
        0.5,
        -0.06,
        "Sink\n(Low)",
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
    )

    # Compact legend.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=GHOST,
            lw=2.4,
            alpha=0.55,
            label="Grey = True DAG Target",
        ),
        Line2D([0], [0], color=CORRECT, lw=2.2, label="Cyan = Correct Edge"),
        Line2D(
            [0], [0], color=REVERSED, lw=2.0, label="Amber = Reversed (Flipping)"
        ),
        Line2D(
            [0],
            [0],
            color=FALSE_POS,
            lw=1.6,
            ls="--",
            label="Dashed Red = Transitive FP (Pruning)",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        edgecolor="#334155",
        facecolor="#0f172a",
        labelcolor=FG,
        borderpad=0.6,
        handlelength=2.4,
    )
    leg.get_frame().set_alpha(0.92)

    phase_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=FG,
        fontweight="bold",
        zorder=8,
    )

    def _phase_label(frame_t: int) -> str:
        if frame_t == 0:
            return "Tick 0 · Edgeless start (A₀ = 0)"
        if frame_t == 1:
            return "Tick 1 · Parallel birth (candidates spark)"
        if sk_f1[frame_t] >= 0.999 and pat_f1[frame_t] >= 0.999:
            return "Converged · Grey ghost fully covered"
        return f"Tick {frame_t} · Progressive death / V-well (s_max={int(s_max[frame_t])})"

    def init():
        _draw_estimated(A_hist[0])
        return (scat, scoreboard, phase_text, subtitle, *labels_txt)

    def update(frame_t: int):
        E_t = E_hist[frame_t]
        A_t = A_hist[frame_t]
        scat.set_array(E_t)
        _draw_estimated(A_t)
        scoreboard.set_text(
            f"Tick {frame_t}  |  s_max={int(s_max[frame_t])}  |  "
            f"Edges={int(n_edges[frame_t])}  |  "
            f"Skeleton F1={sk_f1[frame_t]:.3f}  |  "
            f"Pattern F1={pat_f1[frame_t]:.3f}  |  "
            f"SHD={int(shd[frame_t])}"
        )
        phase_text.set_text(_phase_label(frame_t))
        return (scat, scoreboard, phase_text, *labels_txt)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=list(range(n_ticks)),
        interval=1000 // fps,
        blit=False,
    )

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_gif.with_suffix(".tmp.gif")
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(tmp_path), writer=writer, dpi=DPI)
    plt.close(fig)

    frames = _load_rgb_frames(tmp_path, hold_final=hold_final)
    tmp_path.unlink(missing_ok=True)

    print(f"  writing GIF  → {out_gif}")
    _save_safari_safe_gif(frames, out_gif, fps=fps)
    print(f"  writing MP4  → {out_mp4}")
    _save_mp4(frames, out_mp4, fps=fps)
    return out_gif, out_mp4


def main() -> int:
    enable_x64()
    print(
        f"Building Phase-2 N={N} motif (seed={SEED}, A0=0, "
        f"flat E0=0.5, permute_nodes=True)…"
    )
    A_true, Cov, A0, E0 = build_hero_problem(seed=SEED)
    n_true = int(A_true.sum())
    print(f"  true edges: {n_true}")
    print(
        f"  init directed edges: {_n_directed_edges(A0)}  |  "
        f"E0 flat max|E-0.5|={float(np.max(np.abs(E0 - 0.5))):.2e}"
    )
    print(
        f"  ticks={TICKS}  k={K}  lr={LEARNING_RATE}  "
        f"multihop={MULTIHOP}  thr_d={THRESHOLD_DEATH:.4f}  "
        f"thr_b={THRESHOLD_BIRTH:.4f}"
    )

    print("Recording trajectory via jax.lax.scan (progressive tick)…")
    A_hist, E_hist = run_trajectory(A0, E0, Cov, ticks=TICKS, k=K)
    print(
        f"  E0→E_final spread: std {float(E_hist[0].std()):.2e} → "
        f"{float(E_hist[-1].std()):.3f}  "
        f"(min={float(E_hist[-1].min()):.3f}, max={float(E_hist[-1].max()):.3f})"
    )
    shd, sk_f1, dir_f1, pat_f1, n_edges, s_max_arr = compute_metrics(A_hist, A_true)
    for t in range(A_hist.shape[0]):
        print(
            f"  t={t} s_max={int(s_max_arr[t])} edges={int(n_edges[t])}  "
            f"skel={sk_f1[t]:.3f} pat={pat_f1[t]:.3f} SHD={int(shd[t])}"
        )
    print(
        f"  final SHD={int(shd[-1])}  "
        f"skel F1={sk_f1[-1]:.3f}  dir F1={dir_f1[-1]:.3f}  "
        f"pattern F1={pat_f1[-1]:.3f}  edges={int(n_edges[-1])}"
    )
    if sk_f1[-1] < 0.999 or pat_f1[-1] < 0.999:
        print(
            "WARNING: hero trajectory did not reach perfect recovery; "
            "consider retuning SEED / TICKS.",
            file=sys.stderr,
        )

    print("Rendering animation assets (GIF + MP4)…")
    gif_path, mp4_path = render_animation(A_hist, E_hist, A_true)
    for label, path, limit in (
        ("GIF", gif_path, MAX_GIF_BYTES),
        ("MP4", mp4_path, MAX_MP4_BYTES),
    ):
        size = path.stat().st_size
        print(f"  {label}: {path} ({size / 1024:.1f} KiB)")
        if size > limit:
            print(
                f"WARNING: {label} is {size / limit:.2f}× the size budget "
                f"({limit / 1024 / 1024:.0f} MB)",
                file=sys.stderr,
            )
            return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
