#!/usr/bin/env python3
"""Generate README convergence-animation assets (GIF + MP4).

Records a short JAX CA trajectory on the Phase-2 N=10 motif with the synth
``corr_flat`` default init (symmetric ``|R|`` skeleton + near-flat
``E0 ≈ 0.5``) under **v0.3.6** dynamics (progressive ``s_max``, closed-form
``s=1`` screens, endpoint-local ``s≥2``, V-well / anti-sink) and writes:

* ``assets/elevation_convergence.gif`` — primary README embed
* ``assets/elevation_convergence.mp4`` — iPhone / reduced-motion fallback

Each frame overlays ``Tick t | s_max | Edges | Skeleton F1 | Pattern F1``.
GitHub often freezes animated GIFs when accessibility reduced-motion is on —
the MP4 is a tap-to-play fallback.

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
TICKS = 100
HOLD_FINAL = 14
SEED = 42
K = 3
N_SAMPLES = 1000  # Fisher-z calibration for the hero SEM
FPS = 10
DPI = 110
FIGSIZE = (11, 5)
LEARNING_RATE = 0.10
# Multihop compresses the elevation field; leave off so the plateau → strata
# separation stays visible in the README clip.
MULTIHOP = False
OUT_GIF = _ROOT / "assets" / "elevation_convergence.gif"
OUT_MP4 = _ROOT / "assets" / "elevation_convergence.mp4"
BG = "#0f172a"
FG = "#e2e8f0"
MUTED = "#94a3b8"
TP_COLOR = "#10b981"
FP_COLOR = "#ef4444"
PLATEAU_COLOR = "#64748b"
MAX_GIF_BYTES = 5 * 1024 * 1024
MAX_MP4_BYTES = 10 * 1024 * 1024

THRESHOLD_DEATH = fisher_z_threshold(N_SAMPLES, k_cond=K, alpha=0.01)
THRESHOLD_BIRTH = 1.2 * THRESHOLD_DEATH


def build_hero_problem(
    *,
    seed: int = SEED,
    skeleton_threshold: float = THRESHOLD_BIRTH,
    elevation_noise: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A_true, Cov, A0, E0)`` via synth ``corr_flat`` defaults."""
    bundle = prepare_synthetic_bundle(
        n=N,
        seed=seed,
        permute_nodes=True,
        init_mode="corr_flat",
        skeleton_threshold=skeleton_threshold,
        elevation_noise=elevation_noise,
    )
    return bundle.A_true, bundle.Cov, bundle.A_init, bundle.E_init


def run_trajectory(
    A0: np.ndarray,
    E0: np.ndarray,
    Cov: np.ndarray,
    *,
    ticks: int = TICKS,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan CA ticks and return histories including the t=0 initial state.

    Passes progressive ``tick`` so ``s_max = min(tick + 1, k)`` matches the
    v0.3.6 death schedule used by :class:`~causal_automata.CausalAutomaton`.
    """
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

    # Prepend tick-0 plateau so the GIF starts flat / undirected.
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


def _n_undirected_edges(A: np.ndarray) -> int:
    U = ((A > 0.5) | (A.T > 0.5)) & ~np.eye(A.shape[0], dtype=bool)
    return int(np.sum(np.triu(U, k=1)))


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
        n_edges[t] = _n_undirected_edges(A_t)
        # Frame 0 is the initial state (no death yet); frames 1.. use tick=0..
        tick = max(t - 1, 0)
        s_max[t] = min(tick + 1, K) if K > 0 else 0
    return shd, sk_f1, dir_f1, pat_f1, n_edges, s_max


def _node_x_coords(n: int = N) -> np.ndarray:
    """Even horizontal spread with a slight deterministic offset (no layout lib)."""
    x = np.linspace(0.05, 0.95, n)
    offset = 0.03 * np.sin(np.arange(n) * 1.7)
    return x + offset


def _style_axes(ax: plt.Axes, *, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _force_full_frame_delta(im: Image.Image, frame_idx: int) -> Image.Image:
    """Nudge opposite corners so Pillow cannot crop to a partial dirty-rect.

    Even with ``optimize=False``, Pillow's GIF writer still crops each frame to
    the bbox of pixels that changed vs the previous frame. Differences at both
    ``(0,0)`` and ``(w-1,h-1)`` force a full-canvas write. Invisible in practice
    on a 1210×550 dark plot (1-level channel wobble).
    """
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
    """Two-panel dark-theme animation → GIF + MP4."""
    shd, sk_f1, dir_f1, pat_f1, n_edges, s_max = compute_metrics(A_hist, A_true)
    n_ticks = A_hist.shape[0]  # ticks+1 (includes t=0)

    x_coords = _node_x_coords(N)
    cmap = mpl.colormaps["plasma"]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#334155",
            "text.color": FG,
            "axes.labelcolor": FG,
        }
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.30,
        hspace=0.38,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.11,
    )
    ax_dag = fig.add_subplot(gs[:, 0])
    ax_elev = fig.add_subplot(gs[0, 1])
    ax_met = fig.add_subplot(gs[1, 1])
    ax_f1 = ax_met.twinx()

    fig.suptitle(
        "causal-automata v0.3.6  ·  corr_flat init  ·  E≈0.5 plateau → strata",
        color=FG,
        fontsize=11,
        y=0.97,
    )

    _style_axes(ax_dag, title="Causal Automaton: Live DAG & Gravity Field E_t")
    _style_axes(ax_elev, title="Elevation Trajectories E_t (flat → separated)")
    _style_axes(ax_met, title="Structural Convergence")
    ax_f1.set_facecolor(BG)
    ax_f1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax_f1.spines.values():
        spine.set_color("#334155")

    ax_dag.set_xlim(-0.05, 1.05)
    ax_dag.set_ylim(-0.08, 1.12)
    ax_dag.set_xlabel("Node index (permuted)", fontsize=8, color=MUTED)
    ax_dag.set_ylabel("Topological Elevation E_t", fontsize=8, color=MUTED)
    ax_dag.set_xticks(x_coords)
    ax_dag.set_xticklabels([str(i) for i in range(N)], fontsize=7)
    ax_dag.axhline(
        0.5,
        color=PLATEAU_COLOR,
        lw=0.9,
        ls="--",
        alpha=0.55,
        zorder=1,
    )

    elev_lines = []
    for i in range(N):
        (ln,) = ax_elev.plot([], [], lw=1.15, color=cmap(E_hist[0, i]), alpha=0.85)
        elev_lines.append(ln)
    ax_elev.axhline(
        0.5,
        color=PLATEAU_COLOR,
        lw=0.9,
        ls="--",
        alpha=0.55,
        zorder=1,
    )
    ax_elev.set_xlim(0, n_ticks - 1)
    ax_elev.set_ylim(-0.05, 1.05)
    ax_elev.set_ylabel("E_t[i]", fontsize=8, color=MUTED)
    ax_elev.set_xlabel("tick", fontsize=8, color=MUTED)

    (shd_line,) = ax_met.plot([], [], color="#38bdf8", lw=1.6, label="SHD")
    (sk_line,) = ax_f1.plot([], [], color="#a78bfa", lw=1.4, label="Skeleton F1")
    (pat_line,) = ax_f1.plot([], [], color=TP_COLOR, lw=1.4, label="Pattern F1")
    ax_met.set_xlim(0, n_ticks - 1)
    ax_met.set_ylim(0, max(float(shd.max()) * 1.15, 1.0))
    ax_met.set_xlabel("tick", fontsize=8, color=MUTED)
    ax_met.set_ylabel("SHD", fontsize=8, color=MUTED)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1", fontsize=8, color=MUTED)
    ax_met.legend(
        [shd_line, sk_line, pat_line],
        ["SHD", "Skeleton F1", "Pattern F1"],
        loc="upper right",
        fontsize=7,
        frameon=False,
        labelcolor=MUTED,
    )

    tick_text = ax_dag.text(
        0.02,
        0.98,
        "",
        transform=ax_dag.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
        family="DejaVu Sans Mono",
    )

    scat = ax_dag.scatter(
        x_coords,
        E_hist[0],
        c=E_hist[0],
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        s=90,
        zorder=5,
        edgecolors=BG,
        linewidths=0.6,
    )
    labels_txt = [
        ax_dag.text(
            x_coords[i],
            E_hist[0, i] + 0.045,
            str(i),
            ha="center",
            va="bottom",
            fontsize=7,
            color=FG,
            zorder=6,
        )
        for i in range(N)
    ]

    edge_artists: list = []

    def _clear_edges() -> None:
        nonlocal edge_artists
        for art in edge_artists:
            art.remove()
        edge_artists = []

    def _draw_edges(A_t: np.ndarray, E_t: np.ndarray) -> None:
        _clear_edges()
        for i in range(N):
            for j in range(N):
                if A_t[i, j] <= 0.5:
                    continue
                is_tp = A_true[i, j] > 0.5
                color = TP_COLOR if is_tp else FP_COLOR
                style = "-" if is_tp else "--"
                alpha = 0.90 if is_tp else 0.40
                arrow = FancyArrowPatch(
                    (x_coords[i], E_t[i]),
                    (x_coords[j], E_t[j]),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    lw=1.15,
                    color=color,
                    alpha=alpha,
                    linestyle=style,
                    connectionstyle="arc3,rad=0.08",
                    zorder=3,
                )
                ax_dag.add_patch(arrow)
                edge_artists.append(arrow)

    def init():
        _draw_edges(A_hist[0], E_hist[0])
        return (scat, tick_text, shd_line, sk_line, pat_line, *elev_lines, *labels_txt)

    def update(frame_t: int):
        E_t = E_hist[frame_t]
        A_t = A_hist[frame_t]
        scat.set_offsets(np.column_stack([x_coords, E_t]))
        scat.set_array(E_t)
        for i, txt in enumerate(labels_txt):
            txt.set_position((x_coords[i], E_t[i] + 0.045))
        _draw_edges(A_t, E_t)

        xs = np.arange(frame_t + 1)
        for i, ln in enumerate(elev_lines):
            ln.set_data(xs, E_hist[: frame_t + 1, i])
            ln.set_color(cmap(float(E_t[i])))

        shd_line.set_data(xs, shd[: frame_t + 1])
        sk_line.set_data(xs, sk_f1[: frame_t + 1])
        pat_line.set_data(xs, pat_f1[: frame_t + 1])

        tick_text.set_text(
            f"Tick {frame_t}  |  s_max={int(s_max[frame_t])}  |  "
            f"Edges={int(n_edges[frame_t])}  |  "
            f"Skeleton F1={sk_f1[frame_t]:.3f}  |  "
            f"Pattern F1={pat_f1[frame_t]:.3f}"
        )
        return (scat, tick_text, shd_line, sk_line, pat_line, *elev_lines, *labels_txt)

    # Render ticks only here; exporters append the end hold.
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
        f"Building Phase-2 N={N} motif (seed={SEED}, corr_flat, "
        f"permute_nodes=True)…"
    )
    A_true, Cov, A0, E0 = build_hero_problem(
        skeleton_threshold=float(THRESHOLD_BIRTH)
    )
    n_true = int(A_true.sum())
    print(f"  true edges: {n_true}")
    print(
        f"  init undirected pairs: {int(A0.sum()) // 2}  |  "
        f"A0 symmetric={bool(np.allclose(A0, A0.T))}  |  "
        f"E0≈0.5 max|E-0.5|={float(np.max(np.abs(E0 - 0.5))):.2e}"
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
    shd, sk_f1, dir_f1, pat_f1, n_edges, _ = compute_metrics(A_hist, A_true)
    print(
        f"  final SHD={int(shd[-1])}  "
        f"skel F1={sk_f1[-1]:.3f}  dir F1={dir_f1[-1]:.3f}  "
        f"pattern F1={pat_f1[-1]:.3f}  edges={int(n_edges[-1])}"
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
