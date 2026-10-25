#!/usr/bin/env python3
"""Generate README hero animation assets (MP4 + poster + GIF).

Records a short JAX CA trajectory on a clean 12-node hierarchical DAG and
writes:

* ``assets/elevation_convergence.mp4`` — primary hero (plays on iPhone)
* ``assets/elevation_convergence_poster.jpg`` — tappable poster with play cue
* ``assets/elevation_convergence.gif`` — desktop fallback

GitHub's mobile site often freezes animated GIFs (battery / accessibility),
including third-party READMEs. An H.264 MP4 opened from a poster link is the
reliable cross-device path.

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
from PIL import Image, ImageDraw, ImageFont, ImageSequence

# Repo root on sys.path so `causal_automata` resolves without an editable install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from causal_automata.bayesian_synth import (  # noqa: E402
    edge_metrics,
    empirical_covariance,
    generate_linear_dag_samples,
    skeleton_metrics,
    truth_adjacency,
)
from causal_automata.jax_core import step  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N = 12
TICKS = 50
HOLD_FINAL = 12
SEED = 8
K = 3
FPS = 10
DPI = 110
FIGSIZE = (11, 5)
OUT_GIF = _ROOT / "assets" / "elevation_convergence.gif"
OUT_MP4 = _ROOT / "assets" / "elevation_convergence.mp4"
OUT_POSTER = _ROOT / "assets" / "elevation_convergence_poster.jpg"
BG = "#0f172a"
FG = "#e2e8f0"
MUTED = "#94a3b8"
TP_COLOR = "#10b981"
FP_COLOR = "#ef4444"
MAX_GIF_BYTES = 5 * 1024 * 1024
MAX_MP4_BYTES = 10 * 1024 * 1024
POSTER_FRAME_INDEX = 20  # mid-convergence still for the tappable poster

# 12-node hierarchy (logical order, before random index permutation):
#   3 roots (0,1,2) → 4 intermediates (3,4,6,7) → 3 colliders (5,8,9) → 2 sinks (10,11)
LOGICAL_EDGES: tuple[tuple[int, int], ...] = (
    # roots → intermediates
    (0, 3),
    (1, 3),
    (1, 4),
    (2, 4),
    # intermediate forks / collider into 5
    (3, 5),
    (4, 5),  # collider 5: 3 → 5 ← 4
    (3, 6),
    (4, 7),
    # colliders 8, 9
    (5, 8),
    (6, 8),  # collider 8: 5 → 8 ← 6
    (6, 9),
    (7, 9),  # collider 9: 6 → 9 ← 7
    # sinks
    (8, 10),
    (9, 10),
    (9, 11),
    (8, 11),
)


def build_hero_problem(
    *,
    seed: int = SEED,
    n_samples: int = 12_000,
    weight: float = 2.5,
    noise_scale: float = 0.7,
    edge_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A_true, Cov, A0, E0, perm)`` with randomly permuted indices."""
    rng = np.random.default_rng(seed)
    A_logical = truth_adjacency(N, edges=LOGICAL_EDGES)
    X, _ = generate_linear_dag_samples(
        n_samples=n_samples,
        weight=weight,
        noise_scale=noise_scale,
        rng=rng,
        n=N,
        edges=LOGICAL_EDGES,
    )
    Cov_logical = empirical_covariance(X)

    # Shuffle node indices so the initial state is not trivially ordered.
    perm = rng.permutation(N)
    A_true = A_logical[np.ix_(perm, perm)]
    Cov = Cov_logical[np.ix_(perm, perm)]

    # Dense random upper-triangular init → many coral FPs that die off.
    A0 = np.triu((rng.random((N, N)) < edge_prob).astype(np.float64), k=1)
    E0 = rng.uniform(0.2, 0.8, size=N).astype(np.float64)
    return A_true, Cov, A0, E0, perm


def run_trajectory(
    A0: np.ndarray,
    E0: np.ndarray,
    Cov: np.ndarray,
    *,
    ticks: int = TICKS,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan CA ticks and return histories including the t=0 initial state.

    Instead of discarding intermediates with ``return (A, E), None``, the
    scan body yields ``(A_next, E_next)`` so callers can animate every frame.
    """
    A0_j = jnp.asarray(A0, dtype=jnp.float64)
    E0_j = jnp.asarray(E0, dtype=jnp.float64)
    Cov_j = jnp.asarray(Cov, dtype=jnp.float64)

    def body(carry, _):
        A, E = carry
        A_next, E_next = step(
            A,
            E,
            Cov_j,
            k=k,
            threshold_birth=0.12,
            # Slower than the default 0.1 so the hero GIF lets viewers watch
            # coral FPs die off and nodes glide into strata over ~35–40 ticks.
            learning_rate=0.08,
            multihop=True,
        )
        return (A_next, E_next), (A_next, E_next)

    (_, _), (A_hist, E_hist) = jax.lax.scan(
        body, (A0_j, E0_j), xs=None, length=ticks
    )

    # Prepend tick-0 initial state so the GIF starts unorganized.
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


def compute_metrics(
    A_hist: np.ndarray, A_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tick SHD, skeleton F1, directed F1."""
    T = A_hist.shape[0]
    shd = np.zeros(T, dtype=np.float64)
    sk_f1 = np.zeros(T, dtype=np.float64)
    dir_f1 = np.zeros(T, dtype=np.float64)
    for t in range(T):
        A_t = A_hist[t]
        shd[t] = _structural_hamming(A_t, A_true)
        sk_f1[t] = skeleton_metrics(A_t, A_true).f1
        dir_f1[t] = edge_metrics(A_t, A_true).f1
    return shd, sk_f1, dir_f1


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
        raise RuntimeError("ffmpeg not found on PATH (required for hero MP4)")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = dest_path.parent / ".hero_frames_tmp"
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


def _save_poster(
    frames: list[Image.Image],
    dest_path: Path,
    *,
    frame_index: int = POSTER_FRAME_INDEX,
) -> None:
    """Still frame + play affordance for the README (tap → MP4 on iPhone)."""
    idx = int(np.clip(frame_index, 0, len(frames) - 1))
    base = frames[idx].convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (15, 23, 42, 0))
    draw = ImageDraw.Draw(overlay)

    # Dim vignette so the play control reads clearly on any frame.
    draw.rectangle((0, 0, w, h), fill=(15, 23, 42, 90))

    cx, cy = w // 2, h // 2
    r = max(48, min(w, h) // 8)
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(15, 23, 42, 200),
        outline=(226, 232, 240, 230),
        width=3,
    )
    # Play triangle
    tri = [
        (cx - r // 3, cy - r // 2),
        (cx - r // 3, cy + r // 2),
        (cx + r // 2, cy),
    ]
    draw.polygon(tri, fill=(226, 232, 240, 240))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(18, w // 45))
    except OSError:
        font = ImageFont.load_default()
    caption = "Tap to play  ·  MP4 works on iPhone"
    # Anchor text near the bottom center.
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = (w - tw) // 2, h - th - max(16, h // 30)
    pad = 8
    draw.rounded_rectangle(
        (tx - pad, ty - pad // 2, tx + tw + pad, ty + th + pad // 2),
        radius=6,
        fill=(15, 23, 42, 210),
    )
    draw.text((tx, ty), caption, fill=(226, 232, 240, 255), font=font)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(dest_path, format="JPEG", quality=88, optimize=True)


def render_animation(
    A_hist: np.ndarray,
    E_hist: np.ndarray,
    A_true: np.ndarray,
    *,
    out_gif: Path = OUT_GIF,
    out_mp4: Path = OUT_MP4,
    out_poster: Path = OUT_POSTER,
    hold_final: int = HOLD_FINAL,
    fps: int = FPS,
) -> tuple[Path, Path, Path]:
    """Two-panel dark-theme animation → GIF + MP4 + poster."""
    shd, sk_f1, dir_f1 = compute_metrics(A_hist, A_true)
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
        "causal-automata  ·  synchronous XLA scan  ·  E_t gravity field",
        color=FG,
        fontsize=11,
        y=0.97,
    )

    _style_axes(ax_dag, title="Causal Automaton: Live DAG & Gravity Field E_t")
    _style_axes(ax_elev, title="Elevation Trajectories E_t")
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

    elev_lines = []
    for i in range(N):
        (ln,) = ax_elev.plot([], [], lw=1.15, color=cmap(E_hist[0, i]), alpha=0.85)
        elev_lines.append(ln)
    ax_elev.set_xlim(0, n_ticks - 1)
    ax_elev.set_ylim(-0.05, 1.05)
    ax_elev.set_ylabel("E_t[i]", fontsize=8, color=MUTED)
    ax_elev.set_xlabel("tick", fontsize=8, color=MUTED)

    (shd_line,) = ax_met.plot([], [], color="#38bdf8", lw=1.6, label="SHD")
    (sk_line,) = ax_f1.plot([], [], color="#a78bfa", lw=1.4, label="Skeleton F1")
    (dir_line,) = ax_f1.plot([], [], color=TP_COLOR, lw=1.4, label="Directed F1")
    ax_met.set_xlim(0, n_ticks - 1)
    ax_met.set_ylim(0, max(float(shd.max()) * 1.15, 1.0))
    ax_met.set_xlabel("tick", fontsize=8, color=MUTED)
    ax_met.set_ylabel("SHD", fontsize=8, color=MUTED)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1", fontsize=8, color=MUTED)
    ax_met.legend(
        [shd_line, sk_line, dir_line],
        ["SHD", "Skeleton F1", "Directed F1"],
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
        return (scat, tick_text, shd_line, sk_line, dir_line, *elev_lines, *labels_txt)

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
        dir_line.set_data(xs, dir_f1[: frame_t + 1])

        tick_text.set_text(
            f"t = {frame_t}   SHD = {int(shd[frame_t])}   "
            f"Dir F1 = {dir_f1[frame_t]:.2f}"
        )
        return (scat, tick_text, shd_line, sk_line, dir_line, *elev_lines, *labels_txt)

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
    print(f"  writing poster → {out_poster}")
    _save_poster(frames, out_poster)
    return out_gif, out_mp4, out_poster


def main() -> int:
    print("Building 12-node hierarchical DAG (permuted indices)…")
    A_true, Cov, A0, E0, perm = build_hero_problem()
    n_true = int(A_true.sum())
    print(f"  true edges: {n_true}  |  perm: {perm.tolist()}")
    print(f"  init edges: {int(A0.sum())}  |  ticks={TICKS}  k={K}")

    print("Recording trajectory via jax.lax.scan…")
    A_hist, E_hist = run_trajectory(A0, E0, Cov, ticks=TICKS, k=K)
    shd, sk_f1, dir_f1 = compute_metrics(A_hist, A_true)
    print(
        f"  final SHD={int(shd[-1])}  "
        f"skel F1={sk_f1[-1]:.3f}  dir F1={dir_f1[-1]:.3f}"
    )
    print(
        f"  elevations: min={E_hist[-1].min():.3f}  max={E_hist[-1].max():.3f}  "
        f"std={E_hist[-1].std():.3f}"
    )

    print("Rendering hero assets (GIF + MP4 + poster)…")
    gif_path, mp4_path, poster_path = render_animation(A_hist, E_hist, A_true)
    for label, path, limit in (
        ("GIF", gif_path, MAX_GIF_BYTES),
        ("MP4", mp4_path, MAX_MP4_BYTES),
        ("poster", poster_path, MAX_MP4_BYTES),
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
