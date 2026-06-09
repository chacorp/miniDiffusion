"""0000 → 5 targets with per-point color tracking.

t_start=200, w=1.5, DDIM 20 steps.
Each point keeps its color across all denoising steps.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from dataset import JsonShapeDataset
from model import DiffusionMLP, make_noise_schedule

MODEL_PATH = Path("outputs/20260608_222600/best.pt")
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())

T        = cfg_data["T"]
K        = 12
N_POINTS = 400
T_START  = 200
W        = 1.5
N_STEPS  = 20
SHOW_STEPS = [0, 4, 8, 12, 16, 20]

CLS_SRC  = 0
TARGETS  = [1, 4, 6, 9, 11]

device = torch.device("cpu")
model  = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"], num_layers=cfg_data["num_layers"],
    num_clusters=K, emb_dim=cfg_data["emb_dim"], dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

_, _, alpha_bars, _, _ = make_noise_schedule(T, schedule=cfg_data["schedule"], device=device)

torch.manual_seed(0)
ds    = JsonShapeDataset(data_dir=cfg_data["data_dir"], num_samples=60000)
names = ds.shape_names

# ── Source points (fixed across all rows) ────────────────────────
src_all = ds.points[ds.labels == CLS_SRC]
src     = src_all[:N_POINTS].to(device)

# ── Per-point colors: angle from centroid → HSV colormap ─────────
src_np   = src.numpy()
centroid = src_np.mean(axis=0)
angles   = np.arctan2(src_np[:, 1] - centroid[1],
                       src_np[:, 0] - centroid[0])          # [-π, π]
norm_ang = (angles + np.pi) / (2 * np.pi)                   # [0, 1]
cmap     = plt.cm.hsv
pt_colors = cmap(norm_ang)                                   # (N, 4) RGBA


# ── DDIM step ─────────────────────────────────────────────────────
@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev, cls_tgt):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_t = torch.full((B,), cls_tgt,          device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class,  device=device, dtype=torch.long)

    eps = model(x_t, t, c_n) + W * (model(x_t, t, c_t) - model(x_t, t, c_n))

    ab_t = alpha_bars[t_cur]
    x0   = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    x0   = x0.clamp(-2.0, 2.0)
    if t_prev <= 0:
        return x0
    ab_p = alpha_bars[t_prev]
    return ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps


@torch.no_grad()
def run_ddim(x_start, cls_tgt):
    seq   = np.linspace(T_START, 0, N_STEPS + 1).round().astype(int)
    x     = x_start.clone()
    snaps = [x.cpu().numpy().copy()]
    for i in range(N_STEPS):
        x = ddim_step(x, int(seq[i]), int(seq[i + 1]), cls_tgt)
        snaps.append(x.cpu().numpy().copy())
    return snaps, seq


# ── Run ───────────────────────────────────────────────────────────
print("Running DDIM (t_start=200, w=1.5)...")
results = {}
for tgt in TARGETS:
    print(f"  0000 → {names[tgt]}")
    snaps, seq = run_ddim(src, tgt)
    results[tgt] = (snaps, seq)

# ── Plot ──────────────────────────────────────────────────────────
n_rows = len(TARGETS)
n_cols = 1 + len(SHOW_STEPS) + 1   # src | snaps | tgt ref
cs     = 1.5

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(cs * n_cols, cs * n_rows + 0.4),
                          squeeze=False)
fig.suptitle(f"0000 → targets  (t_start={T_START}, w={W})  — per-point color tracking",
             fontsize=10, y=1.002)

def scatter_tracked(ax, pts, alpha=0.85):
    ax.scatter(pts[:, 0], pts[:, 1],
               c=pt_colors, s=4, alpha=alpha, linewidths=0)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def scatter_ref(ax, pts, color):
    ax.scatter(pts[:, 0], pts[:, 1],
               color=color, s=2, alpha=0.7, linewidths=0)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

TAB20 = plt.cm.tab20.colors

for row, tgt in enumerate(TARGETS):
    snaps, seq = results[tgt]
    tgt_pts    = ds.points[ds.labels == tgt][:N_POINTS].numpy()
    tgt_color  = TAB20[tgt % len(TAB20)]

    # col 0: source
    ax = axes[row, 0]
    scatter_tracked(ax, src_np)
    ax.set_ylabel(f"→ {names[tgt]}", fontsize=8, labelpad=2)
    if row == 0:
        ax.set_title(f"{names[CLS_SRC]}\n(src)", fontsize=8)

    # cols 1..N: denoising snapshots
    for ci, si in enumerate(SHOW_STEPS):
        ax = axes[row, ci + 1]
        scatter_tracked(ax, snaps[si])
        ax.text(0.5, -0.02, f"t={seq[si]}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6)
        if row == 0:
            ax.set_title(f"step {si}", fontsize=8)

    # last col: target reference (single color)
    ax = axes[row, -1]
    scatter_ref(ax, tgt_pts, tgt_color)
    if row == 0:
        ax.set_title("target\n(ref)", fontsize=8)

plt.tight_layout(pad=0.15, h_pad=0.3, w_pad=0.1)
out = "tracking_0000_to_targets.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
