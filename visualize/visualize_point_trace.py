"""Single-point trajectory trace for t=200, w=1.5 (circle → star, DDIM 20 steps).

Grid of all 21 snapshots with one tracked point highlighted,
plus a final overlay panel showing the full trajectory path.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule

# ── Load ──────────────────────────────────────────────────────────
MODEL_PATH = sorted(Path("outputs").glob("*/best.pt"))[-1]
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())

T        = cfg_data["T"]
SCHEDULE = cfg_data.get("schedule", "cosine")
K        = cfg_data["num_clusters"]
NAMES    = ShapeDataset.SHAPE_NAMES
CLS_STAR = NAMES.index("star")

device = torch.device("cpu")
model  = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"], num_layers=cfg_data["num_layers"],
    num_clusters=K, emb_dim=cfg_data["emb_dim"], dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

_, _, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Data ──────────────────────────────────────────────────────────
N_POINTS = 400
N_STEPS  = 20
T_START  = 200
W        = 1.5

torch.manual_seed(0)
ds     = ShapeDataset(num_samples=60000, noise=0.01, outline=True)
CLS_CIRC = NAMES.index("circle")
circle = ds.points[ds.labels == CLS_CIRC][:N_POINTS].to(device)

# ── DDIM ──────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev, w):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_s = torch.full((B,), CLS_STAR,        device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class, device=device, dtype=torch.long)
    eps = model(x_t, t, c_n) + w * (model(x_t, t, c_s) - model(x_t, t, c_n))
    ab_t = alpha_bars[t_cur]
    x0   = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    x0   = x0.clamp(-2.0, 2.0)
    if t_prev <= 0:
        return x0
    ab_p = alpha_bars[t_prev]
    return ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps

seq   = np.linspace(T_START, 0, N_STEPS + 1).round().astype(int)
x     = circle.clone()
snaps = [x.cpu().numpy().copy()]
for i in range(N_STEPS):
    x = ddim_step(x, int(seq[i]), int(seq[i + 1]), W)
    snaps.append(x.cpu().numpy().copy())

# ── Pick a tracked point ──────────────────────────────────────────
# choose the point that ends up farthest from circle center (most "moved")
displacements = np.linalg.norm(snaps[-1] - snaps[0], axis=1)
TRACK_IDX = int(np.argmax(displacements))
print(f"Tracking point {TRACK_IDX}: "
      f"start={snaps[0][TRACK_IDX]}, end={snaps[-1][TRACK_IDX]}")

track_pos = np.stack([s[TRACK_IDX] for s in snaps])   # (21, 2)

# ── Colors ────────────────────────────────────────────────────────
TAB10  = plt.cm.tab10.colors
C_CIRC = np.array(TAB10[CLS_CIRC][:3])
C_STAR = np.array(TAB10[CLS_STAR][:3])
C_TRACK = np.array([1.0, 0.6, 0.0])   # orange

def lerp(a, b, p):
    return tuple((1 - p) * a + p * b)

# ── Figure 1: grid of all 21 steps ───────────────────────────────
n_cols = 7
n_rows = 3   # 7×3 = 21 panels
cs     = 1.6

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(cs * n_cols, cs * n_rows + 0.6),
                          squeeze=False)
fig.suptitle(f"Circle → Star  (t_start={T_START}, w={W})  —  orange = tracked point",
             fontsize=10, y=1.002)

for panel, si in enumerate(range(N_STEPS + 1)):
    ri, ci = divmod(panel, n_cols)
    ax     = axes[ri, ci]
    pts    = snaps[si]
    p      = si / N_STEPS
    col    = lerp(C_CIRC, C_STAR, p)

    # all other points
    mask = np.ones(N_POINTS, dtype=bool)
    mask[TRACK_IDX] = False
    ax.scatter(pts[mask, 0], pts[mask, 1],
               color=col, s=3, alpha=0.6, linewidths=0, zorder=1)

    # tracked point
    ax.scatter(pts[TRACK_IDX, 0], pts[TRACK_IDX, 1],
               color=C_TRACK, s=40, zorder=3, edgecolors="black", linewidths=0.5)

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"step {si}  t={seq[si]}", fontsize=6.5, pad=2)

plt.tight_layout(pad=0.15, h_pad=0.3, w_pad=0.1)
out1 = "point_trace_grid.png"
fig.savefig(out1, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out1}")

# ── Figure 2: trajectory overlay on final point cloud ────────────
fig2, ax2 = plt.subplots(figsize=(5, 5))

# final point cloud as background
final_pts = snaps[-1]
mask = np.ones(N_POINTS, dtype=bool)
mask[TRACK_IDX] = False
ax2.scatter(final_pts[mask, 0], final_pts[mask, 1],
            color=C_STAR, s=5, alpha=0.35, linewidths=0, zorder=1)

# trajectory path with color gradient (blue→orange over time)
cmap = plt.cm.plasma
for i in range(N_STEPS):
    p   = i / (N_STEPS - 1)
    col = cmap(p)
    ax2.plot(track_pos[i:i+2, 0], track_pos[i:i+2, 1],
             color=col, lw=1.8, zorder=2)

# waypoint dots
for si in range(N_STEPS + 1):
    p   = si / N_STEPS
    col = cmap(p)
    ax2.scatter(track_pos[si, 0], track_pos[si, 1],
                color=col, s=18, zorder=3, edgecolors="white", linewidths=0.4)
    ax2.text(track_pos[si, 0] + 0.03, track_pos[si, 1] + 0.03,
             str(seq[si]), fontsize=5, color=col, zorder=4)

# start / end markers
ax2.scatter(*track_pos[0],  s=80, color=C_CIRC, zorder=5,
            edgecolors="black", linewidths=0.8, label="start (circle)")
ax2.scatter(*track_pos[-1], s=80, marker="*", color=C_STAR, zorder=5,
            edgecolors="black", linewidths=0.5, label="end (star)")

ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
ax2.set_aspect("equal"); ax2.axis("off")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_title(f"Tracked point trajectory  (t_start={T_START}, w={W})", fontsize=10)

plt.tight_layout()
out2 = "point_trace_path.png"
fig2.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2}")
print("Done.")
