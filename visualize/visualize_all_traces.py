"""All-point trajectory overlay (t=200, w=1.5).

Each point has a unique color. All 21 steps are overlaid on one figure,
with alpha scaling from dim (early) to bright (late) to show movement.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
CLS_CIRC = NAMES.index("circle")

device = torch.device("cpu")
model  = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"], num_layers=cfg_data["num_layers"],
    num_clusters=K, emb_dim=cfg_data["emb_dim"], dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

_, _, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Data & DDIM ───────────────────────────────────────────────────
N_POINTS = 400
N_STEPS  = 20
T_START  = 200
W        = 1.5

torch.manual_seed(0)
ds     = ShapeDataset(num_samples=60000, noise=0.01, outline=True)
circle = ds.points[ds.labels == CLS_CIRC][:N_POINTS].to(device)

@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_s = torch.full((B,), CLS_STAR,        device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class, device=device, dtype=torch.long)
    eps = model(x_t, t, c_n) + W * (model(x_t, t, c_s) - model(x_t, t, c_n))
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
    x = ddim_step(x, int(seq[i]), int(seq[i + 1]))
    snaps.append(x.cpu().numpy().copy())

# snaps: list of (N_STEPS+1) arrays, each (N_POINTS, 2)
snaps = np.stack(snaps)   # (N_STEPS+1, N_POINTS, 2)

# ── Per-point colors ──────────────────────────────────────────────
# Assign hue by initial angle on the circle so spatially adjacent
# points get similar colors — makes trails easier to follow.
angles      = np.arctan2(snaps[0, :, 1], snaps[0, :, 0])   # (N_POINTS,)
hue_order   = (angles - angles.min()) / (angles.max() - angles.min() + 1e-8)
point_colors = plt.cm.hsv(hue_order)[:, :3]   # (N_POINTS, 3)  RGB

# ── Alpha schedule: quadratic ramp dim→bright ─────────────────────
alpha_min, alpha_max = 0.12, 0.90
step_alphas = np.linspace(0, 1, N_STEPS + 1) ** 1.5   # (N_STEPS+1,)
step_alphas = alpha_min + (alpha_max - alpha_min) * step_alphas

step_sizes = np.full(N_STEPS + 1, 8.0)

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
ax.set_facecolor("white")

for si in range(N_STEPS + 1):
    pts   = snaps[si]                # (N_POINTS, 2)
    alpha = step_alphas[si]
    size  = step_sizes[si]
    # fill: per-point HSV color with step alpha
    rgba  = np.column_stack([point_colors, np.full(N_POINTS, alpha)])
    # edge: black (step 0) → white (step 20)
    p_edge     = si / N_STEPS
    edge_gray  = (p_edge, p_edge, p_edge)
    ax.scatter(pts[:, 0], pts[:, 1],
               c=rgba, s=size, zorder=si,
               edgecolors=[edge_gray], linewidths=0.5)

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"All-point trajectories  (t_start={T_START}, w={W}, DDIM {N_STEPS} steps)\n"
             f"color = point identity · dim→bright = early→late step",
             fontsize=9, color="black", pad=8)

plt.tight_layout()
out = "all_traces.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")
