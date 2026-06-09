"""Circle → Star via CFG denoising (DDIM 20 steps).

Circle outline points are treated AS x_t (no noise injection).
CFG guides denoising toward star from varying assumed timesteps.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule

# ── Load best checkpoint ──────────────────────────────────────────
MODEL_PATH = sorted(Path("outputs").glob("*/best.pt"))[-1]
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())
print(f"checkpoint: {MODEL_PATH}")

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

# ── Samples ───────────────────────────────────────────────────────
N_POINTS = 400
N_STEPS  = 20
W_VALUES = [1.5, 3.0, 7.0]
T_STARTS = [200, 400, 600, 800]   # assume circle is x_t at this timestep

torch.manual_seed(0)
ds     = ShapeDataset(num_samples=60000, noise=0.01, outline=True)
circle = ds.points[ds.labels == CLS_CIRC][:N_POINTS].to(device)
star   = ds.points[ds.labels == CLS_STAR][:N_POINTS].to(device)

# ── DDIM step with CFG → star ─────────────────────────────────────
@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev, w):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_s = torch.full((B,), CLS_STAR,        device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class, device=device, dtype=torch.long)

    eps_u = model(x_t, t, c_n)
    eps_c = model(x_t, t, c_s)
    eps   = eps_u + w * (eps_c - eps_u)

    ab_t = alpha_bars[t_cur]
    x0   = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    x0   = x0.clamp(-2.0, 2.0)

    if t_prev <= 0:
        return x0

    ab_p = alpha_bars[t_prev]
    return ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps


@torch.no_grad()
def run_ddim(x_start, t_start, w):
    seq   = np.linspace(t_start, 0, N_STEPS + 1).round().astype(int)
    x     = x_start.clone()
    snaps = [x.cpu().numpy().copy()]
    for i in range(N_STEPS):
        x = ddim_step(x, int(seq[i]), int(seq[i + 1]), w)
        snaps.append(x.cpu().numpy().copy())
    return snaps, seq


# ── Colors ────────────────────────────────────────────────────────
TAB10  = plt.cm.tab10.colors
C_CIRC = np.array(TAB10[CLS_CIRC][:3])
C_STAR = np.array(TAB10[CLS_STAR][:3])


def lerp(a, b, p):
    return tuple((1 - p) * a + p * b)


# ── Run ───────────────────────────────────────────────────────────
print("Running DDIM denoising (circle as x_t, no injection)...")
results = {}
for t_start in T_STARTS:
    for w in W_VALUES:
        print(f"  t_start={t_start}, w={w}")
        snaps, seq = run_ddim(circle, t_start, w)
        results[(t_start, w)] = (snaps, seq)

# ── Figure: rows = t_start, cols = [circle | step0..20 | star_ref] ──
SHOW_STEPS = [0, 4, 8, 12, 16, 20]
n_rows = len(T_STARTS) * len(W_VALUES)
n_cols = 1 + len(SHOW_STEPS) + 1   # circle | snaps | star ref
cs     = 1.4

print("Rendering...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(cs * n_cols, cs * n_rows + 0.5),
                          squeeze=False)
fig.suptitle("Circle (as x_t) → Star via CFG DDIM 20 steps", fontsize=10, y=1.002)

row = 0
for t_start in T_STARTS:
    for w in W_VALUES:
        snaps, seq = results[(t_start, w)]

        # col 0: original circle
        ax = axes[row, 0]
        ax.scatter(circle[:, 0].numpy(), circle[:, 1].numpy(),
                   color=C_CIRC, s=2, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_ylabel(f"t={t_start}, w={w}", fontsize=7, labelpad=2)
        if row == 0:
            ax.set_title("circle\n(x_t)", fontsize=7)

        # cols 1..N: denoising snapshots
        for ci, si in enumerate(SHOW_STEPS):
            ax  = axes[row, ci + 1]
            pts = snaps[si]
            p   = si / N_STEPS
            col = lerp(C_CIRC, C_STAR, p)
            ax.scatter(pts[:, 0], pts[:, 1], color=col, s=2, alpha=0.75, linewidths=0)
            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            # timestep label per row (seq differs by t_start)
            ax.text(0.5, -0.02, f"t={seq[si]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6)
            if row == 0:
                ax.set_title(f"step {si}", fontsize=7)

        # last col: star reference
        ax = axes[row, -1]
        ax.scatter(star[:, 0].numpy(), star[:, 1].numpy(),
                   color=C_STAR, s=2, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if row == 0:
            ax.set_title("star\n(ref)", fontsize=7)

        row += 1

plt.tight_layout(pad=0.15, h_pad=0.2, w_pad=0.1)
out = "circle2star_cfg.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
print("Done.")
