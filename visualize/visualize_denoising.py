"""Denoising trajectory visualization (DDIM 20 steps, CFG → star).

Saves 4 images:
  denoising_noise_to_star.png      x_T ~ N(0,I)  → star
  denoising_sdedit_50.png          circle + 50%T noise → star
  denoising_sdedit_70.png          circle + 70%T noise → star
  denoising_sdedit_90.png          circle + 90%T noise → star

Each image: rows = w values (1, 3, 7)
            cols = timestep snapshots (start + 20 DDIM steps)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule, q_sample

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH = sorted(Path("outputs").glob("*/model.pt"))[-1]
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())
print(f"model : {MODEL_PATH}")

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

betas, alphas, alpha_bars, sqac, s1mac = make_noise_schedule(
    T, schedule=SCHEDULE, device=device
)

# ── Settings ──────────────────────────────────────────────────────
N_STEPS   = 20
N_POINTS  = 300
W_VALUES  = [1, 3, 7]
T_INJECTS = [0.5, 0.7, 0.9]   # fractions of T

# ── Circle samples ────────────────────────────────────────────────
torch.manual_seed(42)
ds     = ShapeDataset(num_samples=50000, noise=0.01, outline=True)
circle = ds.points[ds.labels == CLS_CIRC][:N_POINTS].to(device)

# ── Pure Gaussian starting noise (same seed for all w) ───────────
x_noise = torch.randn(N_POINTS, 2, device=device)

# ── DDIM step with CFG → star ─────────────────────────────────────
@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev, w):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_s = torch.full((B,), CLS_STAR,         device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class,  device=device, dtype=torch.long)

    eps_u = model(x_t, t, c_n)
    eps_c = model(x_t, t, c_s)
    eps   = eps_u + w * (eps_c - eps_u)

    ab_t  = alpha_bars[t_cur]
    x0    = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()

    if t_prev == 0:
        return x0

    ab_p = alpha_bars[t_prev]
    return ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps


@torch.no_grad()
def run_ddim(x_start, t_start, w):
    """20 DDIM steps from t_start → 0, records N_STEPS+1 snapshots."""
    seq   = np.linspace(t_start, 0, N_STEPS + 1).round().astype(int)
    x     = x_start.clone()
    snaps = [x.cpu().numpy().copy()]
    for i in range(N_STEPS):
        x = ddim_step(x, int(seq[i]), int(seq[i + 1]), w)
        snaps.append(x.cpu().numpy().copy())
    return snaps, seq   # list of N_STEPS+1 arrays, timestep sequence


# ── Figure builder ────────────────────────────────────────────────
TAB10  = plt.cm.tab10.colors
C_STAR = np.array(TAB10[CLS_STAR][:3])
C_GRAY = np.array([0.60, 0.60, 0.60])


def make_figure(all_snaps, all_seqs, row_labels, title):
    n_rows = len(all_snaps)
    n_cols = N_STEPS + 1      # start snapshot + 20 steps
    cs     = 1.2              # cell size in inches

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cs * n_cols, cs * n_rows + 0.55),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=9, y=1.005)

    for ri, (snaps, seq, label) in enumerate(zip(all_snaps, all_seqs, row_labels)):
        for ci, pts in enumerate(snaps):
            ax = axes[ri, ci]

            # Color fades from gray (noise) to star red (clean)
            p   = ci / (n_cols - 1)
            col = tuple((1 - p) * C_GRAY + p * C_STAR)

            ax.scatter(pts[:, 0], pts[:, 1],
                       color=col, s=1.8, alpha=0.75, linewidths=0)
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            # Timestep label (bottom row only)
            if ri == n_rows - 1:
                ax.text(0.5, -0.04, f"t={seq[ci]}",
                        transform=ax.transAxes,
                        ha="center", va="top", fontsize=5.0)

        # w label (left column)
        axes[ri, 0].set_ylabel(label, fontsize=8, labelpad=3)

    plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.10)
    return fig


# ── 1. Noise → Star ───────────────────────────────────────────────
print("Sampling: Noise -> Star ...")
noise_snaps, noise_seqs = [], []
for w in W_VALUES:
    print(f"  w={w}")
    snaps, seq = run_ddim(x_noise, T - 1, w)
    noise_snaps.append(snaps)
    noise_seqs.append(seq)

fig = make_figure(
    noise_snaps, noise_seqs,
    [f"w = {w}" for w in W_VALUES],
    "Noise -> Star  (DDIM 20 steps, CFG)",
)
fig.savefig("denoising_noise_to_star.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved -> denoising_noise_to_star.png")

# ── 2-4. Circle + SDEdit → Star ───────────────────────────────────
for t_frac in T_INJECTS:
    t_inject = int(t_frac * (T - 1))
    pct      = int(t_frac * 100)
    print(f"Sampling: Circle -> Star  (T_inject={t_inject}, {pct}%) ...")

    t_tens     = torch.full((N_POINTS,), t_inject, device=device, dtype=torch.long)
    x_injected, _ = q_sample(circle, t_tens, sqac, s1mac)

    sde_snaps, sde_seqs = [], []
    for w in W_VALUES:
        print(f"  w={w}")
        snaps, seq = run_ddim(x_injected, t_inject, w)
        sde_snaps.append(snaps)
        sde_seqs.append(seq)

    fig = make_figure(
        sde_snaps, sde_seqs,
        [f"w = {w}" for w in W_VALUES],
        f"Circle -> Star  (SDEdit T_inject={t_inject} [{pct}%T], DDIM 20 steps, CFG)",
    )
    out = f"denoising_sdedit_{pct}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")

print("Done.")
