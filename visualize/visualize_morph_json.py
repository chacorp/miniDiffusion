"""0000 (민소매) → 0009 (티셔츠) via CFG DDIM 20 steps.

Source shape points are treated as x_t (SDEdit style).
CFG guides denoising toward target class.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import JsonShapeDataset
from model import DiffusionMLP, make_noise_schedule

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH = Path("outputs/20260608_222600/best.pt")
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())

T        = cfg_data["T"]
SCHEDULE = cfg_data.get("schedule", "cosine")
K        = 12

CLS_SRC = 0   # 0000 민소매
CLS_TGT = 9   # 0009 티셔츠

N_POINTS = 400
N_STEPS  = 20
W_VALUES = [1.5, 3.0, 7.0]
T_STARTS = [200, 400, 600, 800]

device = torch.device("cpu")
model  = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"],
    num_layers=cfg_data["num_layers"],
    num_clusters=K,
    emb_dim=cfg_data["emb_dim"],
    dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"checkpoint: {MODEL_PATH}")

_, _, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Dataset samples ───────────────────────────────────────────────
torch.manual_seed(0)
ds  = JsonShapeDataset(data_dir=cfg_data["data_dir"], num_samples=60000)
src = ds.points[ds.labels == CLS_SRC][:N_POINTS].to(device)
tgt = ds.points[ds.labels == CLS_TGT][:N_POINTS].to(device)

src_name = ds.shape_names[CLS_SRC]
tgt_name = ds.shape_names[CLS_TGT]
print(f"source: {src_name} ({src.shape[0]} pts)  target: {tgt_name} ({tgt.shape[0]} pts)")

# ── DDIM step with CFG toward target ─────────────────────────────
@torch.no_grad()
def ddim_step(x_t, t_cur, t_prev, w):
    B   = x_t.shape[0]
    t   = torch.full((B,), t_cur,  device=device, dtype=torch.long)
    c_t = torch.full((B,), CLS_TGT,         device=device, dtype=torch.long)
    c_n = torch.full((B,), model.null_class, device=device, dtype=torch.long)

    eps_u = model(x_t, t, c_n)
    eps_c = model(x_t, t, c_t)
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
TAB20 = plt.cm.tab20.colors
C_SRC = np.array(TAB20[CLS_SRC % len(TAB20)][:3])
C_TGT = np.array(TAB20[CLS_TGT % len(TAB20)][:3])


def lerp(a, b, p):
    return tuple((1 - p) * a + p * b)


# ── Run ───────────────────────────────────────────────────────────
print("Running DDIM denoising...")
results = {}
for t_start in T_STARTS:
    for w in W_VALUES:
        print(f"  t_start={t_start}, w={w}")
        snaps, seq = run_ddim(src, t_start, w)
        results[(t_start, w)] = (snaps, seq)

# ── Figure ────────────────────────────────────────────────────────
SHOW_STEPS = [0, 4, 8, 12, 16, 20]
n_rows = len(T_STARTS) * len(W_VALUES)
n_cols = 1 + len(SHOW_STEPS) + 1   # src | snaps | tgt ref
cs     = 1.4

print("Rendering...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(cs * n_cols, cs * n_rows + 0.5),
                          squeeze=False)
fig.suptitle(f"{src_name} (as x_t) → {tgt_name} via CFG DDIM {N_STEPS} steps",
             fontsize=10, y=1.002)

row = 0
for t_start in T_STARTS:
    for w in W_VALUES:
        snaps, seq = results[(t_start, w)]

        # col 0: source
        ax = axes[row, 0]
        ax.scatter(src[:, 0].numpy(), src[:, 1].numpy(),
                   color=C_SRC, s=2, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_ylabel(f"t={t_start}, w={w}", fontsize=7, labelpad=2)
        if row == 0:
            ax.set_title(f"{src_name}\n(x_t)", fontsize=7)

        # cols 1..N: denoising snapshots
        for ci, si in enumerate(SHOW_STEPS):
            ax  = axes[row, ci + 1]
            pts = snaps[si]
            p   = si / N_STEPS
            col = lerp(C_SRC, C_TGT, p)
            ax.scatter(pts[:, 0], pts[:, 1], color=col, s=2, alpha=0.75, linewidths=0)
            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.text(0.5, -0.02, f"t={seq[si]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6)
            if row == 0:
                ax.set_title(f"step {si}", fontsize=7)

        # last col: target reference
        ax = axes[row, -1]
        ax.scatter(tgt[:, 0].numpy(), tgt[:, 1].numpy(),
                   color=C_TGT, s=2, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if row == 0:
            ax.set_title(f"{tgt_name}\n(ref)", fontsize=7)

        row += 1

plt.tight_layout(pad=0.15, h_pad=0.2, w_pad=0.1)
out = "morph_0000_to_0009.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
