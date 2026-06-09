"""0000 → 5 targets: all denoising steps overlaid in one cell.

Saturation & alpha increase as t decreases (t=200 low-sat → t=0 high-sat).
Left col: generation overlay. Right col: GT.
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

CLS_SRC = 0
TARGETS = [1, 4, 6, 9, 11]

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

src     = ds.points[ds.labels == CLS_SRC][:N_POINTS].to(device)
src_np  = src.numpy()

# ── Per-point hue from angle ──────────────────────────────────────
centroid = src_np.mean(axis=0)
angles   = np.arctan2(src_np[:, 1] - centroid[1],
                       src_np[:, 0] - centroid[0])
hues     = (angles + np.pi) / (2 * np.pi)   # (N,) in [0, 1]


def make_colors(hues: np.ndarray, sat: float, val: float = 1.0) -> np.ndarray:
    """Convert hue array + scalar sat/val to RGBA."""
    N    = len(hues)
    hsv  = np.stack([hues, np.full(N, sat), np.full(N, val)], axis=1)
    rgba = mcolors.hsv_to_rgb(hsv)
    return rgba   # (N, 3)


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
    snaps = [(int(seq[0]), x.cpu().numpy().copy())]
    for i in range(N_STEPS):
        x = ddim_step(x, int(seq[i]), int(seq[i + 1]), cls_tgt)
        snaps.append((int(seq[i + 1]), x.cpu().numpy().copy()))
    return snaps   # list of (timestep, pts)


# ── Run all targets ───────────────────────────────────────────────
print("Running DDIM (t_start=200, w=1.5)...")
results = {}
for tgt in TARGETS:
    print(f"  0000 → {names[tgt]}")
    results[tgt] = run_ddim(src, tgt)

# ── Saturation / alpha schedule ───────────────────────────────────
# step 0 (t=200): sat_min/alpha_min  →  step 20 (t=0): sat_max/alpha_max
SAT_MIN, SAT_MAX   = 0.10, 1.00
ALPHA_MIN, ALPHA_MAX = 0.12, 0.90


def step_style(step_idx: int, total: int):
    p   = step_idx / total          # 0 → 1
    sat = SAT_MIN + p * (SAT_MAX - SAT_MIN)
    alp = ALPHA_MIN + p * (ALPHA_MAX - ALPHA_MIN)
    return sat, alp


# ── Plot ──────────────────────────────────────────────────────────
n_rows = len(TARGETS)
n_cols = 2   # gen | GT
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 3.2 * n_rows))
fig.suptitle(
    f"0000 → targets  (t_start={T_START}, w={W})\n"
    "low-sat = t=200  →  high-sat = t=0",
    fontsize=10, y=1.01,
)

TAB20 = plt.cm.tab20.colors

for row, tgt in enumerate(TARGETS):
    snaps    = results[tgt]
    tgt_pts  = ds.points[ds.labels == tgt][:N_POINTS].numpy()
    tgt_color = TAB20[tgt % len(TAB20)]

    # ── Left: generation overlay ──────────────────────────────────
    ax = axes[row, 0]
    ax.set_facecolor("white")
    for si, (ts, pts) in enumerate(snaps):
        sat, alp = step_style(si, N_STEPS)
        cols = make_colors(hues, sat)
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=cols, s=5, alpha=alp, linewidths=0, rasterized=True)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_ylabel(f"0000 → {names[tgt]}", fontsize=8, labelpad=3)
    if row == 0:
        ax.set_title("generation\n(overlay)", fontsize=9)

    # ── Right: GT ─────────────────────────────────────────────────
    ax = axes[row, 1]
    ax.set_facecolor("white")
    ax.scatter(tgt_pts[:, 0], tgt_pts[:, 1],
               color=tgt_color, s=5, alpha=0.8, linewidths=0)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    if row == 0:
        ax.set_title("GT\n(target)", fontsize=9)

plt.tight_layout(pad=0.4, h_pad=0.5)
out = "overlay_0000_to_targets.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
