"""Visualize CFG denoising trajectories (x_T -> x_0).

Auto-loads config from the checkpoint folder so schedule/num_clusters stay in sync.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule, p_sample_loop_with_traj

CFG_SCALES   = [0.0, 1.0, 3.0, 7.0]
N_PER_CLASS  = 2
RECORD_EVERY = 50

# ── CLI ───────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", default=None,
                     help="path to model.pt (default: latest run in outputs/)")
_args = _parser.parse_args()

if _args.model is not None:
    MODEL_PATH = Path(_args.model)
else:
    _runs = sorted(Path("outputs").glob("*/model.pt"))
    MODEL_PATH = _runs[-1] if _runs else Path("model.pt")

# Load config from checkpoint folder
_cfg_file = MODEL_PATH.parent / "config.json"
_ckpt_cfg = json.loads(_cfg_file.read_text()) if _cfg_file.exists() else {}

T            = _ckpt_cfg.get("T",            1000)
NUM_CLUSTERS = _ckpt_cfg.get("num_clusters", len(ShapeDataset.SHAPE_NAMES))
HIDDEN_DIM   = _ckpt_cfg.get("hidden_dim",   128)
NUM_LAYERS   = _ckpt_cfg.get("num_layers",   4)
EMB_DIM      = _ckpt_cfg.get("emb_dim",      64)
SCHEDULE     = _ckpt_cfg.get("schedule",     "cosine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"model:  {MODEL_PATH}  (schedule={SCHEDULE}, K={NUM_CLUSTERS})")

# ── Load model ────────────────────────────────────────────────────
model = DiffusionMLP(
    hidden_dim   = HIDDEN_DIM,
    num_layers   = NUM_LAYERS,
    num_clusters = NUM_CLUSTERS,
    emb_dim      = EMB_DIM,
    dropout      = 0.0,
).to(device)

state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
emb_w = state["cluster_emb.weight"]
if emb_w.shape[0] == NUM_CLUSTERS:
    print("Old checkpoint - padding null-class embedding, using class-avg unconditional.")
    state["cluster_emb.weight"] = torch.cat(
        [emb_w, torch.zeros(1, emb_w.shape[1])], dim=0
    )
    cfg_trained = False
else:
    cfg_trained = True
model.load_state_dict(state)
model.eval()

betas, alphas, alphas_cumprod, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Reference data (background) ──────────────────────────────────
_ds        = ShapeDataset(num_samples=4000, noise=0.01)
_bg_pts    = _ds.points.numpy()
_bg_labels = _ds.labels.numpy()

# ── Fixed starting noise ──────────────────────────────────────────
torch.manual_seed(42)
N        = N_PER_CLASS * NUM_CLUSTERS
c_labels = torch.arange(NUM_CLUSTERS, device=device).repeat_interleave(N_PER_CLASS)
x_start  = torch.randn(N, 2, device=device)

# ── Sample trajectories ───────────────────────────────────────────
print("Sampling denoising trajectories...")
all_trajs: dict[float, list] = {}
for w in CFG_SCALES:
    print(f"  w = {w}")
    _, traj = p_sample_loop_with_traj(
        model, c_labels, T, betas, alphas, alphas_cumprod,
        w=w, record_every=RECORD_EVERY, x_start=x_start,
        cfg_trained=cfg_trained,
    )
    all_trajs[w] = traj

# ── Plot ──────────────────────────────────────────────────────────
SHAPE_NAMES = ShapeDataset.SHAPE_NAMES if NUM_CLUSTERS == 4 else [str(i) for i in range(NUM_CLUSTERS)]
TAB10 = plt.cm.tab10.colors

fig, axes = plt.subplots(1, len(CFG_SCALES), figsize=(5.5 * len(CFG_SCALES), 5.5))

for ax, w in zip(axes, CFG_SCALES):
    traj    = all_trajs[w]
    pts_arr = np.stack([p.numpy() for _, p in traj], axis=0)  # (steps, N, 2)

    ax.scatter(_bg_pts[:, 0], _bg_pts[:, 1], c=_bg_labels, cmap="tab10",
               s=4, alpha=0.07, linewidths=0, zorder=0)

    for i in range(N):
        cid  = c_labels[i].item()
        rgb  = TAB10[cid % 10][:3]
        path = pts_arr[:, i, :]
        segs = np.stack([path[:-1], path[1:]], axis=1)
        lc   = LineCollection(segs,
                              colors=[(*rgb, a) for a in np.linspace(0.10, 0.90, len(segs))],
                              linewidths=1.4, zorder=2)
        ax.add_collection(lc)
        ax.scatter(*path[0],  marker="x", s=45, color="#888888", zorder=3, linewidths=1.2)
        ax.scatter(*path[-1], s=65, color=rgb, edgecolors="#222222", linewidths=0.7, zorder=4)

    label = {0.0: "unconditional", 1.0: "standard"}.get(w, f"guidance x{w}")
    ax.set_title(f"w = {w}  ({label})", fontsize=11)
    ax.autoscale()
    pad = 0.3
    xl, xr = ax.get_xlim(); yl, yr = ax.get_ylim()
    ax.set_xlim(xl - pad, xr + pad); ax.set_ylim(yl - pad, yr + pad)
    ax.set_aspect("equal")
    ax.axis("off")

fig.suptitle("CFG Denoising Trajectories  (x_T -> x_0)", fontsize=13)
plt.tight_layout()

out = "cfg_trajectories.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
